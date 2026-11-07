"""2-D plane-strain elastodynamics with a spectral-element discretisation in
FEniCSx, on a medium whose Lame parameters are a Shinozuka--Deodatis random
field (random_field.py).

Spectral elements = Lagrange elements of degree p on quadrilaterals with the
GLL node variant, integrated with the GLL rule of the same order.  Nodes and
quadrature points then coincide, the consistent mass matrix is diagonal, and
the central-difference update costs one vector assembly per step:

    u^{n+1} = 2 u^n - u^{n-1} + dt^2 M_L^{-1} ( F^n - K(u^n) ),
    M_L = diag(M 1)   (row-sum lumping, exact for GLL collocation).

A Ricker point force excites the medium, a Cerjan taper absorbs the lateral
and bottom boundaries, and the top boundary is kept free (natural BC), so
surface waves and a scattered coda both develop.

The medium is isotropic unless `set_anisotropy()` is called, which gives it a
transversely isotropic fabric (anisotropy.py): Thomsen `epsilon`, `delta` and a
symmetry axis tilted by `theta` from the vertical, each of them a constant or a
field sampled on the same background grid as the Lame parameters.

Run inside the course Docker image `fenicsx-jupyter` (Notebooks/Dockerfile).
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import RegularGridInterpolator

import basix
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector

from .anisotropy import check_stability, phase_velocities, thomsen_stiffness
from .random_field import lame_fields


def ricker(t, f0, t0):
    """Ricker wavelet of dominant frequency f0, delayed by t0."""
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


class SEMWave2D:
    """Explicit SEM solver for one realisation of the random medium."""

    def __init__(self, L=2000.0, H=1200.0, nx=40, ny=24, degree=4,
                 f0=6.0, t_end=0.9, cfl=0.28, sponge=180.0, q_damp=None,
                 alpha_damp=0.0, comm=None):
        """Rayleigh damping C = alpha M + beta K gives the medium an intrinsic
        attenuation, so that the coda decays and the record returns to zero.

        `q_damp` sets beta = 1/(Q 2 pi f0) and is kept explicit through a
        backward-difference velocity -- stable only while beta < dt/2, i.e.
        Q >~ 1/(2 pi f0 dt); `alpha_damp` (the mass term, which damps the low
        frequencies that ring longest) is treated implicitly, so it costs
        nothing in stability. Modal damping is
        zeta(omega) = alpha/(2 omega) + beta omega/2.
        """
        self.L, self.H, self.degree, self.f0, self.t_end = L, H, degree, f0, t_end
        self.cfl, self.sponge, self.q_damp = cfl, sponge, q_damp
        self.force_amp = 1.0e12        # point-source amplitude [N]
        self.fault = None              # set by set_fault(); None -> point force
        self.aniso = None              # set by set_anisotropy(); None -> isotropic
        self.alpha = float(alpha_damp)
        self.beta = 0.0 if q_damp is None else 1.0 / (q_damp * 2.0 * np.pi * f0)
        comm = MPI.COMM_WORLD if comm is None else comm

        self.msh = mesh.create_rectangle(
            comm, [np.array([0.0, 0.0]), np.array([L, H])], [nx, ny],
            mesh.CellType.quadrilateral)

        gll = basix.LagrangeVariant.gll_isaac
        vec_el = basix.ufl.element("Lagrange", "quadrilateral", degree,
                                   lagrange_variant=gll, shape=(2,))
        sca_el = basix.ufl.element("Lagrange", "quadrilateral", degree,
                                   lagrange_variant=gll)
        self.V = fem.functionspace(self.msh, vec_el)
        self.Q = fem.functionspace(self.msh, sca_el)
        # GLL quadrature collocated on the GLL nodes -> diagonal mass
        self.dx = ufl.Measure("dx", domain=self.msh,
                              metadata={"quadrature_rule": "GLL",
                                        "quadrature_degree": 2 * degree - 1})
        self.h = min(L / nx, H / ny)

    # ------------------------------------------------------------------- fault
    def set_fault(self, x_a, x_b, n_sub=24, v_rup=None, rng=None,
                  slip_sigma=0.8, n_mode=6, hypo=0.15, m0=1.0e12, taper=0.15):
        """Kinematic extended source on a 2-D fault discontinuity.

        In plane strain a fault is the segment `x_a` -> `x_b`.  It is split
        into `n_sub` sub-sources; each is a **double couple** in the (slip,
        normal) frame, represented discretely by two force couples one node
        apart, so the far-field radiation pattern is the shear-dislocation
        one rather than that of a point force.

        The slip is *heterogeneous*: a lognormal patch
        `s(xi) = exp(slip_sigma * g(xi))` built from `n_mode` random Fourier
        modes, tapered to zero at both tips over `taper` of the length, and
        renormalised so the scalar moment is `m0`.  The rupture sweeps
        bilaterally from `hypo` (fractional position along the fault) at
        `v_rup`, so each sub-source fires at its own delay: this is what
        gives the source its finite duration and directivity.

        `v_rup=None` fires the whole fault at once (a line source).
        """
        rng = np.random.default_rng() if rng is None else rng
        x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
        seg = x_b - x_a
        length = float(np.hypot(*seg))
        d_hat = seg / length                       # slip direction (in-plane)
        n_hat = np.array([-d_hat[1], d_hat[0]])    # fault normal

        xi = (np.arange(n_sub) + 0.5) / n_sub      # cell centres along the fault
        pts = x_a[None, :] + xi[:, None] * seg[None, :]

        # --- heterogeneous slip patch ---------------------------------------
        g = np.zeros_like(xi)
        for m in range(1, n_mode + 1):
            g += (rng.normal() / m) * np.cos(2.0 * np.pi * m * xi
                                             + rng.uniform(0, 2 * np.pi))
        slip = np.exp(slip_sigma * g)
        edge = np.clip(np.minimum(xi, 1.0 - xi) / max(taper, 1e-9), 0.0, 1.0)
        slip *= np.sin(0.5 * np.pi * edge) ** 2    # smooth taper to zero tips
        slip *= m0 / slip.sum()
        self.fault_slip, self.fault_xi, self.fault_pts = slip, xi, pts

        # --- rupture front ---------------------------------------------------
        delays = np.zeros_like(xi) if v_rup is None else \
            np.abs(xi - hypo) * length / float(v_rup)
        delays -= delays.min()

        # --- discrete double couple: M (d x n + n x d) -----------------------
        xdofs = self.V.tabulate_dof_coordinates()[:, :2]
        arm = self.h / self.degree                 # one GLL node spacing
        dofs, amps, dels = [], [], []
        for p, m_i, td in zip(pts, slip, delays):
            f_i = m_i / arm                        # F * arm = M
            for off, direction, sgn in (( 0.5 * arm * n_hat, d_hat, +1.0),
                                        (-0.5 * arm * n_hat, d_hat, -1.0),
                                        ( 0.5 * arm * d_hat, n_hat, +1.0),
                                        (-0.5 * arm * d_hat, n_hat, -1.0)):
                j = int(np.argmin(np.sum((xdofs - (p + off)) ** 2, axis=1)))
                for comp in (0, 1):
                    dofs.append(2 * j + comp)
                    amps.append(sgn * f_i * direction[comp])
                    dels.append(td)
        self.fault = {"dofs": np.asarray(dofs, dtype=np.int32),
                      "amps": np.asarray(amps, dtype=float),
                      "delays": np.asarray(dels, dtype=float),
                      "length": length, "rise": float(delays.max())}
        return self

    # ------------------------------------------------------------------ medium
    def set_medium(self, rng, n_grid=(321, 193), **rf_kw):
        """Sample the random field on a background grid and interpolate the
        Lame parameters onto the SEM nodes."""
        x1 = np.linspace(0.0, self.L, n_grid[0])
        x2 = np.linspace(0.0, self.H, n_grid[1])
        self.grid = (x1, x2)
        self.fields = lame_fields(x1, x2, rng=rng, **rf_kw)

        self.mu = fem.Function(self.Q, name="mu")
        self.lam = fem.Function(self.Q, name="lambda")
        self.rho = fem.Function(self.Q, name="rho")
        for name, fn in (("mu", self.mu), ("lam", self.lam), ("rho", self.rho)):
            interp = RegularGridInterpolator((x2, x1), self.fields[name],
                                             bounds_error=False, fill_value=None)
            fn.interpolate(lambda x, interp=interp:
                           interp(np.column_stack((np.clip(x[1], 0.0, self.H),
                                                   np.clip(x[0], 0.0, self.L)))))
        self.c_max = float(self.fields["vp"].max())
        self._set_dt()
        if self.aniso is not None:            # re-sample a fabric set earlier
            self.set_anisotropy(**self.aniso_kw)
        return self

    def _set_dt(self):
        """CFL on the fastest phase velocity anywhere in the model.  A fabric
        with Thomsen epsilon > 0 is faster along itself than the isotropic
        Vp by sqrt(1 + 2 epsilon), and that is the speed the time step must
        resolve -- forgetting it is a silent instability late in the run."""
        c_max = self.c_max * getattr(self, "_aniso_cfac", 1.0)
        self.dt = self.cfl * (self.h / self.degree) / c_max
        self.n_steps = int(np.ceil(self.t_end / self.dt))

    # -------------------------------------------------------------- anisotropy
    def set_anisotropy(self, epsilon=0.0, delta=0.0, theta=0.0):
        """Give the medium a transversely isotropic fabric.

        `epsilon`, `delta` (Thomsen) and `theta` (tilt of the symmetry axis
        from the vertical, radians) may each be a number, an (n_x2, n_x1)
        array on the background grid of `set_medium`, or a callable
        f(X1, X2) evaluated on that grid.  The isotropic medium is the case
        epsilon = delta = 0, which is reproduced exactly.

        Must be called after `set_medium` -- the fabric is built on top of the
        random Lame fields, and the time step is tightened to the fastest
        phase velocity it creates.
        """
        if not hasattr(self, "grid"):
            raise RuntimeError("set_medium() before set_anisotropy(): the "
                               "fabric is sampled on the medium's grid")
        self.aniso_kw = dict(epsilon=epsilon, delta=delta, theta=theta)
        x1, x2 = self.grid
        X1, X2 = np.meshgrid(x1, x2, indexing="xy")

        arrays, funcs = {}, {}
        for name, value in (("epsilon", epsilon), ("delta", delta),
                            ("theta", theta)):
            arr = np.asarray(value(X1, X2) if callable(value) else value,
                             dtype=float)
            arrays[name] = np.broadcast_to(arr, X1.shape).copy()
            funcs[name] = self._grid_function(arrays[name], name)

        # the fabric must still be an elastic medium: C11 C33 > C13^2
        c = thomsen_stiffness(self.fields["lam"], self.fields["mu"],
                              arrays["epsilon"], arrays["delta"])
        margins = check_stability(*c)
        if min(margins.values()) <= 0.0:
            raise ValueError(f"epsilon/delta are not a stable TI medium: "
                             f"{margins}")
        self.aniso, self.aniso_fields, self.stiffness = funcs, arrays, c
        # the CFL speed is the fastest qP *phase* velocity over every point
        # and every direction, not sqrt(1 + 2 epsilon) * Vp: for delta > epsilon
        # the an-elliptic qP front bulges past both the axis and the fabric
        v_qp, _ = self.phase_velocity(np.linspace(0.0, np.pi, 73))
        self._aniso_cfac = float(v_qp.max()) / self.c_max
        self._set_dt()
        return self

    def _grid_function(self, array, name):
        """Interpolate a field given on the background grid onto the SEM
        nodes, the way `set_medium` does it for the Lame parameters."""
        x1, x2 = self.grid
        fn = fem.Function(self.Q, name=name)
        interp = RegularGridInterpolator((x2, x1), array,
                                         bounds_error=False, fill_value=None)
        fn.interpolate(lambda x, interp=interp:
                       interp(np.column_stack((np.clip(x[1], 0.0, self.H),
                                               np.clip(x[0], 0.0, self.L)))))
        return fn

    def phase_velocity(self, angle, rho=None):
        """qP and qSV phase velocities [m/s] of the fabric, at `angle` [rad]
        from its symmetry axis -- the quality check of an anisotropic run.

        Returns two arrays of shape (n_x2, n_x1, n_angle): one velocity per
        background-grid point and per direction.
        """
        if self.aniso is None:
            raise RuntimeError("isotropic medium: call set_anisotropy() first")
        rho = self.fields["rho"] if rho is None else rho
        angle = np.asarray(angle, dtype=float)
        c = [np.asarray(x)[..., None] for x in self.stiffness]
        return phase_velocities(*c, np.asarray(rho)[..., None], angle)

    def _sigma(self, w):
        """Plane-strain stress of a displacement field: Hooke, or the tilted
        TI law of anisotropy.py once a fabric has been set.

        The strain is rotated into the fabric frame (e' = R^T e R), the TI law
        is applied there, and the stress is rotated back (sig = R sig' R^T),
        with R the rotation by `theta` -- so a dipping fabric costs one 2x2
        product per quadrature point and no change anywhere else.
        """
        e = ufl.sym(ufl.grad(w))
        if self.aniso is None:
            return 2.0 * self.mu * e + self.lam * ufl.div(w) * ufl.Identity(2)
        c11, c13, c33, c55 = thomsen_stiffness(
            self.lam, self.mu, self.aniso["epsilon"], self.aniso["delta"],
            sqrt=ufl.sqrt)
        th = self.aniso["theta"]
        rot = ufl.as_matrix([[ufl.cos(th), -ufl.sin(th)],
                             [ufl.sin(th), ufl.cos(th)]])
        em = ufl.dot(ufl.transpose(rot), ufl.dot(e, rot))
        sm = ufl.as_matrix([[c11 * em[0, 0] + c13 * em[1, 1], 2.0 * c55 * em[0, 1]],
                            [2.0 * c55 * em[0, 1], c13 * em[0, 0] + c33 * em[1, 1]]])
        return ufl.dot(rot, ufl.dot(sm, ufl.transpose(rot)))

    # ------------------------------------------------------------------- setup
    def _assemble_statics(self, src_xy, rec_xy):
        V, dx = self.V, self.dx
        v = ufl.TestFunction(V)
        self.u = fem.Function(V, name="u")

        self.f_int = fem.form(ufl.inner(self._sigma(self.u),
                                        ufl.sym(ufl.grad(v))) * dx)

        # row-sum lumped mass: M 1 with 1 the all-ones displacement field
        m_form = fem.form(self.rho * ufl.inner(ufl.as_vector([1.0, 1.0]), v) * dx)
        m_vec = assemble_vector(m_form)
        m_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.m = m_vec.array_r.copy()

        # dof bookkeeping: closest node to the source and to each receiver.
        # `rec_xy` is one station or a list of them; station 0 is the one the
        # scalar `trace` refers to.
        xdofs = self.V.tabulate_dof_coordinates()[:, :2]
        pts = np.atleast_2d(np.asarray(rec_xy, dtype=float))
        self.i_src = int(np.argmin(np.sum((xdofs - np.asarray(src_xy)) ** 2, axis=1)))
        self.i_recs = np.array(
            [int(np.argmin(np.sum((xdofs - q) ** 2, axis=1))) for q in pts],
            dtype=int)
        self.i_rec = int(self.i_recs[0])
        self.src_dof = 2 * self.i_src + 1          # vertical force
        self.rec_dof = 2 * self.i_rec + 1
        self.rec_dofs_x = 2 * self.i_recs          # horizontal components
        self.rec_dofs_y = 2 * self.i_recs + 1      # vertical components
        self.rec_xy = xdofs[self.i_recs]           # where the nodes really are
        self.xdofs = xdofs

        # Cerjan absorbing taper on the two sides and the bottom
        d = np.minimum.reduce([xdofs[:, 0], self.L - xdofs[:, 0], xdofs[:, 1]])
        taper = np.ones_like(d)
        inside = d < self.sponge
        taper[inside] = np.exp(-(0.25 * (self.sponge - d[inside]) / self.sponge) ** 2)
        self.taper = np.repeat(taper, 2)

    # -------------------------------------------------------------------- solve
    def run(self, src_xy=None, rec_xy=None, snap_at=0.55, progress=False,
            f0=None):
        """One time loop.  `f0` overrides the source corner frequency for
        this run only, so that one assembled medium can be excited by a
        whole ensemble of sources."""
        src_xy = (self.L * 0.5, self.H - 1.0) if src_xy is None else src_xy
        rec_xy = (self.L * 0.80, self.H - 1.0) if rec_xy is None else rec_xy
        self._assemble_statics(src_xy, rec_xy)

        f0 = self.f0 if f0 is None else float(f0)
        snap_times = np.atleast_1d(np.asarray(snap_at, dtype=float))
        snap_steps = [int(ts / self.dt) for ts in snap_times]
        self.snap_times = snap_times
        self.snapshots = [None] * len(snap_steps)
        n_loc = self.u.x.array.size
        u_old = np.zeros(n_loc)
        u_now = np.zeros(n_loc)
        n_rec = self.i_recs.size
        traces_x = np.zeros((n_rec, self.n_steps))
        traces_y = np.zeros((n_rec, self.n_steps))
        t0 = 1.2 / f0

        for n in range(self.n_steps):
            t = n * self.dt
            # K (u^n + beta (u^n - u^{n-1})/dt): Rayleigh stiffness damping at
            # the cost of no extra assembly
            self.u.x.array[:] = u_now if self.beta == 0.0 else \
                u_now + (self.beta / self.dt) * (u_now - u_old)
            r = assemble_vector(self.f_int)
            r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            force = -r.array_r.copy()
            if self.fault is None:
                force[self.src_dof] += self.force_amp * ricker(t, f0, t0)
            else:
                np.add.at(force, self.fault["dofs"],
                          self.fault["amps"]
                          * ricker(t - self.fault["delays"], f0, t0))

            # implicit mass damping: no stability penalty
            c_lo = 0.5 * self.alpha * self.dt
            u_new = (2.0 * u_now - (1.0 - c_lo) * u_old
                     + self.dt**2 * force / self.m) / (1.0 + c_lo)
            u_new *= self.taper
            # central-difference velocity at step n: seismograms are velocities,
            # and the near-field quasi-static offset drops out
            traces_x[:, n] = (u_new[self.rec_dofs_x]
                              - u_old[self.rec_dofs_x]) / (2.0 * self.dt)
            traces_y[:, n] = (u_new[self.rec_dofs_y]
                              - u_old[self.rec_dofs_y]) / (2.0 * self.dt)
            for j, ns in enumerate(snap_steps):
                if n == ns:
                    self.snapshots[j] = (u_new - u_old) / (2.0 * self.dt)
            u_old, u_now = u_now, u_new
            if progress and n % 200 == 0:
                print(f"  step {n}/{self.n_steps}  t = {t:.3f} s", flush=True)

        # `trace` stays the vertical velocity of station 0, as before;
        # `traces` / `traces_x` carry every station and both components
        self.traces, self.traces_x = traces_y, traces_x
        self.trace = traces_y[0]
        self.time = np.arange(self.n_steps) * self.dt
        if self.snapshots[0] is None:
            self.snapshots[0] = (u_now - u_old) / (2.0 * self.dt)
        self.snapshot = self.snapshots[0]
        return self

    # ------------------------------------------------------------- post-process
    def snapshot_grid(self, n=(260, 160), crop=None, which=0):
        """Interpolate the particle-velocity magnitude of the stored snapshot
        onto a regular grid.  `crop` (default: the sponge width) removes the
        absorbing layer on the two sides and the bottom, so that only the
        physical window is plotted."""
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
        crop = self.sponge if crop is None else crop
        snap = self.snapshots[which] if getattr(self, "snapshots", None) \
            else self.snapshot
        mag = np.linalg.norm(snap.reshape(-1, 2), axis=1)
        gx = np.linspace(crop, self.L - crop, n[0])
        gy = np.linspace(crop, self.H, n[1])
        GX, GY = np.meshgrid(gx, gy)
        # the triangulation of the SEM nodes is the expensive part and does
        # not change between snapshots: build it once per solver
        if getattr(self, "_tri", None) is None:
            self._tri = Delaunay(self.xdofs)
        return gx, gy, LinearNDInterpolator(self._tri, mag, fill_value=0.0)(GX, GY)
