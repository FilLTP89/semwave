"""Antiplane (SH) SEM wave solver with a visco-plastic surficial layer.

Same mesh, same GLL spectral elements, same explicit central-difference update
and the same random medium as sem_wave.py / sem_wave_vp.py, but the unknown is
the out-of-plane displacement u3(x1, x2, t), which in a 2-D medium decouples
from the in-plane (P-SV) motion and obeys the scalar wave equation

    rho d2u3/dt2 = div( mu grad u3 ),      tau = (sig_13, sig_23) = mu grad u3.

That is the wave a surface-to-borehole transfer function is read on: one wave
type, one velocity Vs, so the spectral ratio of two stations on the same
vertical shows the resonance of the soil column instead of a mixture of P, SV
and Rayleigh energy.

The stress pair tau is carried as a state variable at the GLL quadrature
points, exactly as the plane-strain visco-plastic solver carries the Voigt
stress, so the top of the model can follow the Lemaitre--Chaboche law in its
antiplane restriction (chaboche_sh.py):

    assemble f_int(tau^n)  ->  u^{n+1}  ->  de = grad(u^{n+1}-u^n)/2
                           ->  tau^{n+1} = chaboche_sh(tau^n, de)

Rayleigh stiffness damping is kept by adding beta * 2 mu de/dt to the stress
the next assembly sees (same explicit treatment, same limit beta < dt/2).

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

from .chaboche_sh import ChabocheSH, j2_sh
from .random_field import lame_fields
from .sem_wave import ricker


class SEMWaveSH2D:
    """Explicit SEM solver for the antiplane motion of the random medium."""

    force_amp = 1.0e12          # out-of-plane point force; scaled to put the
                                # layer at a prescribed multiple of its yield

    def __init__(self, L=2000.0, H=1200.0, nx=40, ny=24, degree=4,
                 f0=6.0, t_end=0.9, cfl=0.28, sponge=180.0, q_damp=None,
                 alpha_damp=0.0, comm=None):
        self.L, self.H, self.degree, self.f0, self.t_end = L, H, degree, f0, t_end
        self.cfl, self.sponge, self.q_damp = cfl, sponge, q_damp
        self.alpha = float(alpha_damp)
        self.beta = 0.0 if q_damp is None else 1.0 / (q_damp * 2.0 * np.pi * f0)
        self.layer_depth = 0.0
        self._chab_kw = dict(sigma_y=np.inf, yield_strain=None, c_kin=30.0,
                             gamma=100.0, q_iso=0.2, b_iso=150.0, k_visc=4.0,
                             n_visc=5.0)
        comm = MPI.COMM_WORLD if comm is None else comm

        self.msh = mesh.create_rectangle(
            comm, [np.array([0.0, 0.0]), np.array([L, H])], [nx, ny],
            mesh.CellType.quadrilateral)

        gll = basix.LagrangeVariant.gll_isaac
        sca_el = basix.ufl.element("Lagrange", "quadrilateral", degree,
                                   lagrange_variant=gll)
        self.Q = fem.functionspace(self.msh, sca_el)
        self.dx = ufl.Measure("dx", domain=self.msh,
                              metadata={"quadrature_rule": "GLL",
                                        "quadrature_degree": 2 * degree - 1})
        self.h = min(L / nx, H / ny)

    # ------------------------------------------------------------------ medium
    def set_medium(self, rng, n_grid=(321, 193), **rf_kw):
        """Same random medium as the plane-strain solver; only mu and rho are
        needed here, and the time step is set by Vs, not Vp."""
        x1 = np.linspace(0.0, self.L, n_grid[0])
        x2 = np.linspace(0.0, self.H, n_grid[1])
        self.grid = (x1, x2)
        self.fields = lame_fields(x1, x2, rng=rng, **rf_kw)

        self.mu = fem.Function(self.Q, name="mu")
        self.rho = fem.Function(self.Q, name="rho")
        for name, fn in (("mu", self.mu), ("rho", self.rho)):
            interp = RegularGridInterpolator((x2, x1), self.fields[name],
                                             bounds_error=False, fill_value=None)
            fn.interpolate(lambda x, interp=interp:
                           interp(np.column_stack((np.clip(x[1], 0.0, self.H),
                                                   np.clip(x[0], 0.0, self.L)))))
        c_max = float(self.fields["vs"].max())
        self.dt = self.cfl * (self.h / self.degree) / c_max
        self.n_steps = int(np.ceil(self.t_end / self.dt))
        return self

    def set_plastic_layer(self, depth=150.0, sigma_y=np.inf, yield_strain=None,
                          c_kin=30.0, gamma=100.0, q_iso=0.2, b_iso=150.0,
                          k_visc=4.0, n_visc=5.0):
        """Mark the top `depth` metres as visco-plastic.  `sigma_y = inf`
        keeps it elastic; `yield_strain` gives the yield as sqrt(3) mu gamma_y,
        i.e. from a yield engineering shear strain."""
        self.layer_depth = float(depth)
        self._chab_kw = dict(sigma_y=sigma_y, yield_strain=yield_strain,
                             c_kin=c_kin, gamma=gamma, q_iso=q_iso,
                             b_iso=b_iso, k_visc=k_visc, n_visc=n_visc)
        return self

    # ------------------------------------------------------------- quadrature
    def _setup_state(self):
        qdeg = 2 * self.degree - 1
        self.qpts, _ = basix.make_quadrature(basix.CellType.quadrilateral,
                                             qdeg, basix.QuadratureType.gll)
        q2 = basix.ufl.quadrature_element("quadrilateral", value_shape=(2,),
                                          scheme="gll", degree=qdeg)
        q1 = basix.ufl.quadrature_element("quadrilateral", value_shape=(),
                                          scheme="gll", degree=qdeg)
        self.Wt = fem.functionspace(self.msh, q2)
        self.Ws = fem.functionspace(self.msh, q1)

        mu_q = fem.Function(self.Ws)
        mu_q.interpolate(fem.Expression(self.mu, self.qpts))
        self.mu_q = mu_q.x.array.copy()
        xq = fem.Function(self.Wt)
        xq.interpolate(fem.Expression(ufl.SpatialCoordinate(self.msh), self.qpts))
        self.xq = xq.x.array.reshape(-1, 2).copy()
        self.active = self.xq[:, 1] > self.H - self.layer_depth

        kw = self._chab_kw
        sy = kw["sigma_y"]
        if kw["yield_strain"] is not None:
            sy = np.sqrt(3.0) * self.mu_q * float(kw["yield_strain"])
        sy = np.broadcast_to(np.asarray(sy, dtype=float), self.mu_q.shape).copy()
        self.model = ChabocheSH(sigma_y=sy, C_kin=kw["c_kin"] * sy,
                                gamma=kw["gamma"], Q_iso=kw["q_iso"] * sy,
                                b_iso=kw["b_iso"], K_visc=kw["k_visc"] * sy,
                                n_visc=kw["n_visc"])

        nq = self.mu_q.size
        self.sig = np.zeros((nq, 2))
        self.Xb = np.zeros((nq, 2))
        self.Riso = np.zeros(nq)
        self.pacc = np.zeros(nq)

    # ------------------------------------------------------------------- setup
    def _assemble_statics(self, src_xy, rec_xy, plane_depth=None):
        Q, dx = self.Q, self.dx
        v = ufl.TestFunction(Q)
        self.u = fem.Function(Q, name="u3")

        # row-sum lumped mass of the scalar field
        m_vec = assemble_vector(fem.form(self.rho * v * dx))
        m_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.m = m_vec.array_r.copy()

        xdofs = self.Q.tabulate_dof_coordinates()[:, :2]
        self.xdofs = xdofs
        self.i_src = int(np.argmin(np.sum((xdofs - np.asarray(src_xy)) ** 2,
                                          axis=1)))
        self.src_dof = self.i_src
        pts = np.atleast_2d(np.asarray(rec_xy, dtype=float))
        self.rec_pts = pts
        self.rec_dofs = np.array(
            [int(np.argmin(np.sum((xdofs - q) ** 2, axis=1))) for q in pts],
            dtype=int)
        self.rec_dof = int(self.rec_dofs[0])
        self.rec_xy = xdofs[self.rec_dofs]

        # optional plane-wave drive: a body force per unit mass, uniform in x
        # and Gaussian in depth, so that the acceleration it imposes is the
        # same at every x -- a vertically propagating plane SH wave, free of
        # the GLL node weighting a line of point forces would carry
        if plane_depth is None:
            self.src_weight = None
        else:
            y0 = self.H - float(plane_depth)
            width = 2.0 * self.h / self.degree
            self.src_weight = self.m * np.exp(
                -((xdofs[:, 1] - y0) / width) ** 2)

        # Cerjan absorbing taper on the two sides and the bottom
        d = np.minimum.reduce([xdofs[:, 0], self.L - xdofs[:, 0], xdofs[:, 1]])
        taper = np.ones_like(d)
        inside = d < self.sponge
        taper[inside] = np.exp(-(0.25 * (self.sponge - d[inside]) / self.sponge) ** 2)
        self.taper = taper

        self._setup_state()

        # internal force from the stored stress, f_int = int tau . grad(v)
        self.sig_tot = fem.Function(self.Wt, name="tau")
        self.f_int = fem.form(ufl.inner(self.sig_tot, ufl.grad(v)) * dx)

        # tensor strain increment de = grad(du)/2 at the quadrature points
        self.du = fem.Function(Q, name="du")
        g = ufl.grad(self.du)
        self.deps_fn = fem.Function(self.Wt)
        self.eps_expr = fem.Expression(
            ufl.as_vector([0.5 * g[0], 0.5 * g[1]]), self.qpts)

    # -------------------------------------------------------------------- solve
    def run(self, src_xy=None, rec_xy=None, snap_at=0.55, progress=False,
            f0=None, plane_depth=None):
        src_xy = (self.L * 0.5, self.H - 1.0) if src_xy is None else src_xy
        rec_xy = (self.L * 0.80, self.H - 1.0) if rec_xy is None else rec_xy
        self._assemble_statics(src_xy, rec_xy, plane_depth=plane_depth)

        f0 = self.f0 if f0 is None else float(f0)
        snap_times = np.atleast_1d(np.asarray(snap_at, dtype=float))
        snap_steps = [int(ts / self.dt) for ts in snap_times]
        self.snap_times = snap_times
        self.snapshots = [None] * len(snap_steps)
        n_loc = self.u.x.array.size
        u_old = np.zeros(n_loc)
        u_now = np.zeros(n_loc)
        traces = np.zeros((self.rec_dofs.size, self.n_steps))
        t0 = 1.2 / f0
        self.j2_max = 0.0
        self.n_yield_events = 0
        c_lo = 0.5 * self.alpha * self.dt

        for n in range(self.n_steps):
            t = n * self.dt
            r = assemble_vector(self.f_int)
            r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            force = -r.array_r.copy()
            s = self.force_amp * ricker(t, f0, t0)
            if self.src_weight is None:
                force[self.src_dof] += s
            else:
                force += s * self.src_weight

            u_new = (2.0 * u_now - (1.0 - c_lo) * u_old
                     + self.dt**2 * force / self.m) / (1.0 + c_lo)
            u_new *= self.taper

            self.du.x.array[:] = u_new - u_now
            self.deps_fn.interpolate(self.eps_expr)
            deps = self.deps_fn.x.array.reshape(-1, 2)
            dsig_el, n_yield = self.model.update(
                self.sig, self.Xb, self.Riso, self.pacc, deps,
                self.mu_q, self.dt, self.active)
            self.n_yield_events += n_yield

            sig_tot = self.sig if self.beta == 0.0 else \
                self.sig + (self.beta / self.dt) * dsig_el
            self.sig_tot.x.array[:] = sig_tot.reshape(-1)

            if self.active.any():
                self.j2_max = max(self.j2_max,
                                  float(j2_sh(self.sig[self.active]).max()))

            traces[:, n] = (u_new[self.rec_dofs]
                            - u_old[self.rec_dofs]) / (2.0 * self.dt)
            for j, ns in enumerate(snap_steps):
                if n == ns:
                    self.snapshots[j] = (u_new - u_old) / (2.0 * self.dt)
            u_old, u_now = u_now, u_new
            if progress and n % 400 == 0:
                print(f"  step {n}/{self.n_steps}  t = {t:.3f} s  "
                      f"max p = {self.pacc.max():.2e}", flush=True)

        self.traces = traces
        self.trace = traces[0]
        self.time = np.arange(self.n_steps) * self.dt
        if self.snapshots[0] is None:
            self.snapshots[0] = (u_now - u_old) / (2.0 * self.dt)
        self.snapshot = self.snapshots[0]
        return self

    # ------------------------------------------------------------- post-process
    def snapshot_grid(self, n=(260, 160), crop=None, which=0):
        """|v3| of the stored snapshot on a regular grid (the antiplane
        particle velocity is a scalar, so the magnitude is its modulus)."""
        from scipy.interpolate import griddata
        crop = self.sponge if crop is None else crop
        snap = self.snapshots[which] if getattr(self, "snapshots", None) \
            else self.snapshot
        gx = np.linspace(crop, self.L - crop, n[0])
        gy = np.linspace(crop, self.H, n[1])
        GX, GY = np.meshgrid(gx, gy)
        grid = griddata(self.xdofs, np.abs(snap), (GX, GY), method="linear",
                        fill_value=0.0)
        return gx, gy, grid

    def plastic_strain_grid(self, n=(260, 160), crop=None):
        """Accumulated plastic strain p interpolated onto a regular grid."""
        from scipy.interpolate import griddata
        crop = self.sponge if crop is None else crop
        gx = np.linspace(crop, self.L - crop, n[0])
        gy = np.linspace(crop, self.H, n[1])
        GX, GY = np.meshgrid(gx, gy)
        grid = griddata(self.xq, self.pacc, (GX, GY), method="linear",
                        fill_value=0.0)
        return gx, gy, grid
