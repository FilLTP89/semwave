"""SEM wave solver with a visco-plastic surficial layer.

Same spectral-element discretisation and same explicit central-difference
update as sem_wave.py, but the internal force is assembled from a stress that
is carried as a state variable at the quadrature points instead of being
recomputed from the displacement by Hooke's law.  That is what lets the top of
the model follow a Lemaitre--Chaboche visco-plastic law (chaboche.py) while
the rest stays linear elastic.

The GLL quadrature of degree 2p-1 has exactly the (p+1)^2 GLL nodes of degree
p as its points, so the stress state lives on the collocation nodes and the
lumped-mass central-difference update is untouched:

    assemble f_int(sigma^n)  ->  u^{n+1}  ->  d eps = eps(u^{n+1}-u^n)
                             ->  sigma^{n+1} = chaboche(sigma^n, d eps)

Rayleigh stiffness damping is kept by adding beta C:(d eps/dt) to the stress
that the next assembly sees, which is the same explicit treatment (and the
same stability limit beta < dt/2) as in the linear solver.

Run inside the course Docker image `fenicsx-jupyter` (Notebooks/Dockerfile).
"""
from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector

from .chaboche import Chaboche, j2
from .sem_wave import SEMWave2D, ricker


class SEMWaveVP2D(SEMWave2D):
    """Explicit SEM solver whose surficial layer is visco-plastic."""

    force_amp = 1.0e12          # vertical point force; scaled to put the layer
                                # at a prescribed multiple of its yield stress

    def set_plastic_layer(self, depth=150.0, sigma_y=np.inf, yield_strain=None,
                          c_kin=30.0, gamma=100.0, q_iso=0.2, b_iso=150.0,
                          k_visc=4.0, n_visc=5.0, dp_max=1.0e-6):
        """Mark the top `depth` metres as visco-plastic.

        `sigma_y` is the von Mises yield stress (Pa); passing `np.inf` keeps
        the layer elastic, which is how the peak elastic stress is measured
        before calibrating.  Alternatively `yield_strain` gives it as
        sqrt(3) * mu * gamma_y, i.e. from a yield shear strain.  The hardening
        and viscous parameters are given as multiples of `sigma_y`.
        """
        self.layer_depth = float(depth)
        self._chab_kw = dict(sigma_y=sigma_y, yield_strain=yield_strain,
                             c_kin=c_kin, gamma=gamma, q_iso=q_iso,
                             b_iso=b_iso, k_visc=k_visc, n_visc=n_visc,
                             dp_max=dp_max)
        return self

    # ------------------------------------------------------------- quadrature
    def _setup_state(self):
        qdeg = 2 * self.degree - 1
        self.qpts, _ = basix.make_quadrature(basix.CellType.quadrilateral,
                                             qdeg, basix.QuadratureType.gll)
        q4 = basix.ufl.quadrature_element("quadrilateral", value_shape=(4,),
                                          scheme="gll", degree=qdeg)
        q2 = basix.ufl.quadrature_element("quadrilateral", value_shape=(2,),
                                          scheme="gll", degree=qdeg)
        q1 = basix.ufl.quadrature_element("quadrilateral", value_shape=(),
                                          scheme="gll", degree=qdeg)
        self.Wt = fem.functionspace(self.msh, q4)
        self.Wv = fem.functionspace(self.msh, q2)
        self.Ws = fem.functionspace(self.msh, q1)

        # material parameters and coordinates at the quadrature points
        mu_q, lam_q = fem.Function(self.Ws), fem.Function(self.Ws)
        mu_q.interpolate(fem.Expression(self.mu, self.qpts))
        lam_q.interpolate(fem.Expression(self.lam, self.qpts))
        self.mu_q = mu_q.x.array.copy()
        self.lam_q = lam_q.x.array.copy()
        xq = fem.Function(self.Wv)
        xq.interpolate(fem.Expression(ufl.SpatialCoordinate(self.msh), self.qpts))
        self.xq = xq.x.array.reshape(-1, 2).copy()
        self.active = self.xq[:, 1] > self.H - self.layer_depth

        # the Chaboche parameters, per quadrature point
        kw = self._chab_kw
        sy = kw["sigma_y"]
        if kw["yield_strain"] is not None:
            sy = np.sqrt(3.0) * self.mu_q * float(kw["yield_strain"])
        sy = np.broadcast_to(np.asarray(sy, dtype=float), self.mu_q.shape).copy()
        self.model = Chaboche(sigma_y=sy, C_kin=kw["c_kin"] * sy,
                              gamma=kw["gamma"], Q_iso=kw["q_iso"] * sy,
                              b_iso=kw["b_iso"], K_visc=kw["k_visc"] * sy,
                              n_visc=kw["n_visc"], dp_max=kw["dp_max"])

        nq = self.mu_q.size
        self.sig = np.zeros((nq, 4))
        self.Xb = np.zeros((nq, 4))
        self.Riso = np.zeros(nq)
        self.pacc = np.zeros(nq)

    def _assemble_statics(self, src_xy, rec_xy):
        if getattr(self, "aniso", None) is not None:
            raise NotImplementedError(
                "SEMWaveVP2D carries the stress at the quadrature points "
                "through an isotropic elastic increment (chaboche.py): a TI "
                "fabric would have to go into Chaboche.update() as well. Use "
                "SEMWave2D for anisotropic runs.")
        pts = np.atleast_2d(np.asarray(rec_xy, dtype=float))
        super()._assemble_statics(src_xy, pts[0])
        # one vertical-velocity dof per station
        self.rec_pts = pts
        self.rec_dofs = np.array(
            [2 * int(np.argmin(np.sum((self.xdofs - q) ** 2, axis=1))) + 1
             for q in pts], dtype=int)
        self.rec_xy = np.array([self.xdofs[d // 2] for d in self.rec_dofs])
        self._setup_state()

        # internal force from the stored stress: sig_zz does no work because
        # eps_zz = 0 in plane strain, so only (xx, yy, xy) enter
        self.sig_tot = fem.Function(self.Wt, name="sigma")
        v = ufl.TestFunction(self.V)
        s = ufl.as_matrix([[self.sig_tot[0], self.sig_tot[3]],
                           [self.sig_tot[3], self.sig_tot[1]]])
        self.f_int = fem.form(ufl.inner(s, ufl.sym(ufl.grad(v))) * self.dx)

        # strain increment of a displacement increment, at the quadrature points
        self.du = fem.Function(self.V, name="du")
        e = ufl.sym(ufl.grad(self.du))
        self.deps_fn = fem.Function(self.Wt)
        self.eps_expr = fem.Expression(
            ufl.as_vector([e[0, 0], e[1, 1], 0.0 * e[0, 0], e[0, 1]]), self.qpts)

    # -------------------------------------------------------------------- solve
    def run(self, src_xy=None, rec_xy=None, snap_at=0.55, progress=False,
            f0=None):
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
        trace = np.zeros(self.n_steps)
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
            force[self.src_dof] += self.force_amp * ricker(t, f0, t0)

            u_new = (2.0 * u_now - (1.0 - c_lo) * u_old
                     + self.dt**2 * force / self.m) / (1.0 + c_lo)
            u_new *= self.taper

            # constitutive update on the strain increment just produced
            self.du.x.array[:] = u_new - u_now
            self.deps_fn.interpolate(self.eps_expr)
            deps = self.deps_fn.x.array.reshape(-1, 4)
            dsig_el, n_yield = self.model.update(
                self.sig, self.Xb, self.Riso, self.pacc, deps,
                self.lam_q, self.mu_q, self.dt, self.active)
            self.n_yield_events += n_yield

            # Rayleigh stiffness damping, explicit, as in the linear solver
            sig_tot = self.sig if self.beta == 0.0 else \
                self.sig + (self.beta / self.dt) * dsig_el
            self.sig_tot.x.array[:] = sig_tot.reshape(-1)

            # peak equivalent deviatoric stress reached inside the layer
            if self.active.any():
                dev = self.sig[self.active].copy()
                dev[:, :3] -= dev[:, :3].mean(axis=1)[:, None]
                self.j2_max = max(self.j2_max, float(j2(dev).max()))

            trace[n] = (u_new[self.rec_dof] - u_old[self.rec_dof]) / (2.0 * self.dt)
            traces[:, n] = (u_new[self.rec_dofs]
                            - u_old[self.rec_dofs]) / (2.0 * self.dt)
            for j, ns in enumerate(snap_steps):
                if n == ns:
                    self.snapshots[j] = (u_new - u_old) / (2.0 * self.dt)
            u_old, u_now = u_now, u_new
            if progress and n % 400 == 0:
                print(f"  step {n}/{self.n_steps}  t = {t:.3f} s  "
                      f"max p = {self.pacc.max():.2e}", flush=True)

        self.trace = trace
        self.traces = traces
        self.time = np.arange(self.n_steps) * self.dt
        if self.snapshots[0] is None:
            self.snapshots[0] = (u_now - u_old) / (2.0 * self.dt)
        self.snapshot = self.snapshots[0]
        return self

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
