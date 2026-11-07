"""The Neumann series, term by term, computed -- not asserted.

The first version of the coda figure drew the orders as a cartoon: one pulse
per order, each one later and smaller than the last.  That is wrong twice
over.  The orders are not separated in time (each one has support from the
direct arrival onwards, and they overlap), and the late coda is not "order n
with amplitude delta_c^n" -- it is where *every* order contributes and the
truncated series stops being usable.  This script measures both statements.

Setup (same as run_coda2.py, so the frame keeps its geometry): homogeneous
background, extended fault source, one receiver at the free surface, Cerjan
sponge on the other three sides.

The perturbation is made **exactly linear in sigma**, which the log-normal
field of the earlier runs was not:

    mu = mu0 (1 + sigma g),   lambda = lambda0 (1 + sigma g),   rho = rho0,

with g a unit-variance von Karman pattern (clipped at +-2.5 so the stiffness
stays positive).  Then K_delta = sigma K_g, and the Neumann terms of

    M u0'' = -K0 u0 + f,       M un'' = -K0 un - K_delta u_{n-1}

obey u_n(sigma) = sigma^n u_n(1) exactly -- verified to four digits, so the
orders are solved once, at SIGMA_REF, and rescaled to every other level.  Each
order is a wave propagating in the *background*, driven by the previous order
scattering off the heterogeneity: that is the Neumann series, discretised.

All orders are marched in one time loop (2N+1 assemblies per step), with the
same lumped mass, the same Rayleigh damping and the same sponge taper as the
full run, so sum_n u_n solves the discrete full problem exactly, order by
order.  The exact solution is then run separately at each sigma, and the
truncation error of the N-term partial sum is measured against it.

    RUN IN THE CONTAINER (see semwave/README.md):
    SEMWAVE=/path/to/semwave
    DOLFIN=/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -e PYTHONPATH="/semwave:$DOLFIN" -v "$SEMWAVE":/semwave \
        -v "$PWD":/work -w /work fenicsx-jupyter python3 run_neumann_orders.py

Writes neumann_orders.npz, which make_coda_neumann.py reads.
"""
from __future__ import annotations

import time

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc
from scipy.interpolate import RegularGridInterpolator

from semwave import SEMWave2D
from semwave.random_field import shinozuka_deodatis_2d
from semwave.sem_wave import ricker

GEOM = dict(L=2600., H=1600., nx=52, ny=32, degree=4,
            f0=6.0, t_end=3.5, q_damp=800., alpha_damp=0.10, sponge=200.)
NGRID = (261, 161)
VS0, VP_VS = 1200.0, 1.78
B1, B2, SEED = 250.0, 100.0, 3
GCLIP = 2.5                      # keeps 1 + sigma g positive at SIGMAS[-1]
FAULT = dict(x_a=(900., 500.), x_b=(1350., 720.), n_sub=20,
             v_rup=1600., m0=1.0e12)
REC = (2050., 1599.)
SIGMAS = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])   # stiffness contrast
SIGMA_REF = 0.04                 # the level the orders are solved at
N_ORD = 3                        # orders n = 1 .. N_ORD (plus the ballistic)
SNAP_AT = 1.2


def background():
    vp = VP_VS * VS0
    rho = 310.0 * vp ** 0.25
    return rho * VS0 ** 2, rho * (vp ** 2 - 2.0 * VS0 ** 2), rho


def pattern(sim):
    """The unit-variance von Karman pattern, on the SEM nodes and on a grid."""
    x1 = np.linspace(0.0, sim.L, NGRID[0])
    x2 = np.linspace(0.0, sim.H, NGRID[1])
    g = np.clip(shinozuka_deodatis_2d(x1, x2, sigma=1.0, b1=B1, b2=B2, nu=0.5,
                                      rng=np.random.default_rng(SEED)),
                -GCLIP, GCLIP)
    interp = RegularGridInterpolator((x2, x1), g, bounds_error=False,
                                     fill_value=None)
    fn = fem.Function(sim.Q, name="g")
    fn.interpolate(lambda x: interp(np.column_stack(
        (np.clip(x[1], 0.0, sim.H), np.clip(x[0], 0.0, sim.L)))))
    return fn, g


def make_sim(sigma, dt=None, n_steps=None):
    """One realisation of the medium at contrast `sigma` (0 = background)."""
    sim = SEMWave2D(**GEOM)
    sim.set_medium(np.random.default_rng(SEED), n_grid=NGRID, layers=None,
                   vs0=VS0, vp_vs=VP_VS, sigma_ln=0.0, b1=B1, b2=B2)
    mu0, lam0, rho0 = background()
    g_fn, _ = pattern(sim)
    sim.mu.x.array[:] = mu0 * (1.0 + sigma * g_fn.x.array)
    sim.lam.x.array[:] = lam0 * (1.0 + sigma * g_fn.x.array)
    sim.rho.x.array[:] = rho0
    sim.c_max = VP_VS * VS0 * np.sqrt(1.0 + sigma * GCLIP)
    sim._set_dt()
    if dt is not None:
        sim.dt, sim.n_steps = dt, n_steps
    sim.set_fault(rng=np.random.default_rng(101), **FAULT)
    return sim, g_fn


def run_orders(sigma, dt, n_steps):
    """March u_0 ... u_N together, all of them in the background medium."""
    sim, g_fn = make_sim(0.0, dt, n_steps)
    sim._assemble_statics(FAULT["x_a"], REC)
    V, dx = sim.V, sim.dx
    v = ufl.TestFunction(V)
    mu0, lam0, _ = background()
    ua, ub = fem.Function(V), fem.Function(V)

    def stress(w, m, l):
        return 2.0 * m * ufl.sym(ufl.grad(w)) + l * ufl.div(w) * ufl.Identity(2)

    f0_form = fem.form(ufl.inner(stress(ua, mu0, lam0),
                                 ufl.sym(ufl.grad(v))) * dx)
    fd_form = fem.form(ufl.inner(stress(ub, sigma * mu0 * g_fn,
                                        sigma * lam0 * g_fn),
                                 ufl.sym(ufl.grad(v))) * dx)

    n_loc = sim.u.x.array.size
    old = [np.zeros(n_loc) for _ in range(N_ORD + 1)]
    now = [np.zeros(n_loc) for _ in range(N_ORD + 1)]
    traces = np.zeros((N_ORD + 1, sim.n_steps))
    snaps = [None] * (N_ORD + 1)
    snap_step = int(SNAP_AT / sim.dt)
    t0 = 1.2 / sim.f0
    c_lo = 0.5 * sim.alpha * sim.dt

    def apply(form, fn, vec):
        fn.x.array[:] = vec
        r = assemble_vector(form)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return r.array_r.copy()

    for n in range(sim.n_steps):
        t = n * sim.dt
        damped = [now[k] if sim.beta == 0.0 else
                  now[k] + (sim.beta / sim.dt) * (now[k] - old[k])
                  for k in range(N_ORD + 1)]
        new = []
        for k in range(N_ORD + 1):
            force = -apply(f0_form, ua, damped[k])
            if k == 0:
                np.add.at(force, sim.fault["dofs"],
                          sim.fault["amps"]
                          * ricker(t - sim.fault["delays"], sim.f0, t0))
            else:
                force -= apply(fd_form, ub, damped[k - 1])
            u_new = (2.0 * now[k] - (1.0 - c_lo) * old[k]
                     + sim.dt ** 2 * force / sim.m) / (1.0 + c_lo)
            new.append(u_new * sim.taper)
        for k in range(N_ORD + 1):
            traces[k, n] = (new[k][sim.rec_dof] - old[k][sim.rec_dof]) \
                / (2.0 * sim.dt)
            if n == snap_step:
                snaps[k] = (new[k] - old[k]) / (2.0 * sim.dt)
        old, now = now, new
        if n % 1000 == 0:
            print(f"    step {n}/{sim.n_steps}", flush=True)

    grids = []
    for k in range(N_ORD + 1):
        sim.snapshots = [snaps[k]]
        gx, gy, sg = sim.snapshot_grid()
        grids.append(sg)
    return traces, np.asarray(grids), gx, gy, sim


if __name__ == "__main__":
    t_all = time.time()

    probe, _ = make_sim(SIGMAS.max())          # lock dt on the strongest medium
    DT, NSTEP = probe.dt, probe.n_steps
    _, g_grid = pattern(probe)
    dc = np.sqrt(1.0 + SIGMAS[:, None, None] * g_grid[None]) - 1.0
    dc_rms = np.sqrt((dc ** 2).mean(axis=(1, 2)))
    print(f"locked dt = {DT:.3e} s, {NSTEP} steps; "
          "rms dc/c = " + ", ".join(f"{s:.3f}->{r * 100:.2f}%"
                                    for s, r in zip(SIGMAS, dc_rms)), flush=True)

    t0 = time.time()
    orders, order_snaps, gx, gy, sim = run_orders(SIGMA_REF, DT, NSTEP)
    print(f"  orders at sigma={SIGMA_REF}: {time.time() - t0:.0f}s; "
          "peak per order / ballistic: "
          + ", ".join(f"n={k}: {np.abs(orders[k]).max() / np.abs(orders[0]).max():.3e}"
                      for k in range(N_ORD + 1)), flush=True)

    exact = np.zeros((SIGMAS.size, NSTEP))
    for j, s in enumerate(SIGMAS):
        t0 = time.time()
        sm, _ = make_sim(float(s), DT, NSTEP)
        sm.run(src_xy=FAULT["x_a"], rec_xy=REC, snap_at=SNAP_AT)
        exact[j] = sm.trace
        print(f"  exact sigma={s:.3f}: {time.time() - t0:.0f}s "
              f"max={np.abs(sm.trace).max():.3e}", flush=True)

    # truncation error of the N-term partial sum, against the exact solution
    print("\n  relative truncation error  max|U - sum_{n<=N} u_n| / max|U|:")
    for j, s in enumerate(SIGMAS):
        row = []
        for N in range(N_ORD + 1):
            part = sum((s / SIGMA_REF) ** k * orders[k] for k in range(N + 1))
            row.append(np.abs(exact[j] - part).max() / np.abs(exact[j]).max())
        print(f"    dc/c={dc_rms[j] * 100:5.2f}%  "
              + "  ".join(f"N={N}: {e:.3e}" for N, e in enumerate(row)),
              flush=True)

    np.savez_compressed(
        "neumann_orders.npz", t=np.arange(NSTEP) * DT, dt=DT,
        sigmas=SIGMAS, dc_rms=dc_rms, sigma_ref=np.asarray(SIGMA_REF),
        orders=orders, exact=exact, order_snaps=order_snaps, gx=gx, gy=gy,
        g_grid=g_grid, snap_at=np.asarray(SNAP_AT),
        src=np.asarray(FAULT["x_a"]), rec=np.asarray(sim.rec_xy[0]),
        f0=np.asarray(GEOM["f0"]))
    print(f"\ncached neumann_orders.npz  ({time.time() - t_all:.0f}s total)",
          flush=True)
