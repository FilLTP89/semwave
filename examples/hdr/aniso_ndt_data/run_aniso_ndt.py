"""Laser-ultrasound in a textured polycrystal: what anisotropy does to the coda.

The NDT counterpart of the seismological forward-map frame, with the setup of
the EAAI paper (Dorval, Gatti et al., metamodelling of LU in WLAM parts):

  * IN718 at room temperature: rho = 8200, vP = 5800, vS = 3100 m/s for the
    equivalent homogeneous isotropic medium;
  * 25 x 10 mm coupon, vertical surface force at x = 12.5 mm (ablation-regime
    laser source), 41 surface receivers within +-5 mm of it: one B-scan;
  * grains from a Voronoi tessellation, elongated by k_el along a direction
    alpha_el, each with its own crystal orientation phi ~ U(-45, 45) deg
    ("transverse texture").

Single crystals are cubic, and the paper fixes their constants so that the
**Hill average of the texture is the same isotropic medium whatever the Zener
ratio A = 2 C44 / (C11 - C12)**:

    C11(A) = rho vP^2 - (A-1)(A+3)/(A^2+6A+1) rho vS^2
    C12(A) = rho vP^2 - (A^2+10A+5)/(A^2+6A+1) rho vS^2
    C44(A) = 4A(A+1)/(A^2+6A+1) rho vS^2

That is the whole point of the experiment: every run below has the same mean
medium and the same source, so anything that changes in the record is the
texture speaking, not a faster or slower material.

In 2-D plane strain a rotated cubic crystal is exactly a tilted TI medium, so
semwave's Thomsen interface represents it with no approximation:

    C33 = lam + 2 mu = C11,  C55 = mu = C44  ->  mu = C44, lam = C11 - 2 C44,
    epsilon = (C11 - C33)/(2 C33) = 0,
    delta   = ((C12 + C44)^2 - (C11 - C44)^2) / (2 C11 (C11 - C44)),
    theta   = phi  (crystal orientation, counterclockwise from the vertical).

Cases:
    iso     A = 1, homogeneous                      the reference kernel
    mono    A = 2.5, phi = -20 deg everywhere       the cusped wavefront
    poly_A  A in POLY_A, one Voronoi texture        scattering and coda
    (plus EXTRA_SEEDS further textures at A = 2.5, for the spread)

    RUN IN THE CONTAINER (see semwave/README.md):
    SEMWAVE=/path/to/semwave
    DOLFIN=/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -e PYTHONPATH="/semwave:$DOLFIN" -v "$SEMWAVE":/semwave \
        -v "$PWD":/work -w /work fenicsx-jupyter python3 run_aniso_ndt.py

Writes aniso_ndt.npz, which ../make_aniso_fabric.py and ../make_aniso_coda.py read.
"""
from __future__ import annotations

import time

import numpy as np

from semwave import SEMWave2D
from semwave.anisotropy import phase_velocities

# ----------------------------------------------------------------- material
RHO, VP, VS = 8200.0, 5800.0, 3100.0          # IN718, equivalent isotropic
# ------------------------------------------------------------------ geometry
MM = 1.0e-3
# the mesh is set by the RAYLEIGH wavelength, not the body waves: at 4.6
# nodes per lambda_R the isotropic half-space keeps a 10% numerical tail that
# would be read as a coda.  At 11 nodes per lambda_R it is 1.6%, i.e. well
# under the texture-generated coda this figure is about.
# H and the sponge are set by the bottom reflection, not by the paper's
# coupon: with H = 10 mm and a 3 mm sponge the P wave comes back at 3.4 us
# and carries 83% of what the isotropic run calls "coda".  At H = 14 mm with
# a 4.5 mm sponge it lands past the record and is absorbed on the way.
GEOM = dict(L=25.0 * MM, H=14.0 * MM, nx=100, ny=56, degree=4,
            f0=4.0e6, t_end=5.0e-6, cfl=0.28, sponge=4.5 * MM,
            q_damp=None, alpha_damp=0.0)
NGRID = (401, 225)
SRC = (12.5 * MM, GEOM["H"])                   # laser spot, free surface
REC_OFFSETS = np.linspace(-5.0, 5.0, 41) * MM  # the B-scan line
SNAP_AT = (0.6e-6, 2.2e-6)                     # front, then coda
# ------------------------------------------------------------- microstructure
N_GRAINS = 85                                  # ~4 mm^2 per grain over 350 mm^2
K_EL, ALPHA_EL = 3.0, np.deg2rad(-20.0)        # elongation, from the vertical
SEED_MICRO = 333
POLY_A = (1.5, 2.0, 2.5)
EXTRA_SEEDS = (334, 335, 336)                  # more textures at A = POLY_A[-1]
MONO_PHI = np.deg2rad(-20.0)


def cubic_constants(A):
    """(C11, C12, C44) of the single crystal, Hill-equivalent to (vP, vS)."""
    den = A * A + 6.0 * A + 1.0
    c11 = RHO * VP ** 2 - (A - 1.0) * (A + 3.0) / den * RHO * VS ** 2
    c12 = RHO * VP ** 2 - (A * A + 10.0 * A + 5.0) / den * RHO * VS ** 2
    c44 = 4.0 * A * (A + 1.0) / den * RHO * VS ** 2
    return c11, c12, c44


def thomsen_of(A):
    """The same crystal as (lam, mu, epsilon, delta) for the TI interface."""
    c11, c12, c44 = cubic_constants(A)
    mu, lam = c44, c11 - 2.0 * c44
    delta = ((c12 + c44) ** 2 - (c11 - c44) ** 2) / (2.0 * c11 * (c11 - c44))
    return lam, mu, 0.0, delta


def orientation_field(x1, x2, seed, k_el=K_EL, alpha_el=ALPHA_EL,
                      n_grains=N_GRAINS):
    """Voronoi grains, elongated along `alpha_el`, one crystal angle each."""
    rng = np.random.default_rng(seed)
    sx = rng.uniform(0.0, x1[-1], n_grains)
    sy = rng.uniform(0.0, x2[-1], n_grains)
    phi = rng.uniform(-np.pi / 4.0, np.pi / 4.0, n_grains)
    X1, X2 = np.meshgrid(x1, x2)               # (n_x2, n_x1)
    ca, sa = np.cos(alpha_el), np.sin(alpha_el)
    best = np.full(X1.shape, np.inf)
    who = np.zeros(X1.shape, dtype=int)
    for g in range(n_grains):
        dx, dy = X1 - sx[g], X2 - sy[g]
        along = -sa * dx + ca * dy             # the elongation axis
        across = ca * dx + sa * dy
        d2 = (along / k_el) ** 2 + across ** 2
        hit = d2 < best
        best, who = np.where(hit, d2, best), np.where(hit, g, who)
    return phi[who], (sx, sy, phi)


def build(A, theta_field=None, dt=None, n_steps=None):
    """One coupon: single-crystal constants, texture, receivers."""
    lam, mu, eps, delta = thomsen_of(A)
    sim = SEMWave2D(**GEOM)
    sim.set_medium(np.random.default_rng(1), n_grid=NGRID, layers=None,
                   vs0=VS, vp_vs=VP / VS, sigma_ln=0.0)
    # swap the isotropic Hill medium for the single crystal: same rho
    for name, val in (("mu", mu), ("lam", lam), ("rho", RHO)):
        sim.fields[name][:] = val
        getattr(sim, name).x.array[:] = val
    sim.fields["vs"][:] = np.sqrt(mu / RHO)
    sim.fields["vp"][:] = np.sqrt((lam + 2.0 * mu) / RHO)
    sim.c_max = float(sim.fields["vp"].max())
    sim._set_dt()
    if A != 1.0 or theta_field is not None:
        theta = 0.0 if theta_field is None else theta_field
        sim.set_anisotropy(epsilon=eps, delta=delta, theta=theta)
    if dt is not None:
        sim.dt, sim.n_steps = dt, n_steps
    return sim


def stations():
    return [(SRC[0] + off, GEOM["H"]) for off in REC_OFFSETS]


def run(sim, tag, t0):
    sim.run(src_xy=SRC, rec_xy=stations(), snap_at=SNAP_AT)
    print(f"  {tag:<16} {sim.n_steps} steps, dt={sim.dt:.2e} s, "
          f"peak |u_y| = {np.abs(sim.traces).max():.3e} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return sim


if __name__ == "__main__":
    t_all = time.time()
    out = {}

    # the strongest fabric sets the time step for everybody, so that every
    # record lives on one time grid and can be subtracted
    x1 = np.linspace(0.0, GEOM["L"], NGRID[0])
    x2 = np.linspace(0.0, GEOM["H"], NGRID[1])
    phi, seeds = orientation_field(x1, x2, SEED_MICRO)
    probe = build(max(POLY_A), theta_field=phi)
    DT, NSTEP = probe.dt, probe.n_steps
    v_qp, v_qs = probe.phase_velocity(np.linspace(0.0, np.pi, 181))
    print(f"locked dt = {DT:.3e} s, {NSTEP} steps", flush=True)
    for A in (1.0,) + tuple(POLY_A):
        c11, c12, c44 = cubic_constants(A)
        lam, mu, _, delta = thomsen_of(A)
        vp, vs = phase_velocities(c11, c12, c11, c44, RHO,
                                  np.linspace(0.0, np.pi / 2.0, 91))
        print(f"  A={A:<4} C11={c11:.3e} C12={c12:.3e} C44={c44:.3e} "
              f"delta={delta:+.3f}  qP {vp.min():.0f}-{vp.max():.0f} m/s, "
              f"qS {vs.min():.0f}-{vs.max():.0f} m/s", flush=True)

    # ---- the reference: the Hill-equivalent isotropic medium --------------
    t0 = time.time()
    iso = run(build(1.0, dt=DT, n_steps=NSTEP), "isotropic", t0)
    out["traces_iso"] = iso.traces
    for j in range(len(SNAP_AT)):
        gx, gy, sg = iso.snapshot_grid(which=j)
        out["gx"], out["gy"] = gx, gy
        out[f"snap_iso_{j}"] = sg

    # ---- one crystal, no grain boundaries: the cusped wavefront -----------
    t0 = time.time()
    mono = run(build(max(POLY_A), theta_field=MONO_PHI, dt=DT, n_steps=NSTEP),
               "monocrystal", t0)
    out["traces_mono"] = mono.traces
    for j in range(len(SNAP_AT)):
        _, _, sg = mono.snapshot_grid(which=j)
        out[f"snap_mono_{j}"] = sg

    # ---- the textured polycrystal, at three anisotropy ratios -------------
    for A in POLY_A:
        t0 = time.time()
        sim = run(build(A, theta_field=phi, dt=DT, n_steps=NSTEP),
                  f"poly A={A}", t0)
        out[f"traces_poly_{A:g}"] = sim.traces
        for j in range(len(SNAP_AT)):
            _, _, sg = sim.snapshot_grid(which=j)
            out[f"snap_poly_{A:g}_{j}"] = sg

    # ---- more textures at the strongest A, for the spread -----------------
    for s in EXTRA_SEEDS:
        t0 = time.time()
        ph, _ = orientation_field(x1, x2, s)
        sim = run(build(max(POLY_A), theta_field=ph, dt=DT, n_steps=NSTEP),
                  f"poly seed {s}", t0)
        out[f"traces_seed_{s}"] = sim.traces

    np.savez_compressed(
        "aniso_ndt.npz", t=np.arange(NSTEP) * DT, dt=DT,
        phi=phi, x1=x1, x2=x2, seeds_xy=np.array(seeds[:2]),
        poly_A=np.asarray(POLY_A), extra_seeds=np.asarray(EXTRA_SEEDS),
        mono_phi=np.asarray(MONO_PHI), k_el=np.asarray(K_EL),
        alpha_el=np.asarray(ALPHA_EL), n_grains=np.asarray(N_GRAINS),
        rec_x=np.asarray([s[0] for s in stations()]),
        rec_offsets=REC_OFFSETS, src=np.asarray(SRC),
        snap_at=np.asarray(SNAP_AT), f0=np.asarray(GEOM["f0"]),
        angles=np.linspace(0.0, np.pi, 181),
        v_qp=v_qp.reshape(-1, 181).mean(axis=0),
        v_qs=v_qs.reshape(-1, 181).mean(axis=0),
        vp_iso=np.asarray(VP), vs_iso=np.asarray(VS), rho=np.asarray(RHO),
        **out)
    print(f"cached aniso_ndt.npz  ({time.time() - t_all:.0f}s total)",
          flush=True)
