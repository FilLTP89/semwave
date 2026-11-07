"""The ballistic front, the coda it turns into, and the record that sees both.

One elastic run of semwave's 2-D SEM solver on a medium that is

  * heterogeneous -- layered cake x von Karman log-Gaussian fluctuations
    (sigma_ln = 15%, correlation lengths growing with depth), which is what
    scatters the direct wave into a coda, and
  * anisotropic   -- a transversely isotropic fabric dipping 30 deg from the
    vertical, strongest in the young sediments and fading with depth as they
    compact (Thomsen epsilon = 0.25 at the surface, delta = 0.4 epsilon).

No plasticity anywhere: the point of the slide this feeds is that a *linear*
elastic medium already destroys the compact pulse, with the kernel still
intact.  The nonlinear soil layer is the next slide's argument, and the next
run.

The run stores a small movie of snapshots (velocity magnitude on a regular
grid) plus the full time history at a surface station, so the figure script
can pick which instants to show without solving anything again.

Run inside the fenicsx-jupyter image (see semwave/README.md), from this
directory:

    SEMWAVE=/path/to/semwave
    DOLFIN=/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -e PYTHONPATH="/semwave:$DOLFIN" \
        -v "$SEMWAVE":/semwave -v "$PWD":/work -w /work \
        fenicsx-jupyter python3 run_ballistic_coda.py

then ../make_ballistic_coda.py on the host.
"""
import time

import numpy as np

from semwave import (DEFAULT_LAYERS, SEMWave2D, layer_depths,
                     layered_profile, phase_velocities, thomsen_stiffness)

OUT = "ballistic_coda.npz"
SEED = 1000                      # the geology of the HDR figures

# 2600 x 1300 m, 50 m elements of degree 4 (12.5 m node spacing, ~12 nodes per
# shear wavelength at 6 Hz).  The sponge is cropped away when plotting, which
# leaves a 2200 x 1100 m window -- a 2:1 panel, the shape the slide wants.
GEO = dict(L=2600.0, H=1300.0, nx=52, ny=26, degree=4, f0=6.0, t_end=3.5,
           q_damp=120.0, alpha_damp=1.2, sponge=200.0)
MED = dict(n_grid=(261, 131), sigma_ln=0.15, b1=250.0, b2=100.0,
           layers=DEFAULT_LAYERS)
SRC = (1000.0, 650.0)                       # buried 650 m: body phases first
REC = [(1900.0, 1299.0), (1900.0, 1000.0)]  # free surface, and 300 m below it

# snapshots: dense enough that the figure can pick the instant when the front
# is at the station and one deep in the coda, without re-solving
SNAPS = (0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.80, 2.10,
         2.40, 2.70, 3.00)

EPSILON_0, Z_FABRIC, DIP = 0.25, 600.0, np.deg2rad(30.0)


def epsilon_field(x1, x2):
    """Thomsen epsilon: 25% P anisotropy in the surficial sediments, decaying
    with depth over 600 m as compaction destroys the preferred orientation."""
    return EPSILON_0 * np.exp(-(GEO["H"] - x2) / Z_FABRIC)


def delta_field(x1, x2):
    """delta < epsilon: the qP front is flatter than the ellipse near the
    symmetry axis, and the qSV front bulges at 45 deg."""
    return 0.4 * epsilon_field(x1, x2)


def ray_arrivals(src, rec, n=400):
    """Straight-ray P and S travel times through the *background* medium.

    Not a ray tracer: the path is the straight segment, and the velocities are
    the qP/qSV phase velocities of the layered background at the angle the
    segment makes with the symmetry axis.  It is only there to put "direct P"
    and "direct S" on the record at the right place, and it lands within a
    dominant period of the onset the solver produces.
    """
    src, rec = np.asarray(src, float), np.asarray(rec, float)
    xi = (np.arange(n) + 0.5) / n
    pts = src + xi[:, None] * (rec - src)
    ds = float(np.hypot(*(rec - src))) / n
    ray_from_vertical = np.arctan2(abs(rec[0] - src[0]), abs(rec[1] - src[1]))

    vs = layered_profile(pts[:, 1], GEO["H"], DEFAULT_LAYERS)
    vp = 1.78 * vs
    rho = 310.0 * vp**0.25                      # Gardner, as in random_field
    mu, lam = rho * vs**2, rho * (vp**2 - 2.0 * vs**2)
    eps = epsilon_field(pts[:, 0], pts[:, 1])
    c = thomsen_stiffness(lam, mu, eps, delta_field(pts[:, 0], pts[:, 1]))
    v_qp, v_qsv = phase_velocities(*c, rho, abs(ray_from_vertical - DIP))
    return float((ds / v_qp).sum()), float((ds / v_qsv).sum())


t_all = time.time()
sim = SEMWave2D(**GEO)
sim.set_medium(np.random.default_rng(SEED), **MED)
sim.set_anisotropy(epsilon=epsilon_field, delta=delta_field, theta=DIP)

ang = np.deg2rad([0.0, 45.0, 90.0])
v_qp, v_qsv = sim.phase_velocity(ang)
top = sim.fields["vs"].shape[0] - 1          # the free-surface row of the grid
print(f"fabric: epsilon {sim.aniso_fields['epsilon'].max():.3f} at the surface "
      f"-> {sim.aniso_fields['epsilon'].min():.3f} at depth, dip "
      f"{np.rad2deg(DIP):.0f} deg", flush=True)
print("  qP  at the surface, 0/45/90 deg from the symmetry axis: "
      + " ".join(f"{v:7.1f}" for v in v_qp[top].mean(axis=0)), flush=True)
print("  qSV at the surface, 0/45/90 deg from the symmetry axis: "
      + " ".join(f"{v:7.1f}" for v in v_qsv[top].mean(axis=0)), flush=True)
print(f"  dt {sim.dt:.3e} s, {sim.n_steps} steps "
      f"(CFL speed {sim._aniso_cfac:.3f} x isotropic Vp_max)", flush=True)

t_p, t_s = ray_arrivals(SRC, REC[0])
print(f"  straight-ray arrivals at the surface station: P {t_p:.3f} s, "
      f"S {t_s:.3f} s", flush=True)

t0 = time.time()
sim.run(src_xy=SRC, rec_xy=REC, snap_at=SNAPS, progress=True)
print(f"  solved in {time.time() - t0:.0f} s", flush=True)

gx, gy, _ = sim.snapshot_grid(which=0)
snaps = np.stack([sim.snapshot_grid(which=j)[2] for j in range(len(SNAPS))])
print(f"  snapshots {snaps.shape}, peak {snaps.max():.3e} m/s", flush=True)

# the fabric's own polar diagram, at the surface: the figure draws it as a
# glyph, so it has to come from the same stiffnesses the solver used
polar = np.linspace(0.0, 2.0 * np.pi, 361)
v_qp_s, v_qsv_s = sim.phase_velocity(polar)
polar_qp, polar_qsv = v_qp_s[top].mean(axis=0), v_qsv_s[top].mean(axis=0)

# the medium itself, on the same window, for the record
x1, x2 = sim.grid
ix = (x1 >= gx[0]) & (x1 <= gx[-1])
iy = (x2 >= gy[0]) & (x2 <= gy[-1])
cut = np.ix_(iy, ix)

np.savez_compressed(
    OUT,
    gx=gx, gy=gy, snaps=snaps, snap_times=np.asarray(SNAPS, dtype=float),
    time=sim.time, trace_z=sim.traces, trace_x=sim.traces_x,
    rec=sim.rec_xy, src=np.asarray(SRC, dtype=float),
    vs=sim.fields["vs"][cut], epsilon=sim.aniso_fields["epsilon"][cut],
    mx=x1[ix], my=x2[iy],
    layer_tops=np.asarray(layer_depths(DEFAULT_LAYERS), dtype=float),
    theta=DIP, epsilon_0=EPSILON_0, z_fabric=Z_FABRIC,
    t_p_ray=t_p, t_s_ray=t_s,
    polar_angle=polar, polar_qp=polar_qp, polar_qsv=polar_qsv,
    L=GEO["L"], H=GEO["H"], f0=GEO["f0"], dt=sim.dt, sponge=GEO["sponge"],
    sigma_ln=MED["sigma_ln"], seed=SEED,
    q_damp=GEO["q_damp"], alpha_damp=GEO["alpha_damp"])
print(f"total {time.time() - t_all:.0f} s -> {OUT}", flush=True)
