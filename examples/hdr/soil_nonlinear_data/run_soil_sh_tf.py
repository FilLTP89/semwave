"""Amplitude sweep behind hdr_soil_nonlinear.png: one soil column, one event,
three drive levels, and the SH transfer function each of them produces.

This is the antiplane (SH) version of run_soil_tf.py, and it replaces it: a
surface-to-borehole transfer function is read on *shear* waves, and in a 2-D
plane-strain run the vertical component mixes P, SV and Rayleigh energy at
whatever incidence the source happens to give, which buries the resonance of
the column.  The out-of-plane motion decouples (sem_wave_sh.py), travels at
Vs alone, and the source sits **under the station column**, so the column is
read at vertical incidence -- exactly the configuration a surface/borehole
pair measures.

Same geometry, same random medium (MEDIUM_SEED) and the same Lemaitre-Chaboche
surficial layer (in its antiplane restriction, chaboche_sh.py) as the
plane-strain runs.  What changes between the passes is only the **amplitude of
the source**:

  pass 0   the layer kept elastic (sigma_y = inf).  Linear, so its stresses and
           its traces scale exactly with the force: it both calibrates the
           sweep (peak elastic J2 in the layer) and provides the
           amplitude-independent reference transfer function.
  pass k   the layer visco-plastic, sigma_y = sqrt(3) mu gamma_y with
           gamma_y = 1e-4, and the force scaled so that the peak elastic J2 in
           the layer is LEVELS[k] times that yield stress -- i.e. the peak
           elastic engineering shear strain is LEVELS[k] * gamma_y.

Two stations sit on the same vertical: one at the free surface, one in the
elastic substratum just below the base of the layer.  Their spectral ratio is
the transfer function H of the soil column.  Because the only thing that
changes between passes is the source, every difference between the H curves is
a statement about the operator: G(c, alpha s) != alpha G(c, s).

The two snapshots of each pass are kept as well -- the front about to enter
the soil, and the coda reverberating inside it -- so the wavefield panels and
the transfer function on the figure come from one and the same run.

    RUN IN THE CONTAINER:
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -v "$PWD":/work -w /work fenicsx-jupyter python3 run_soil_sh_tf.py

Writes figures/soil_sh_tf.npz, which graphics/make_soil_nonlinear.py reads.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from semwave import DEFAULT_LAYERS, SEMWaveSH2D

FIGDIR = Path(__file__).resolve().parent
GEO = dict(L=2600.0, H=1300.0, nx=52, ny=26, degree=4,
           q_damp=120.0, alpha_damp=1.2, sponge=200.0)
N_GRID = (261, 131)
MEDIUM_SEED = 1000
SIGMA_LN, B1, B2 = 0.15, 250.0, 100.0
SRC_DEPTH = 650.0

TIMES = (0.70, 2.20)        # front reaching the soil; the coda, reverberating
T_END = 6.0                 # 0.17 Hz resolution: the fundamental is at ~1 Hz
LAYER_DEPTH = DEFAULT_LAYERS[0][0]          # the surficial unit, 150 m
YIELD_STRAIN = 1.0e-4                       # soil-like strength, not a fit
LEVELS = (1.0, 6.0, 20.0)                   # peak elastic J2 / sigma_y
# surface and borehole on the same vertical, plus two spares inside the layer
# and deeper in the substratum
STATION_DEPTHS = (1.0, 75.0, 175.0, 300.0)
COL_X = 0.50                # station column, as a fraction of L
SRC_X = COL_X               # the source sits under the column: vertical
                            # incidence, which is how H is measured
CHAB = dict(c_kin=5.0, gamma=100.0, q_iso=0.2, b_iso=150.0,
            k_visc=0.5, n_visc=3.0)


def one_pass(force_amp=None, progress=False, **layer_kw):
    geo = dict(GEO, t_end=T_END)
    sim = SEMWaveSH2D(f0=6.0, **geo)
    if force_amp is not None:
        sim.force_amp = float(force_amp)
    sim.set_medium(np.random.default_rng(MEDIUM_SEED), n_grid=N_GRID,
                   sigma_ln=SIGMA_LN, b1=B1, b2=B2)
    sim.set_plastic_layer(depth=LAYER_DEPTH, **layer_kw)
    stations = [(COL_X * geo["L"], geo["H"] - d) for d in STATION_DEPTHS]
    sim.run(src_xy=(SRC_X * geo["L"], geo["H"] - SRC_DEPTH),
            rec_xy=stations, snap_at=TIMES, progress=progress)
    return sim


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()

    print("pass 0: layer elastic -- calibration and linear reference",
          flush=True)
    el = one_pass(sigma_y=np.inf)
    sigma_y = float(np.sqrt(3.0)
                    * np.median(np.asarray(el.mu_q)[el.active]) * YIELD_STRAIN)
    unit_scale = sigma_y / el.j2_max          # force giving J2 = sigma_y
    print(f"  {el.n_steps} steps, dt = {el.dt:.2e} s, peak J2 = "
          f"{el.j2_max:.3e} Pa at {el.force_amp:.1e} N; sigma_y = "
          f"{sigma_y:.3e} Pa ({time.time() - t0:.0f} s)", flush=True)

    out = dict(times=np.asarray(TIMES), levels=np.asarray(LEVELS),
               layer_depth=np.asarray(LAYER_DEPTH),
               yield_strain=np.asarray(YIELD_STRAIN),
               sigma_y=np.asarray(sigma_y), t_end=np.asarray(T_END),
               H=np.asarray(GEO["H"]), L=np.asarray(GEO["L"]),
               src=np.asarray(el.xdofs[el.i_src]),
               station_depths=np.asarray(STATION_DEPTHS),
               station_xy=el.rec_xy, t=el.time,
               # the elastic pass, renormalised to the weakest drive level so
               # that it is directly comparable with pass 1
               traces_el=el.traces * (LEVELS[0] * unit_scale))
    for j in range(len(TIMES)):
        gx, gy, snap = el.snapshot_grid(which=j)
        out["gx"], out["gy"] = gx, gy
        out[f"snap_el_{j}"] = snap

    for k, lev in enumerate(LEVELS):
        t1 = time.time()
        amp = el.force_amp * lev * unit_scale
        print(f"pass {k + 1}/{len(LEVELS)}: visco-plastic, peak elastic "
              f"J2 = {lev:g} sigma_y (gamma ~ {lev * YIELD_STRAIN:.1e}), "
              f"force {amp:.3e} N", flush=True)
        vp = one_pass(force_amp=amp, yield_strain=YIELD_STRAIN, **CHAB)
        yielded = 100.0 * np.count_nonzero(vp.pacc > 1e-9) / vp.pacc.size
        ratios = [float(np.abs(vp.traces[s]).max()
                        / (np.abs(el.traces[s]).max() * lev * unit_scale))
                  for s in range(len(STATION_DEPTHS))]
        print(f"  max p = {vp.pacc.max():.3e}, yielding at {yielded:.1f}% of "
              f"the quadrature points, {vp.n_yield_events:.3e} yield events; "
              f"peak-velocity ratio vp/elastic "
              + " ".join(f"{d:.0f} m: {r:.3f}"
                         for d, r in zip(STATION_DEPTHS, ratios))
              + f" ({time.time() - t1:.0f} s)", flush=True)
        out[f"traces_{k}"] = vp.traces
        out[f"p_max_{k}"] = np.asarray(vp.pacc.max())
        out[f"yielded_{k}"] = np.asarray(yielded)
        out[f"force_{k}"] = np.asarray(amp)
        for j in range(len(TIMES)):
            _, _, snap = vp.snapshot_grid(which=j)
            out[f"snap_{k}_{j}"] = snap
        px, py, pgrid = vp.plastic_strain_grid()
        out["px"], out["py"] = px, py
        out[f"pgrid_{k}"] = pgrid

    path = os.path.join(FIGDIR, "soil_sh_tf.npz")
    np.savez_compressed(path, **out)
    print(f"cached {path}  ({time.time() - t0:.0f} s total)", flush=True)
