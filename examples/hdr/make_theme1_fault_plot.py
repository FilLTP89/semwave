#!/usr/bin/env python3
"""Theme-1 divider raster: heterogeneous medium, wave snapshot, GLL spectral
mesh, absorbing layer and the fault.  Deliberately carries NO text -- every
label is placed in LaTeX by \themecardsoloannot, using the normalised anchors
printed at the end of this script.

Design rules, so the card stays readable at 43 mm height:
  * the medium never goes darker than mid grey, so black LaTeX labels read
    anywhere on it; its structure is carried by iso-Vs contours, not by tone;
  * the wavefield is translucent, so the medium shows through it;
  * the full-domain GLL sub-grid is NOT drawn (it is invisible at this size
    and reads as graph paper); element edges are drawn, and the GLL pattern
    is shown magnified in a callout;
  * the absorbing layer is a hatched band, not a black veil.
"""
from pathlib import Path

import numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

mpl.rcParams.update({"pdf.fonttype": 42, "hatch.linewidth": 0.30})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
d = np.load(HERE / "fault_snap.npz")
L, H = float(d["L"]), float(d["H"])
NX, NY, DEG, SP = int(d["NX"]), int(d["NY"]), int(d["DEG"]), float(d["sponge"])
fa, fb = d["fault_a"], d["fault_b"]

# ---- full domain, absorbing layer included -------------------------------
gx = np.linspace(0.0, L, 1000)
gy = np.linspace(0.0, H, 640)
GX, GY = np.meshgrid(gx, gy)
snap = griddata(d["xdofs"], d["snap"], (GX, GY), method="linear", fill_value=0.0)
vs = griddata(np.column_stack([np.repeat(d["x1"], d["x2"].size),
                               np.tile(d["x2"], d["x1"].size)]),
              d["vs"].T.ravel(), (GX, GY), method="linear")
snap, vs = snap[::-1], vs[::-1]        # row 0 must be the free surface
ext = [0.0, L / 1e3, H / 1e3, 0.0]     # x km, depth km (top-down)

# ---- the medium: light tones only, structure carried by contours ---------
cmap_vs = LinearSegmentedColormap.from_list(
    "vslight", ["#ffffff", "#eeedea", "#d6d5ce", "#bab9b1", "#9d9c95"])

fig, ax = plt.subplots(figsize=(3.7, 2.36))
ax.imshow(vs, extent=ext, aspect="auto", cmap=cmap_vs, origin="upper",
          vmin=np.nanmin(vs), vmax=np.nanmax(vs), interpolation="bilinear",
          zorder=1)
vs_s = gaussian_filter(np.nan_to_num(vs, nan=np.nanmean(vs)), sigma=11.0)
lev = np.nanquantile(vs_s, [0.22, 0.48, 0.72, 0.90])
ax.contour(np.linspace(0, L / 1e3, vs.shape[1]),
           np.linspace(0, H / 1e3, vs.shape[0]), vs_s, levels=lev,
           colors="#3f3e3a", linewidths=0.32, alpha=0.42, zorder=2)

# ---- the wavefield: translucent, so the medium survives underneath -------
w = snap / (np.abs(snap).max() + 1e-30)
ax.imshow(w, extent=ext, aspect="auto", cmap="RdBu_r", origin="upper",
          vmin=-0.55, vmax=0.55, alpha=np.clip(np.abs(w) * 3.2, 0, 0.62),
          interpolation="bilinear", zorder=3)

# ---- the spectral-element mesh: element edges only ----------------------
hx, hy = L / NX, H / NY
for i in range(NX + 1):
    ax.axvline(i * hx / 1e3, color="#1a1a1a", lw=0.45, alpha=0.40, zorder=4)
for j in range(NY + 1):
    ax.axhline(j * hy / 1e3, color="#1a1a1a", lw=0.45, alpha=0.40, zorder=4)

# ---- absorbing (Cerjan) layer: hatched band, left, right and bottom -----
for xy, wdt, hgt in (((0.0, 0.0), SP, H),
                     ((L - SP, 0.0), SP, H),
                     ((0.0, H - SP), L, SP)):
    xy_km = (xy[0] / 1e3, xy[1] / 1e3)
    ax.add_patch(Rectangle(xy_km, wdt / 1e3, hgt / 1e3, fc="white",
                           alpha=0.42, ec="none", zorder=5))
    ax.add_patch(Rectangle(xy_km, wdt / 1e3, hgt / 1e3, fc="none",
                           ec="#55544f", hatch="//", lw=0.0, alpha=0.30,
                           zorder=6))
ax.plot([SP / 1e3, SP / 1e3, (L - SP) / 1e3, (L - SP) / 1e3],
        [0.0, (H - SP) / 1e3, (H - SP) / 1e3, 0.0],
        ls=(0, (4, 2.5)), lw=0.6, color="#1a1a1a", alpha=0.9, zorder=7)

# ---- GLL callout: one element, magnified, with its 5x5 nodes ------------
gll = (np.array([-1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7), 1.0]) + 1) / 2
EI, EJ = 9, 4                                   # source element (col, row)
sx0, sx1 = EI * hx / 1e3, (EI + 1) * hx / 1e3   # km
sy0, sy1 = EJ * hy / 1e3, (EJ + 1) * hy / 1e3   # km, depth-down
bx0, bx1, by0, by1 = 2.95, 3.55, 0.14, 0.74     # callout box, km / depth km

ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, fc="none",
                       ec="#1a1a1a", lw=0.7, zorder=9))
for p, q in (((sx1, sy0), (bx0, by0)), ((sx1, sy1), (bx0, by1))):
    ax.plot([p[0], q[0]], [p[1], q[1]], lw=0.4, color="#1a1a1a", alpha=0.55,
            zorder=9)

sub = (slice(int(sy0 / (H / 1e3) * vs.shape[0]), int(sy1 / (H / 1e3) * vs.shape[0])),
       slice(int(sx0 / (L / 1e3) * vs.shape[1]), int(sx1 / (L / 1e3) * vs.shape[1])))
bext = [bx0, bx1, by1, by0]
ax.imshow(vs[sub], extent=bext, aspect="auto", cmap=cmap_vs, origin="upper",
          vmin=np.nanmin(vs), vmax=np.nanmax(vs), interpolation="bilinear",
          zorder=10)
ax.imshow(w[sub], extent=bext, aspect="auto", cmap="RdBu_r", origin="upper",
          vmin=-0.55, vmax=0.55,
          alpha=np.clip(np.abs(w[sub]) * 3.2, 0, 0.62),
          interpolation="bilinear", zorder=11)

gx_b = bx0 + gll * (bx1 - bx0)
gy_b = by0 + gll * (by1 - by0)
for xv in gx_b:
    ax.plot([xv, xv], [by0, by1], lw=0.32, color="#5d5c57", alpha=0.85,
            zorder=12)
for yv in gy_b:
    ax.plot([bx0, bx1], [yv, yv], lw=0.32, color="#5d5c57", alpha=0.85,
            zorder=12)
NG, MG = np.meshgrid(gx_b, gy_b)
ax.plot(NG.ravel(), MG.ravel(), "o", ms=2.6, mfc="#111111", mec="white",
        mew=0.35, ls="none", zorder=13)
ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc="none",
                       ec="#111111", lw=0.9, zorder=14))

# ---- fault, hypocentre exactly at its lower tip --------------------------
fx = [fa[0] / 1e3, fb[0] / 1e3]
fy = [(H - fa[1]) / 1e3, (H - fb[1]) / 1e3]
ax.plot(fx, fy, "-", lw=2.6, color="w", alpha=0.95, solid_capstyle="round",
        zorder=13)
ax.plot(fx, fy, "-", lw=1.4, color="#111111", solid_capstyle="round", zorder=14)
ax.plot(fx[0], fy[0], "*", ms=7.5, mfc="#E9A23B", mec="#111111", mew=0.6,
        zorder=15, clip_on=False)

ax.plot([0, L / 1e3], [0, 0], color="#111111", lw=1.3, zorder=8)
ax.set_xlim(0, L / 1e3); ax.set_ylim(H / 1e3, 0)
ax.set_axis_off()
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(ROOT / "docs" / "figures" / "hdr" / "hdr_theme1_fault_card.png",
            dpi=600)


def nrm(x, y):
    return x / L, y / H          # TikZ: (0,0) bottom-left, (1,1) top-right


def nrm_km(xkm, ykm_depth):      # km, depth-down  ->  TikZ normalised
    return xkm * 1e3 / L, 1.0 - ykm_depth * 1e3 / H


print("\n--- normalised anchors for \\themecardsoloannot ---")
print("hypocentre (fault lower tip) : (%.4f,%.4f)" % nrm(*fa))
print("fault upper tip              : (%.4f,%.4f)" % nrm(*fb))
print("fault midpoint               : (%.4f,%.4f)" % nrm(*(0.5 * (fa + fb))))
print("absorbing layer, inner edge  : x=%.4f and x=%.4f, y=%.4f"
      % (SP / L, 1 - SP / L, SP / H))
print("absorbing band centre  left  : (%.4f,%.4f)" % (0.5 * SP / L, 0.50))
print("                      right  : (%.4f,%.4f)" % (1 - 0.5 * SP / L, 0.50))
print("                     bottom  : (%.4f,%.4f)" % (0.50, 0.5 * SP / H))
print("GLL callout, box centre      : (%.4f,%.4f)"
      % nrm_km(0.5 * (bx0 + bx1), 0.5 * (by0 + by1)))
print("GLL callout, below the box   : (%.4f,%.4f)"
      % nrm_km(0.5 * (bx0 + bx1), by1 + 0.04))
print("interior Omega (clear spot)  : (%.4f,%.4f)" % (0.80, 0.30))
