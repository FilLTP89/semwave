"""hdr_aniso_fabric.png -- the parameter of the NDT forward map: a grain fabric.

The seismological frame has a scalar wave speed c(x) with a random fluctuation.
In a laser-ultrasound inspection of a wire-laser additively manufactured
(WLAM) part the parameter is a different object: the elastic **tensor** field,
carried by a polycrystal whose grains are elongated by the thermal gradient of
the process and whose crystal axes are rotated grain by grain.

The panel is the input of the simulations behind hdr_aniso_coda.png, drawn
from the generator itself (aniso_ndt_data/run_aniso_ndt.py), with the setup of
the EAAI paper: 25 x 10 mm of IN718, ~60 Voronoi grains elongated by k_el
along alpha_el, crystal orientation phi ~ U(-45, 45) deg, an ablation-regime
laser source at the centre of the free surface and 41 receivers within +-5 mm
of it.  The colour is phi, and it is cyclic: cubic crystals are pi/2-periodic,
so -45 and +45 degrees are the same crystal.

The rose on the right is the Christoffel solution of that single crystal at
A = 2.5: the qP and qSV phase velocities against the propagation direction.
It is why the kernel is directional -- 5.5 to 6.3 km/s for qP, 2.5 to 3.9 km/s
for qSV, with the extrema of the two swapped, which is what cusps the inner
wavefront.

    python3 make_aniso_fabric.py
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "aniso_ndt_data" / "aniso_ndt.npz"
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_aniso_fabric.png"

INK, MUTED, ACCENT, MARK = "#1A1A19", "#6E6E68", "#D55E00", "#0072B2"
GREEN = "#009E73"
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
DARK = [pe.withStroke(linewidth=1.4, foreground="black", alpha=0.6)]

d = np.load(SRC)
phi, x1, x2 = d["phi"], d["x1"], d["x2"]
src, rec_x = d["src"], d["rec_x"]
ang, v_qp, v_qs = d["angles"], d["v_qp"], d["v_qs"]
A_max = float(d["poly_A"].max())

# drawn at the size it is used (61 mm on the slide): the point sizes
# below are the slide's point sizes, so do not rescale it there
fig = plt.figure(figsize=(2.24, 0.93), dpi=400)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.070, 0.135, 0.580, 0.700])
cax = fig.add_axes([0.660, 0.135, 0.013, 0.700])
rose = fig.add_axes([0.775, 0.115, 0.225, 0.740], projection="polar")

# ---- the fabric ---------------------------------------------------------
im = ax.imshow(np.rad2deg(phi), origin="upper", cmap="twilight",
               vmin=-45.0, vmax=45.0, aspect="auto", interpolation="nearest",
               extent=[x1[0] * 1e3, x1[-1] * 1e3, x2[-1] * 1e3, x2[0] * 1e3])
ax.set_xlim(x1[0] * 1e3, x1[-1] * 1e3)
ax.set_ylim(10.0, 0.0)                       # depth downwards, free surface up
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_yticks([0, 5, 10])
ax.set_xlabel(r"surface position [mm]", fontsize=5.4, color=INK, labelpad=0.6)
ax.set_ylabel(r"depth [mm]", fontsize=5.4, color=INK, labelpad=0.8)
ax.tick_params(labelsize=4.8, colors=MUTED, length=1.6, width=0.5)

# the laser source and the receiver line, the way the B-scan is acquired
ax.annotate("", xy=(src[0] * 1e3, 0.35), xytext=(src[0] * 1e3, -2.3),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ACCENT,
                            mutation_scale=4.0))
ax.text(src[0] * 1e3 + 0.5, -2.1, "laser", fontsize=5.0, color=ACCENT,
        ha="left", va="top", clip_on=False)
ax.plot([rec_x[0] * 1e3, rec_x[-1] * 1e3], [-0.9, -0.9], lw=0.9, color=GREEN,
        clip_on=False, solid_capstyle="butt")
ax.plot(rec_x[::4] * 1e3, np.full(rec_x[::4].size, -0.9), "v", ms=1.4,
        color=GREEN, clip_on=False)
ax.text(rec_x[0] * 1e3 - 0.4, -0.9, r"41 rec.", fontsize=5.0, color=GREEN,
        ha="right", va="center", clip_on=False)

cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$\phi$ [$^\circ$]", fontsize=4.8, color=INK, labelpad=0.4)
cb.set_ticks([-45, 0, 45])
cb.ax.tick_params(labelsize=4.4, colors=MUTED, length=1.2, width=0.4, pad=0.8)
cb.outline.set_linewidth(0.4)
cb.outline.set_edgecolor(MUTED)

# ---- the Christoffel rose of one crystal --------------------------------
th = np.concatenate([ang, ang + np.pi])
for v, c, lab in ((v_qp, MARK, "qP"), (v_qs, ACCENT, "qSV")):
    r = np.concatenate([v, v]) / 1e3
    rose.plot(th, r, lw=0.8, color=c)
    rose.text(np.deg2rad(45 if lab == "qP" else 225),
              (v.max() if lab == "qP" else v.min()) / 1e3 * 1.02,
              lab, fontsize=4.8, color=c, ha="center", va="center",
              path_effects=HALO)
rose.set_rmax(7.0)
rose.set_rticks([3, 6])
rose.set_rlabel_position(118.0)
rose.set_yticklabels([r"3", r"6\,km/s"], fontsize=4.0, color=MUTED)
rose.set_xticks(np.deg2rad([0, 90, 180, 270]))
rose.set_xticklabels([])
rose.tick_params(pad=-1.0)
rose.grid(lw=0.3, color=MUTED, alpha=0.5)
rose.spines["polar"].set_linewidth(0.4)
rose.spines["polar"].set_color(MUTED)
rose.set_title(rf"one crystal, $A={A_max:g}$", fontsize=5.0, color=INK,
               pad=1.5)

for sp in ax.spines.values():
    sp.set_color(MUTED); sp.set_linewidth(0.5)

fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
            pad_inches=0.02)
print("wrote", OUT)
print(f"  phi in [{np.rad2deg(phi).min():.1f}, {np.rad2deg(phi).max():.1f}] deg, "
      f"{int(d['n_grains'])} grains, k_el={float(d['k_el']):g}, "
      f"alpha_el={np.rad2deg(float(d['alpha_el'])):.0f} deg")
print(f"  qP {v_qp.min():.0f}-{v_qp.max():.0f} m/s, "
      f"qSV {v_qs.min():.0f}-{v_qs.max():.0f} m/s")
