"""hdr_ballistic_coda.png -- the ballistic front, the coda, and the record.

Three views of ONE elastic run on a medium that is heterogeneous *and*
anisotropic (ballistic_coda_data/run_ballistic_coda.py):

  (a) the compact front leaving the buried source, as it reaches the station
      -- the n = 0 term of the Neumann series of the figure below it on the
      slide, and orange for that reason;
  (b) the same realisation two seconds later: no front anywhere, the energy
      spread over the whole model -- the n >= 1 terms, in blue;
  (c) what a station on the free surface records.  The same two colours split
      the trace, at one dominant period past the straight-ray S arrival: the
      direct arrivals, then everything the heterogeneity made of them.

The medium is linear elastic throughout: the point of the slide is that the
front is destroyed without any nonlinearity, the map still a convolution and
only the kernel unrecognisable.  (The nonlinear soil layer is the next slide,
and a separate run.)

Both snapshots are scaled to their own peak -- the amplitudes are the business
of (c), and (b) is annotated with the real ratio.

The figure is drawn at the size it is used, FIG_MM wide, so that a font size
here is the font size on the slide; nothing is rescaled by \\includegraphics.

Reproduce: ballistic_coda_data/run_ballistic_coda.py in the fenicsx-jupyter
image (see semwave/README.md for the docker invocation), then this script.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "ballistic_coda_data" / "ballistic_coda.npz"
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_ballistic_coda.png"

INK, MUTED, ACCENT, MARK = "#1A1A19", "#6E6E68", "#D55E00", "#0072B2"
FIG_MM = 60.0                  # \includegraphics[width=60mm] on the slide
T_A, T_B = 0.60, 2.40          # the two instants shown, out of the stored movie
CROP_Y = 420.0                 # the bottom of the model is sponge and darkness
HALO = [pe.withStroke(linewidth=1.3, foreground="black")]

d = np.load(SRC)
gx, gy, snaps, snap_t = d["gx"], d["gy"], d["snaps"], d["snap_times"]
keep = gy >= CROP_Y                     # keep the near-surface window
gy, snaps = gy[keep], snaps[:, keep, :]
t, v = d["time"], d["trace_z"][0]
src, rec = d["src"], d["rec"][0]
H, f0 = float(d["H"]), float(d["f0"])
t_p, t_s = float(d["t_p_ray"]), float(d["t_s_ray"])
ja, jb = int(np.argmin(abs(snap_t - T_A))), int(np.argmin(abs(snap_t - T_B)))

# the direct arrivals end one dominant period after the straight-ray S; what
# follows is coda and surface waves
t_split = t_s + 1.0 / f0
is_bal = t <= t_split
share = (v[~is_bal] ** 2).sum() / (v ** 2).sum()
ratio = snaps[jb].max() / snaps[ja].max()
print(f"  split at {t_split:.2f} s; coda carries {100*share:.0f}% of the "
      f"recorded energy; peak(b)/peak(a) = {ratio:.2f}")


def wave_norm(snap, pct=99.8, gamma=0.78):
    """Power-law norm clipped at a high percentile, so that one setting serves
    an early peaky snapshot and a late diffuse one."""
    return mcolors.PowerNorm(gamma, vmin=0.0, vmax=np.percentile(snap, pct))


# ------------------------------------------------------------------- canvas
W = FIG_MM / 25.4
left, right, gap, top_pad = 0.105, 0.035, 0.055, 0.100     # inches
trace_h, foot = 0.205, 0.200
pw = (W - left - right - gap) / 2.0                        # panel width
ph = pw * (gy[-1] - gy[0]) / (gx[-1] - gx[0])              # true aspect
HT = top_pad + ph + 0.055 + trace_h + foot
fig = plt.figure(figsize=(W, HT), dpi=600)
fig.patch.set_facecolor("white")

py = (HT - top_pad - ph) / HT
rects = [[left / W, py, pw / W, ph / HT],
         [(left + pw + gap) / W, py, pw / W, ph / HT]]
ax_t = fig.add_axes([left / W, foot / HT, (W - left - right) / W, trace_h / HT])

extent = [gx[0] / 1e3, gx[-1] / 1e3, gy[0] / 1e3, gy[-1] / 1e3]
titles = (r"(a) ballistic front", r"(b) coda $+$ surface waves")
for k, (j, rect, colour) in enumerate(zip((ja, jb), rects, (ACCENT, MARK))):
    ax = fig.add_axes(rect)
    snap = snaps[j]
    ax.imshow(snap, origin="lower", cmap="magma", norm=wave_norm(snap),
              extent=extent, aspect="auto", interpolation="bilinear")
    # the layered cake the fluctuations are drawn on
    for z in d["layer_tops"][1:]:
        y = (H - z) / 1e3
        if extent[2] < y < extent[3]:
            ax.axhline(y, color="white", lw=0.3, ls=(0, (3.5, 3)), alpha=0.35)
    ax.plot(src[0] / 1e3, src[1] / 1e3, "*", ms=4.0, color="white",
            mec="black", mew=0.3)
    ax.plot(rec[0] / 1e3, rec[1] / 1e3 + 0.022, "v", ms=3.4, color="white",
            mec="black", mew=0.35, clip_on=False, zorder=5)
    ax.text(0.03, 0.08, rf"$t={snap_t[j]:.2f}$ s", color="white",
            fontsize=5.0, transform=ax.transAxes, path_effects=HALO)
    ax.set_title(titles[k], fontsize=5.7, pad=1.6, color=colour)
    for sp in ax.spines.values():
        sp.set_color(colour); sp.set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])

    if k == 0:
        ax.text(src[0] / 1e3 + 0.035, src[1] / 1e3, "source", color="white",
                fontsize=4.6, va="center", path_effects=HALO)
        ax.text(rec[0] / 1e3 - 0.03, rec[1] / 1e3 - 0.075, "station",
                color="white", fontsize=4.6, ha="right", va="center",
                path_effects=HALO)
        # (the fabric is stated once, under the record: a glyph small enough
        #  to fit this panel is a glyph nobody can read)
    else:
        ax.text(0.97, 0.08, rf"peak $\times{ratio:.2f}$", color="white",
                fontsize=4.8, ha="right", transform=ax.transAxes,
                path_effects=HALO)

# ------------------------------------------------------- (c) the seismogram
vn = v / np.abs(v).max()
ax_t.plot(t[is_bal], vn[is_bal], lw=0.45, color=ACCENT)
ax_t.plot(t[~is_bal], vn[~is_bal], lw=0.45, color=MARK)
ax_t.axvspan(t[0], t_split, color=ACCENT, alpha=0.09, lw=0)
ax_t.axvspan(t_split, t[-1], color=MARK, alpha=0.09, lw=0)
ax_t.axhline(0, color=MUTED, lw=0.3)

for tt, lab in ((t_p, "P"), (t_s, "S")):
    ax_t.plot([tt, tt], [-1.30, -1.02], lw=0.45, color=MUTED, clip_on=False)
    ax_t.text(tt, -1.40, lab, fontsize=4.6, color=MUTED, ha="center", va="top")
for tt, lab, colour in ((snap_t[ja], "(a)", ACCENT), (snap_t[jb], "(b)", MARK)):
    ax_t.plot([tt], [1.26], marker="v", ms=2.2, color=colour, clip_on=False)
    ax_t.text(tt + 0.04, 1.20, lab, fontsize=4.8, color=colour, va="center")

ax_t.text(0.04, -0.80, "direct arrivals", fontsize=5.0, color=ACCENT,
          ha="left", va="center", fontweight="bold")
ax_t.text(t[-1] - 0.04, -0.80, rf"coda: {100*share:.0f}\,\% of the energy",
          fontsize=5.0, color=MARK, ha="right", va="center", fontweight="bold")
ax_t.set_xlim(t[0], t[-1]); ax_t.set_ylim(-1.3, 1.3)
ax_t.set_xticks(np.arange(0.0, 3.1, 1.0))
ax_t.set_xticks(np.arange(0.5, 3.6, 1.0), minor=True)
ax_t.set_yticks([])
ax_t.text(1.0, -0.40, "time [s]", fontsize=5.4, color=INK, ha="right",
          va="top", transform=ax_t.transAxes)
ax_t.set_ylabel(r"$v_z$", fontsize=5.4, color=INK, labelpad=1.2, rotation=0,
                va="center")
ax_t.tick_params(labelsize=4.8, colors=MUTED, length=1.4, width=0.4, pad=0.8)
ax_t.tick_params(which="minor", colors=MUTED, length=0.8, width=0.35)
for sp in ("top", "right", "left"):
    ax_t.spines[sp].set_visible(False)
ax_t.spines["bottom"].set_color(MUTED); ax_t.spines["bottom"].set_linewidth(0.4)
ax_t.set_facecolor("white")

# the medium, stated once, on the line the axis label leaves free
fig.text(0.008, 0.02,
         rf"elastic, heterogeneous ($\sigma={100*float(d['sigma_ln']):.0f}$\,\%) "
         rf"$+$ TI fabric ($\varepsilon={float(d['epsilon_0']):.2f}$, "
         rf"dip ${np.rad2deg(float(d['theta'])):.0f}^\circ$)",
         fontsize=4.3, color=MUTED, ha="left", va="bottom")

fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote", OUT, f"{W*25.4:.1f} x {HT*25.4:.1f} mm")
