"""hdr_soil_nonlinear.png -- what a nonlinear surficial layer does to the
solution operator, measured on a surface-to-borehole SH transfer function.

Notation: the shear modulus is mu (NOT G) -- G and \\hat{G}_c are the Green
kernel and its Fourier transform on the companion frame.

The companion frame shows the forward map in a heterogeneous *elastic* medium:
there is still a kernel, the heterogeneity only moves energy from the direct
wave into the coda.  Here the top 150 m of the very same model follows a
Lemaitre--Chaboche visco-plastic law, and three source amplitudes make the
whole statement.

The experiment is the antiplane (SH) one, because that is the wave a transfer
function is read on: the out-of-plane motion decouples from P-SV, travels at
Vs alone, and the source sits under the station column, so the column is
sampled at vertical incidence.  (The plane-strain version of this run measured
the vertical component of a point source 47 degrees off the vertical, which
mixes P, SV and Rayleigh energy and buries the resonance of the column.)

  (a), (b)  the wavefield of the strongest run: the front about to enter the
            soil, and the coda reverberating inside it.  Two stations on one
            vertical, one at the free surface and one in the elastic
            substratum right below the base of the layer, are the pair whose
            spectral ratio is H: exactly a surface-to-borehole measurement.
  (c)       that H, for the three amplitudes.  The medium c, the path, the
            stations and the source position are identical in the three runs;
            only the amplitude of s changes.  The curves separate, so

                G(c, alpha s) != alpha G(c, s),

            the resonant peaks of the column collapse as the moduli follow the
            strain, and H stops being a property of the medium alone: it
            carries the source with it, so C and S can no longer be separated.
            The weakest run stays at the yield strain and reproduces the
            elastic run to 0.5% of the peak, i.e. it *is* the linear operator
            the companion frame assumes.

Band: the mesh resolves the soft layer up to about 10 Hz (12.5 m node spacing,
Vs = 600 m/s, 5 points per wavelength) and the Ricker carries energy from
about 1.5 Hz, so the panel shows 1.5-9.5 Hz and nothing else; below that the
plastic offset of the strong runs, not the column, sets the spectrum.  The
ratio is smoothed with the usual Konno-Ohmachi log window.

Everything here is computed, not sketched: soil_nonlinear_data/run_soil_sh_tf.py
(the SEM visco-plastic SH solver of Notebooks/hdr_random_wavefield, in the
fenicsx-jupyter image, ~2.5 min) writes soil_sh_tf.npz, this script only reads
it.

    python3 make_soil_nonlinear.py
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "soil_nonlinear_data" / "soil_sh_tf.npz"
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_soil_nonlinear.png"

INK, MUTED, ACCENT, MARK = "#1A1A19", "#6E6E68", "#D55E00", "#0072B2"
GREEN = "#009E73"
CMAP_WAVE = "magma"                 # the colours of hdr_ballistic_coda.png
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
DARK = [pe.withStroke(linewidth=1.5, foreground="black", alpha=0.6)]

I_SURF, I_BORE = 0, 2               # stations at 1 m and 175 m depth
FBAND = (1.5, 9.5)                  # resolved by the mesh, excited by the source
CROP_X = 280.0                      # margin cropped off the snapshots [m]
KO_B = 30.0                         # Konno-Ohmachi smoothing coefficient
COLS = (GREEN, MARK, ACCENT)

d = np.load(SRC)
t = d["t"]
dt = float(t[1] - t[0])
levels = d["levels"]
gamma_y = float(d["yield_strain"])
layer_depth, Hbox = float(d["layer_depth"]), float(d["H"])
interfaces = (150.0, 400.0, 800.0)  # random_field.DEFAULT_LAYERS
k_strong = len(levels) - 1


# ---------------------------------------------------------------- spectra
def konno_ohmachi(f, A, b=KO_B):
    """Smooth |A|(f) with the standard log window: constant resolution per
    octave, which is how a transfer function is read."""
    out = np.empty_like(A)
    pos = f > 0
    for i, fc in enumerate(f):
        if not pos[i]:
            out[i] = A[i]
            continue
        x = np.ones_like(f)
        x[pos] = f[pos] / fc
        lg = b * np.log10(x)
        w = np.where(np.abs(lg) < 1e-6, 1.0,
                     (np.sin(lg) / np.where(lg == 0.0, 1.0, lg)) ** 4)
        w[~pos] = 0.0
        out[i] = float((w * A).sum() / w.sum())
    return out


def transfer(tr):
    """Surface-to-borehole spectral ratio of one run."""
    f = np.fft.rfftfreq(t.size, dt)
    H = (konno_ohmachi(f, np.abs(np.fft.rfft(tr[I_SURF])))
         / konno_ohmachi(f, np.abs(np.fft.rfft(tr[I_BORE]))))
    band = (f >= FBAND[0]) & (f <= FBAND[1])
    return f[band], H[band]


def wave_norm(snap, pct=99.5, gamma=0.5):
    return mcolors.PowerNorm(gamma=gamma, vmin=0.0,
                             vmax=float(np.percentile(np.abs(snap), pct)))


# ----------------------------------------------------------------- canvas
fig = plt.figure(figsize=(4.92, 1.62), dpi=400)
fig.patch.set_facecolor("white")

gx, gy = d["gx"], d["gy"]
# the margins of the domain carry nothing (the source and the station column
# sit at mid-width): cropping them makes the panels taller, which is the only
# vertical room this three-in-a-row layout has
keep = (gx >= CROP_X) & (gx <= float(d["L"]) - CROP_X)
gx = gx[keep]
extent = [gx[0] / 1e3, gx[-1] / 1e3, gy[0] / 1e3, gy[-1] / 1e3]
ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])

w_pan = 0.300                                   # panel width, figure fraction
h_pan = w_pan * fig.get_figwidth() / ratio / fig.get_figheight()
y_pan = 0.150
ax_a = fig.add_axes([0.004, y_pan, w_pan, h_pan])
ax_b = fig.add_axes([0.312, y_pan, w_pan, h_pan])
ax_c = fig.add_axes([0.735, y_pan + 0.040, 0.250, h_pan - 0.040])

# ------------------------------------- (a), (b) the wavefield of the strong run
times = d["times"]
st = d["station_xy"]
x_col = st[I_SURF, 0] / 1e3
labels = ("(a) SH front, entering the soil", "(b) coda, trapped in the soil")
for j, ax in enumerate((ax_a, ax_b)):
    snap = d[f"snap_{k_strong}_{j}"][:, keep]
    ax.imshow(snap, origin="lower", cmap=CMAP_WAVE, norm=wave_norm(snap),
              extent=extent, aspect="auto", interpolation="bilinear")
    # the layered cake, with the nonlinear unit marked out of it
    for depth in interfaces[1:]:
        y = (Hbox - depth) / 1e3
        if extent[2] < y < extent[3]:
            ax.axhline(y, color="white", lw=0.4, ls=(0, (4, 3)), alpha=0.40)
    y_nl = (Hbox - layer_depth) / 1e3
    ax.axhspan(y_nl, extent[3], color="white", alpha=0.16, lw=0)
    ax.axhline(y_nl, color="white", lw=0.9, ls=(0, (2.2, 1.3)), alpha=0.95)

    # the source, and the column the two stations read
    src = d["src"]
    ax.plot(src[0] / 1e3, src[1] / 1e3, "*", ms=5.5, color="white",
            mec="black", mew=0.35)
    ax.plot([x_col, x_col], [st[I_BORE, 1] / 1e3, st[I_SURF, 1] / 1e3],
            color="white", lw=0.7, ls=(0, (1.3, 1.1)), alpha=0.95)
    ax.plot(x_col, st[I_SURF, 1] / 1e3, "v", ms=4.4, color="white",
            mec="black", mew=0.35, clip_on=False)
    ax.plot(st[I_BORE, 0] / 1e3, st[I_BORE, 1] / 1e3, "o", ms=3.4,
            color="white", mec="black", mew=0.35)
    ax.text(0.035, 0.065, rf"$t={times[j]:.2f}$ s", color="white",
            fontsize=5.6, transform=ax.transAxes, path_effects=DARK)
    ax.set_title(labels[j], fontsize=6.4, color=INK, pad=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)

# the labels that make the pair readable, written once, where the panel is dark
ax_a.text(0.035, 0.885, r"nonlinear soil, 150\,m", color="white", fontsize=5.4,
          transform=ax_a.transAxes, path_effects=DARK)
ax_a.annotate(r"$u_{\mathrm{surf}}$",
              xy=(x_col, st[I_SURF, 1] / 1e3), xytext=(-4.0, -2.0),
              textcoords="offset points", color="white", fontsize=5.4,
              ha="right", va="center", path_effects=DARK)
ax_a.annotate(r"$u_{\mathrm{bore}}$",
              xy=(st[I_BORE, 0] / 1e3, st[I_BORE, 1] / 1e3),
              xytext=(-4.0, -1.0), textcoords="offset points", color="white",
              fontsize=5.4, ha="right", va="center", path_effects=DARK)
ax_a.text(0.965, 0.065, rf"$\times{levels[k_strong] / levels[0]:.0f}$ run",
          color=ACCENT, fontsize=5.4, ha="right", transform=ax_a.transAxes,
          path_effects=DARK)

# ----------------------------------------- (c) the operator carries the source
curves, peaks = [], []
for k in range(len(levels)):
    f_k, H_k = transfer(d[f"traces_{k}"])
    lab = (r"weak $s$: linear" if k == 0
           else rf"$\times{levels[k] / levels[0]:.0f}$")
    ax_c.plot(f_k, H_k, lw=1.1, color=COLS[k], zorder=3 + k, label=lab)
    curves.append((f_k, H_k))
    peaks.append((float(f_k[np.argmax(H_k)]), float(H_k.max())))

# the resonance of the column: visible on the linear run, and on it only
f_lin, H_lin_c = curves[0]
loc = np.where((H_lin_c[1:-1] > H_lin_c[:-2])
               & (H_lin_c[1:-1] > H_lin_c[2:]))[0] + 1
ax_c.plot(f_lin[loc], H_lin_c[loc] + 0.17, marker="v", ls="none", ms=2.0,
          color=GREEN, mec="none", zorder=5, clip_on=False)
ax_c.text(0.985, 0.86, "modes of the column", fontsize=5.0, color=GREEN,
          ha="right", va="top", transform=ax_c.transAxes, zorder=7,
          path_effects=HALO)

# the measurement the panel is about: at the resonance of the linear column,
# what the same column does to a source 20 times bigger
f_res = peaks[0][0]
i_res = int(np.argmin(np.abs(curves[0][0] - f_res)))
H_lin, H_str = curves[0][1][i_res], curves[-1][1][i_res]
ax_c.annotate("", xy=(f_res, H_str + 0.04), xytext=(f_res, H_lin - 0.04),
              arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color=MUTED,
                              mutation_scale=4.0), zorder=6)
ax_c.annotate(rf"$\div{H_lin / H_str:.1f}$", xy=(f_res, 0.5 * (H_lin + H_str)),
              xytext=(7.0, 1.0), textcoords="offset points", fontsize=5.4,
              color=MUTED, ha="left", va="center", path_effects=HALO, zorder=7)

ax_c.set_xlim(*FBAND)
ax_c.set_ylim(0.0, 3.2)
ax_c.set_xticks([2, 4, 6, 8])
ax_c.set_xlabel("frequency [Hz]", fontsize=6.4, color=INK, labelpad=1.0)
ax_c.set_ylabel(r"$|H|=|\hat u_{\mathrm{surf}}/\hat u_{\mathrm{bore}}|$",
                fontsize=5.8, color=INK, labelpad=1.5)
ax_c.set_title(r"(c) $\mathcal{G}\to\mathcal{G}_t$: $C,S$ inseparable",
               fontsize=6.4, color=INK, pad=3.0)
ax_c.legend(fontsize=5.0, frameon=False, loc="upper left", handlelength=1.1,
            labelspacing=0.12, borderpad=0.05, borderaxespad=0.1)

ax_c.set_facecolor("white")
ax_c.tick_params(labelsize=5.4, colors=MUTED, length=2.0, width=0.55)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
for sp in ("left", "bottom"):
    ax_c.spines[sp].set_color(MUTED); ax_c.spines[sp].set_linewidth(0.55)

fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
            pad_inches=0.02)
print("wrote", OUT)
print("  peak of |H| in %g-%g Hz: " % FBAND + ", ".join(
    rf"x{levels[k] / levels[0]:.0f}: {p[1]:.2f} at {p[0]:.2f} Hz"
    for k, p in enumerate(peaks)))
print("  weak run vs elastic run: "
      f"{np.abs(d['traces_0'] - d['traces_el']).max() / np.abs(d['traces_el']).max():.3f}"
      " of the peak (the linear limit is recovered)")
print("  peak accumulated plastic strain: " + ", ".join(
    f"x{levels[k] / levels[0]:.0f}: {float(d[f'p_max_{k}']):.2e}"
    for k in range(len(levels))))
