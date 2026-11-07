"""hdr_coda_neumann.png -- the Neumann series, computed term by term.

Notation: G_0 is the background Green kernel, V the scattering potential.
The contrast is quoted as the RMS of delta_c/c over the whole domain,
sqrt(<(sqrt(1+sigma g)-1)^2>), not as a peak: the field is clipped at 2.5
sigma, so peak contrasts are about 2.6 times the number on the axis.
The series itself is typeset on the slide, in LaTeX (sections/hdr_opening.tex);
this figure only carries its three measured panels, colour-keyed to it.

The first version of this figure drew panel (a) as a cartoon: one pulse per
order, each later and smaller than the last, labelled delta_c, delta_c^2,
delta_c^3.  That picture is wrong in the two ways that matter.  The orders are
**not separated in time** -- each one is a full wave train that starts at the
same first arrival -- and the late coda is not "the high-order term", it is
where every order contributes at once and where a truncated series stops
being usable.  Both statements are now measured, on the solver itself.

The terms are produced by marching them together in one time loop
(coda_neumann_data/run_neumann_orders.py):

    M u0'' = -K0 u0 + f,        M un'' = -K0 un - K_delta u_{n-1},

i.e. each order is a wave propagating in the *background*, driven by the
previous order scattering off the heterogeneity.  The perturbation is exactly
linear in the contrast, K_delta = sigma K_g, so u_n(sigma) = sigma^n u_n(1)
holds to four digits (checked), the orders are solved once and rescaled, and
the exact solution is run separately at each contrast.

Panels
  (a) the four terms at delta_c/c = 1.9%, each normalised to its own peak so
      the time structure is visible, with the true amplitude ratio written
      next to it.  They all start at the same first arrival and all run to the
      end of the record: the series orders amplitudes, not arrival times.
  (b) the share of each order, as a function of lapse time: the ratio of
      envelopes |u_n|/|u_0|.  It climbs -- n=3 is 2% of the ballistic term in
      the direct window and 14% in the late coda.  *That* is the sense in
      which late means high order.
  (c) what the truncation costs: the relative error of the N-term partial sum
      against the exact solution, at six contrasts.  Born (N=1) is worth
      0.6% at delta_c/c = 0.5% and 13% at 1.9%; past about 4% the series stops
      converging -- adding terms makes the answer worse, which is the honest
      reason the coda is not done with a truncated Neumann series.

Simulations: semwave 2-D SEM in the fenicsx-jupyter image, homogeneous
background so G_0 is the half-space Green function, extended 2-D fault with
a rupture front, one receiver at the free surface.

Reproduce: coda_neumann_data/run_neumann_orders.py (see semwave/README.md for
the docker invocation), then this script.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{xcolor}"
    r"\definecolor{mk}{HTML}{0072B2}\definecolor{ac}{HTML}{D55E00}")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "coda_neumann_data" / "neumann_orders.npz"
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_coda_neumann.png"

INK, MUTED, ACCENT, MARK = "#1A1A19", "#6E6E68", "#D55E00", "#0072B2"
COLS = (ACCENT, MARK, "#4C9BD4", "#9CC6E5")      # n = 0, 1, 2, 3
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

T_WIN = (0.55, 3.5)              # the record, from just before the first break
EARLY = (0.70, 1.40)             # direct arrivals
LATE = 2.50                      # the late coda starts here
ENV_SMOOTH = 150                 # samples of the envelope moving average

d = np.load(SRC)
t, orders, exact = d["t"], d["orders"], d["exact"]
dc, sigmas = d["dc_rms"], d["sigmas"]
j_ref = int(np.argmin(np.abs(sigmas - float(d["sigma_ref"]))))
n_ord = orders.shape[0]


def envelope(x, n=ENV_SMOOTH):
    return np.convolve(np.abs(hilbert(x)), np.ones(n) / n, mode="same")


def rms(x, m):
    return float(np.sqrt(np.mean(x[m] ** 2)))


# drawn at the size it is used (118 mm on the slide), so the point
# sizes below are the slide's point sizes -- do not rescale it there.
# The Neumann series itself is NOT drawn here: it is set in LaTeX on the
# slide (sections/hdr_opening.tex), where it is readable as mathematics and
# can say that \hat{G}_c is the Fourier transform of the kernel G_c.  The
# colour key is kept: n = 0 in ACCENT, n >= 1 in MARK.
fig = plt.figure(figsize=(4.87, 1.28), dpi=400)
fig.patch.set_facecolor("white")

axA = fig.add_axes([0.038, 0.215, 0.330, 0.700])
axB = fig.add_axes([0.467, 0.215, 0.205, 0.700])
axC = fig.add_axes([0.790, 0.215, 0.195, 0.700])

# ---- (a) the terms themselves, computed ---------------------------------
m = (t >= T_WIN[0]) & (t <= T_WIN[1])
peak0 = np.abs(orders[0]).max()
gap = 2.3
for k in range(n_ord):
    y = orders[k][m] / np.abs(orders[k]).max()
    axA.plot(t[m], y - gap * k, lw=0.45, color=COLS[k])
    lab = (r"$n=0$" if k == 0
           else rf"$n={k}$,\;$\times{np.abs(orders[k]).max() / peak0:.2f}$")
    axA.text(T_WIN[1] - 0.03, -gap * k + 1.25, lab, fontsize=5.4,
             color=COLS[k], ha="right", va="top", path_effects=HALO)

# where each order breaks, and where it peaks: the series orders the
# *centre of mass* of the energy, not the arrival time
breaks, peaks_t = [], []
for k in range(n_ord):
    pk = np.abs(orders[k]).max()
    breaks.append(float(t[np.argmax(np.abs(orders[k]) > 1e-3 * pk)]))
    peaks_t.append(float(t[np.argmax(np.abs(orders[k]))]))
    axA.plot(peaks_t[k], -gap * k + 1.05, "v", ms=2.0, color=COLS[k],
             clip_on=False)
axA.axvline(np.mean(breaks), color=MUTED, lw=0.5, ls=(0, (2.2, 1.8)))
axA.text(np.mean(breaks) - 0.04, -gap * (n_ord - 1) - 1.55,
         rf"same first break, {min(breaks):.2f}--{max(breaks):.2f}\,s",
         fontsize=5.0, color=MUTED, ha="left", va="bottom",
         path_effects=HALO)
axA.text(T_WIN[1] - 0.03, -gap * (n_ord - 1) - 1.55,
         rf"peak drifts {peaks_t[0]:.1f}\,s\,$\to$\,{peaks_t[-1]:.1f}\,s",
         fontsize=5.0, color=MUTED, ha="right", va="bottom",
         path_effects=HALO)
axA.set_xlim(*T_WIN)
axA.set_ylim(-gap * (n_ord - 1) - 1.7, 1.5)
axA.set_yticks([])
axA.set_xlabel(r"$t$ [s]", fontsize=6.4, color=INK, labelpad=0.8)
axA.set_title(r"(a) the terms, computed ($\delta c/c=%.1f\%%$)"
              % (dc[j_ref] * 100), fontsize=6.4, color=INK, pad=3.0)
for sp in ("left", "right", "top"):
    axA.spines[sp].set_visible(False)

# ---- (b) the share of each order grows with lapse time ------------------
env0 = envelope(orders[0])
early = (t >= EARLY[0]) & (t <= EARLY[1])
late = t >= LATE
share = []
for k in range(1, n_ord):
    r = envelope(orders[k]) / env0
    axB.semilogy(t[m], r[m], lw=0.9, color=COLS[k])
    share.append((rms(orders[k], early) / rms(orders[0], early),
                  rms(orders[k], late) / rms(orders[0], late)))
    i_lab = int(np.argmin(np.abs(t - 1.07)))
    axB.text(t[i_lab], r[i_lab] * 1.45, rf"$n={k}$", fontsize=5.4,
             color=COLS[k], ha="center", va="bottom", path_effects=HALO)
axB.axhline(1.0, color=MUTED, lw=0.5, ls=(0, (2.2, 1.8)))
axB.set_xlim(*T_WIN)
axB.set_ylim(2e-3, 2.0)
axB.set_yticks([1e-2, 1e-1, 1.0])
axB.set_yticklabels([r"$1\%$", r"$10\%$", r"$100\%$"])
axB.set_xlabel(r"$t$ [s]", fontsize=6.4, color=INK, labelpad=0.8)
axB.set_ylabel(r"$|u_n|/|u_0|$", fontsize=6.4, color=INK, labelpad=1.5)
axB.set_title(r"(b) later $=$ higher order", fontsize=6.4, color=INK, pad=3.0)

# ---- (c) what the truncation costs --------------------------------------
err = np.zeros((n_ord, dc.size))
for j, s in enumerate(sigmas):
    for N in range(n_ord):
        part = sum((s / sigmas[j_ref]) ** k * orders[k] for k in range(N + 1))
        err[N, j] = np.abs(exact[j] - part).max() / np.abs(exact[j]).max()

x = dc * 100.0
for N in range(n_ord):
    axC.loglog(x, err[N], "o-", ms=1.8, lw=0.9, color=COLS[N],
               label=rf"$N={N}$")
# where adding the first scattering term stops helping: the series diverges
bad = np.where(err[1] > err[0])[0]
if bad.size:
    axC.axvspan(x[bad[0] - 1] * 1.3, x[-1] * 1.4, color=MUTED, alpha=0.11, lw=0)
    axC.text(x[-1] * 1.35, 1.2e-4, "series\ndiverges", fontsize=5.0,
             color=MUTED, ha="right", va="bottom", path_effects=HALO)
axC.axhline(0.1, color=MUTED, lw=0.5, ls=(0, (2.2, 1.8)))
axC.text(x[-1] * 1.35, 0.13, r"$10\%$", fontsize=5.0, color=MUTED,
         ha="right", va="bottom", path_effects=HALO)
axC.text(x[0] * 1.02, err[1, 0] * 0.42, r"Born", fontsize=5.4,
         color=MARK, ha="left", va="top", path_effects=HALO)
axC.set_xlim(x[0] * 0.95, x[-1] * 1.45)
axC.set_ylim(5e-5, 2e2)
axC.set_xticks([0.5, 1, 2, 4, 8, 15])
axC.set_xticklabels(["0.5", "1", "2", "4", "8", "15"])
axC.minorticks_off()
axC.set_xlabel(r"rms $\delta c/c$ [\%]", fontsize=6.4, color=INK, labelpad=1.0)
axC.set_ylabel(r"$\|U-\sum_{n\leq N}u_n\|_\infty/\|U\|_\infty$",
               fontsize=5.6, color=INK, labelpad=1.5)
axC.set_title(r"(c) what truncation costs", fontsize=6.4, color=INK, pad=3.0)
axC.legend(fontsize=4.8, frameon=False, loc="upper left", handlelength=1.0,
           labelspacing=0.1, borderpad=0.05, borderaxespad=0.1, ncol=2,
           columnspacing=0.6)

for ax in (axA, axB, axC):
    ax.set_facecolor("white")
    ax.tick_params(labelsize=5.4, colors=MUTED, length=2.0, width=0.55)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.55)
axA.spines["left"].set_visible(False)

fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
            pad_inches=0.02)
print("wrote", OUT)
print("  order peaks / ballistic: " + ", ".join(
    f"n={k}: {np.abs(orders[k]).max() / peak0:.3f}" for k in range(n_ord)))
print("  rms share early -> late: " + ", ".join(
    f"n={k + 1}: {e:.3f} -> {l:.3f}" for k, (e, l) in enumerate(share)))
print("  truncation error:")
for j in range(dc.size):
    print(f"    dc/c={x[j]:5.2f}%  "
          + "  ".join(f"N={N}: {err[N, j]:.2e}" for N in range(n_ord)))
