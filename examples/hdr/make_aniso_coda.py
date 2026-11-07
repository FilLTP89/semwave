"""hdr_aniso_coda.png -- what the grain fabric writes into the ultrasonic record.

Companion of hdr_aniso_fabric.png, same runs (aniso_ndt_data/run_aniso_ndt.py).
The single-crystal constants are fixed so that the **Hill average of the
texture is the same isotropic medium whatever the Zener ratio A**, so every
run here has the same mean medium, the same source and the same stations:
anything that changes is the texture speaking.

  (a), (b) the B-scan -- 41 surface receivers, which is what a laser
           interferometer delivers -- for A = 1 and A = 2.5, on one colour
           scale.  The Rayleigh "V" arrives at the same time in both; it is
           weaker in (b), and what fills in behind it is the coda the grains
           build out of the energy it lost.
  (c)      one A-scan, at +3 mm.  Same arrival time to a few tens of ns (the
           mean medium is the same), a fraction of the amplitude, and a tail
           that does not stop.
  (d)      both windows against A, normalised by the direct energy of the
           isotropic run: the ballistic term is depleted and the coda is fed
           -- the same statement as the seismological frame, six orders of
           magnitude smaller.  The monocrystal (open triangle) has the same
           anisotropy and no grain boundaries, so it barely codas: the
           scattering needs the *disorder*, not just the anisotropy.  The
           small points are three other textures with the same statistics and
           the same A, a factor two apart -- which is why the surrogate has to
           map the orientation FIELD and not the scalar A.

    python3 make_aniso_coda.py
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
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_aniso_coda.png"

INK, MUTED, ACCENT, MARK = "#1A1A19", "#6E6E68", "#D55E00", "#0072B2"
GREEN = "#009E73"
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]

OFF_SHOW = 3.0e-3                # the A-scan drawn in panel (c)
W_DIRECT = (-0.35e-6, 0.45e-6)   # window around the Rayleigh arrival
W_CODA = 0.90e-6                 # the coda starts this long after it
CODA_END = 3.70e-6               # ... and stops before the bottom reflection
                                 # comes back (2H/vP = 4.8 us, and the sponge
                                 # only thins it): past 3.75 us the isotropic
                                 # run is no longer quiet, so nothing is
                                 # measured or shown there
T_RISE = 0.25e-6                 # source rise, so t_R(0) is not 0

d = np.load(SRC)
t, off = d["t"], d["rec_offsets"]
poly_A = d["poly_A"]
A_max = float(poly_A.max())
v_r = 0.92 * float(d["vs_iso"])          # Rayleigh speed of the Hill medium
T_SHOW = (0.0, CODA_END * 1e6 + 0.1)


def t_rayleigh(offset):
    return abs(offset) / v_r + T_RISE


def windows(i):
    tr = t_rayleigh(off[i])
    return ((t > tr + W_DIRECT[0]) & (t < tr + W_DIRECT[1]),
            (t > tr + W_CODA) & (t < CODA_END))


def energies(b):
    """(direct, coda) energy summed over the receiver line."""
    e_d = e_c = 0.0
    for i in range(off.size):
        wd, wc = windows(i)
        e_d += float(np.sum(b[i][wd] ** 2))
        e_c += float(np.sum(b[i][wc] ** 2))
    return e_d, e_c


def direct_peak_ratio(b, ref):
    """Median peak ratio and arrival-time shift in the direct window."""
    r, dt_ = [], []
    for i in range(off.size):
        wd, _ = windows(i)
        r.append(np.abs(b[i][wd]).max() / np.abs(ref[i][wd]).max())
        dt_.append(t[wd][np.argmax(np.abs(b[i][wd]))]
                   - t[wd][np.argmax(np.abs(ref[i][wd]))])
    return float(np.median(r)), float(np.median(dt_))


# drawn at the size it is used (118 mm on the slide): the point sizes below
# are the slide's point sizes, so do not rescale it there
fig = plt.figure(figsize=(4.87, 1.62), dpi=400)
fig.patch.set_facecolor("white")
axA = fig.add_axes([0.043, 0.175, 0.132, 0.570])
axB = fig.add_axes([0.207, 0.175, 0.132, 0.570])
axC = fig.add_axes([0.428, 0.175, 0.258, 0.570])
axD = fig.add_axes([0.805, 0.175, 0.180, 0.570])

# ---- (a), (b) the B-scan, same colour scale -----------------------------
b_iso, b_poly = d["traces_iso"], d[f"traces_poly_{A_max:g}"]
vmax = float(np.percentile(np.abs(b_poly), 99.5))
for ax, b, lab in ((axA, b_iso, r"(a) isotropic"),
                   (axB, b_poly, rf"(b) $A={A_max:g}$")):
    ax.imshow(b.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              interpolation="bilinear",
              extent=[off[0] * 1e3, off[-1] * 1e3, t[-1] * 1e6, t[0] * 1e6])
    ax.set_ylim(T_SHOW[1], T_SHOW[0])
    ax.set_xticks([-5, 0, 5])
    ax.set_xlabel(r"offset [mm]", fontsize=5.4, color=INK, labelpad=0.6)
    ax.set_title(lab, fontsize=5.6, color=INK, pad=2.0)
axA.set_ylabel(r"$t$ [$\mu$s]", fontsize=5.4, color=INK, labelpad=1.0)
axA.set_yticks([0, 2, 4])
axB.set_yticks([0, 2, 4])
axB.set_yticklabels([])
axB.text(0.5, 0.56, "the grains\nwrite here", fontsize=4.6, color=INK,
         ha="center", va="center", transform=axB.transAxes,
         path_effects=HALO)

# ---- (c) one A-scan: same time, less amplitude, endless tail ------------
i_rec = int(np.argmin(np.abs(off - OFF_SHOW)))
tr = t_rayleigh(off[i_rec])
scale = np.abs(b_iso[i_rec]).max()
ratio, shift = direct_peak_ratio(b_poly, b_iso)
axC.axvspan((tr + W_DIRECT[0]) * 1e6, (tr + W_DIRECT[1]) * 1e6,
            color=GREEN, alpha=0.10, lw=0)
axC.axvspan((tr + W_CODA) * 1e6, T_SHOW[1], color=MARK, alpha=0.07, lw=0)
axC.plot(t * 1e6, b_iso[i_rec] / scale, lw=0.5, color=MUTED, label=r"$A=1$")
axC.plot(t * 1e6, b_poly[i_rec] / scale, lw=0.5, color=MARK,
         label=rf"$A={A_max:g}$")
# the coda carries 1% of the direct energy: it has to be drawn at gain, and
# the gain has to be written on the figure
GAIN = 10.0
m_c = (t > tr + W_CODA) & (t < CODA_END)
axC.plot(t[m_c] * 1e6, GAIN * b_iso[i_rec][m_c] / scale, lw=0.45, color=MUTED)
axC.plot(t[m_c] * 1e6, GAIN * b_poly[i_rec][m_c] / scale, lw=0.45, color=MARK)
axC.text(CODA_END * 1e6 - 0.05, 0.62, rf"$\times{GAIN:.0f}$", fontsize=4.6,
         color=MARK, ha="right", va="bottom", path_effects=HALO)
axC.set_xlim(*T_SHOW)
axC.set_ylim(-1.25, 1.55)
axC.set_yticks([])
axC.set_xlabel(r"$t$ [$\mu$s]", fontsize=5.4, color=INK, labelpad=0.6)
axC.set_title(rf"(c) one receiver, $+{OFF_SHOW * 1e3:.0f}$ mm", fontsize=5.6,
              color=INK, pad=2.0)
axC.text(tr * 1e6, 1.42,
         rf"same arrival ($+{shift * 1e9:.0f}$\,ns), $\div{1 / ratio:.1f}$",
         fontsize=4.6, color=GREEN, ha="center", va="top",
         path_effects=HALO)
axC.text((tr + W_CODA + 0.15e-6) * 1e6, -1.15, "coda", fontsize=4.8,
         color=MARK, ha="left", va="bottom", path_effects=HALO)
axC.legend(fontsize=4.6, frameon=False, loc="upper right", handlelength=0.9,
           labelspacing=0.08, borderpad=0.05, borderaxespad=0.05)

# ---- (d) the coda needs disorder, and it is a fingerprint -------------
def coda_vector(b):
    return np.concatenate([b[i][windows(i)[1]] for i in range(off.size)])


def corr(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


e_d_iso, e_c_iso = energies(b_iso)
E = {a: energies(d[f"traces_poly_{a:g}"]) for a in poly_A}
e_mono = energies(d["traces_mono"])
seeds = [d[f"traces_seed_{s}"] for s in d["extra_seeds"]]
e_seeds = [energies(b)[1] / e_d_iso for b in seeds]

coda = np.array([E[a][1] for a in poly_A]) / e_d_iso
axD.semilogy(poly_A, coda, "o-", ms=2.6, lw=1.0, color=MARK)
axD.semilogy([1.0], [e_c_iso / e_d_iso], "s", ms=2.6, color=MUTED,
             mfc="white", mew=0.7)
axD.semilogy([A_max], [e_mono[1] / e_d_iso], "^", ms=3.2, color=ACCENT,
             mfc="white", mew=0.8)
axD.semilogy(np.full(len(e_seeds), A_max + 0.11), e_seeds, ".", ms=2.4,
             color=MARK, alpha=0.85)
axD.plot([A_max + 0.11] * 2, [min(e_seeds), max(e_seeds)], lw=0.6,
         color=MARK, alpha=0.6)

# the coda of two textures with the same statistics is the same size and a
# different signal: that is the whole case for learning an operator
cc = [corr(coda_vector(seeds[i]), coda_vector(seeds[j]))
      for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
cd = [corr(coda_vector(b_poly), coda_vector(b)) for b in seeds]

axD.text(A_max + 0.20, max(e_seeds) * 1.5,
         rf"other textures:" "\n" rf"same size, corr\,$\approx${np.mean(cc + cd):.2f}",
         fontsize=4.4, color=MARK, ha="right", va="bottom",
         path_effects=HALO)
axD.text(A_max - 0.07, e_mono[1] / e_d_iso, "one crystal:\nno disorder",
         fontsize=4.4, color=ACCENT, ha="right", va="center",
         path_effects=HALO)
axD.text(1.03, e_c_iso / e_d_iso * 0.75, "isotropic", fontsize=4.4,
         color=MUTED, ha="left", va="top", path_effects=HALO)
axD.set_xlim(0.92, 2.92)
axD.set_ylim(5e-5, 9e-2)
axD.set_xticks([1.0, 1.5, 2.0, 2.5])
axD.set_xlabel(r"Zener ratio $A$", fontsize=5.4, color=INK, labelpad=0.6)
axD.set_ylabel(r"coda\,/\,direct energy", fontsize=5.2, color=INK, labelpad=1.0)
axD.set_title(r"(d) the coda needs disorder", fontsize=5.6, color=INK, pad=2.0)
axD.minorticks_off()

for ax in (axA, axB, axC, axD):
    ax.set_facecolor("white")
    ax.tick_params(labelsize=4.8, colors=MUTED, length=1.8, width=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.5)
for ax in (axA, axB):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.5)
axC.spines["left"].set_visible(False)

fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight",
            pad_inches=0.02)
print("wrote", OUT)
print(f"  direct-window peak at the receivers: x{ratio:.2f} "
      f"(median over {off.size} stations), arrival shift {shift * 1e9:+.0f} ns")
print(f"  coda/direct(A=1) energy: isotropic {e_c_iso / e_d_iso:.2e}, "
      f"monocrystal {e_mono[1] / e_d_iso:.2e}, "
      + ", ".join(f"A={a:g}: {c:.2e}" for a, c in zip(poly_A, coda)))
print(f"  direct energy / isotropic: monocrystal {e_mono[0] / e_d_iso:.3f}, "
      + ", ".join(f"A={a:g}: {E[a][0] / e_d_iso:.3f}" for a in poly_A))
print("  other textures at A=%g: coda " % A_max
      + ", ".join(f"{c:.2e}" for c in e_seeds))
print(f"  coda waveform correlation between textures: {np.mean(cc + cd):+.3f} "
      f"(min {min(cc + cd):+.3f}, max {max(cc + cd):+.3f})")
