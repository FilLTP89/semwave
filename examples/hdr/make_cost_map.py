#!/usr/bin/env python3
"""Iso-demand map of the SEM3D cost law in the (f_max, Vs_min) plane.

    cost  ~  f_max^4 * L^3 * T / (Vs_min)^3

Dividing out L^3 T makes runs of very different size and duration
comparable, so every test case reduces to one point (f_max, Vs_min) and the
contours are the *worst-case* demand per unit of space-time, normalised to
the Le Teil regional run.

Also overlays the GPU virtual improvement (x3.34 OpenACC A100 speedup):
  - Shifted iso-cost contours in purple showing the GPU constant shift
    (+log10(3.34) = +0.52 decades).
  - Virtual GPU-improved markers for the same 6 test cases at equal budget
    (f_max -> 3.34^(1/4) f_max = 1.35 f_max, or Vs_min -> 3.34^(-1/3) Vs_min).
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "figures" / "hdr" / "hdr_cost_map.png"

MARK = "#0072B2"      # CPU test-case marker (Okabe-Ito blue)
CPU_LINE = "#2C5282"  # CPU iso-cost contour (slate navy)
GPU_COL = "#7C2A90"   # GPU iso-cost contour & virtual reach (HDR theme purple thE)
ACCENT = "#B5121B"    # Target zone & L-path (CentraleSupelec red)
SOIL = "#008060"      # Measured Vs_min effect (deep teal-green)
INK, MUTED = "#1A1A19", "#555550"

SPEEDUP_GPU = 3.34    # Jean Zay 8xA100 vs 512 MPI CPU ranks

# name, machine, f_max [Hz], Vs_min [m/s], (Lx,Ly,Lz) [km], T [s], cost [core-h]
CASES = [
    dict(name="Le Teil",              short="Le Teil",      machine="TGCC",
         f=5.0,  vs=2180, L=(80, 92, 79), T=60.0,  cost=61440.0,
         src="Lehmann et al. 2022; 18.3 M elements, Vs_min 2180 m/s"),
    dict(name="HEMEW$^\\mathregular{S}$-3D", short="HEMEW",  machine="Jean Zay",
         f=5.0,  vs=1071, L=(9.6, 9.6, 9.6), T=6.4, cost=22.72,
         src="JCP2024 tab. stats_materials (Vs clipped 1071-4500); 22.72 core-h"),
    dict(name="Niigata KKNPP",        short="Niigata",      machine="Fusion",
         f=7.0,  vs=700,  L=(60, 60, 60), T=None,  cost=None,
         src="DARI A4/A6 resume: 60 km cube, 200 m grid, Vs_min 700 m/s"),
    dict(name="Argostoli O5",         short="Argostoli",    machine="Occigen",
         f=10.0, vs=600,  L=(44, 44, 63), T=20.0,  cost=81100.0,
         src="DARI A7; MDPI2022 tab. basin_argo_geo (Vs 600 m/s at z=0)"),
    dict(name="Cadarache",            short="Cadarache",    machine="Fusion",
         f=10.5, vs=399,  L=(50, 60, 60), T=30.0,  cost=960.0,
         src="GJI2022: lambda_min 38 m at 10.5 Hz -> Vs_min 399 m/s; 960 core-h"),
    dict(name="Volvi",                short="Volvi",        machine="Ruche",
         f=5.0,  vs=450,  L=(13, 11, None), T=None,
         cost=1512.0,
         src="216 MPI cores x 7 h on Ruche = 62 CPU-days; 13x11 km, 0-5 Hz;"
             " Vs_min 450 m/s in the SEM3D mesh"),
]

def demand(f, vs_ms):
    """Worst-case cost per unit space-time, f^4 / Vs^3 with Vs in km/s."""
    return f ** 4 / (vs_ms / 1000.0) ** 3

REF = demand(CASES[0]["f"], CASES[0]["vs"])          # Le Teil = 1

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(4.35, 3.12), dpi=400)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

FLO, FHI, VLO, VHI = 2.0, 40.0, 110.0, 3200.0
ff = np.logspace(np.log10(FLO), np.log10(FHI), 420)
vv = np.logspace(np.log10(VLO), np.log10(VHI), 420)
F, V = np.meshgrid(ff, vv)
R = np.log10(demand(F, V) / REF)

levels = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
# Soft cool-to-warm pastel background (light blue -> pale yellow -> soft coral)
ax.contourf(F, V, R, levels=levels, cmap="RdYlBu_r", alpha=0.34,
            vmin=-2.2, vmax=7.8, zorder=0)

# 1) Baseline CPU iso-cost contours (slate navy solid)
ax.contour(F, V, R, levels=levels, colors=CPU_LINE,
           linewidths=0.75, alpha=0.55, zorder=1)
cs = ax.contour(F, V, R, levels=[0, 2, 4, 6], colors=CPU_LINE,
                linewidths=1.0, alpha=0.85, zorder=1)

# 2) GPU-shifted iso-cost contours (x3.34 constant shift -> +0.524 decades, purple dashed)
gpu_shift = np.log10(SPEEDUP_GPU)
cs_gpu = ax.contour(F, V, R - gpu_shift, levels=levels, colors=GPU_COL,
                    linestyles=[(0, (3.2, 2.0))], linewidths=0.95, alpha=0.82, zorder=2)

# Contour labels on CPU lines (positioned clear of legend & markers)
MANUAL = [(2.25, 760), (3.6, 305), (31.0, 1150), (35.0, 292)]
_cl = ax.clabel(cs, inline=True, fontsize=6.8, fmt=lambda x:
                (r"$\times 1$" if abs(x) < 1e-9 else rf"$\times10^{{{int(x)}}}$"),
                colors=CPU_LINE, inline_spacing=4, manual=MANUAL)
for _t in _cl:
    _t.set_fontweight("bold")
    _t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

# --- engineering demand zone -----------------------------------------
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
ax.add_patch(Rectangle((20, 150), 10, 150, transform=ax.transData,
                       facecolor="none", edgecolor=ACCENT,
                       linestyle=(0, (3.5, 2)), linewidth=1.7, zorder=4))
ax.text(25.0, 143, "structural demand on soft soil", fontsize=7.0,
        color=ACCENT, ha="center", va="top", fontweight="bold",
        zorder=7, path_effects=HALO)

# --- the two effects, decomposed as an L-path ------------------------
f0, v0 = CASES[0]["f"], CASES[0]["vs"]
f_factor = SPEEDUP_GPU ** (1.0 / 4.0)   # 1.352x in f_max at fixed Vs_min
v_factor = SPEEDUP_GPU ** (-1.0 / 3.0)  # 0.669x in Vs_min at fixed f_max

ax.annotate("", xy=(25.0, v0), xytext=(f0 * f_factor, v0), zorder=5,
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.6,
                            shrinkA=4, shrinkB=0))
ax.annotate("", xy=(25.0, 312), xytext=(25.0, v0), zorder=5,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6,
                            shrinkA=0, shrinkB=0,
                            mutation_scale=14))
ax.text(17.5, v0 * 1.07, r"$5\to30$ Hz: $\times\,1296$", fontsize=7.3,
        color=ACCENT, fontweight="bold", ha="center", va="bottom", zorder=7,
        path_effects=HALO)
ax.text(24.0, 1280, "2180 $\\to$ 200 m/s\n" r"$\times\,1295$", fontsize=7.3,
        color=ACCENT, fontweight="bold", ha="right", va="center", zorder=7,
        path_effects=HALO)

# --- measured Vs effect: Le Teil, HEMEW and Volvi all sit at 5 Hz ------
ax.plot([5.0, 5.0], [CASES[0]["vs"], 450], color=SOIL, lw=1.4, zorder=2,
        linestyle=(0, (1.6, 1.6)))
ax.text(5.15, 330, r"$V_s^{\min}$ alone: $\times114$", fontsize=7.0,
        color=SOIL, fontweight="bold", ha="left", va="top", zorder=7,
        path_effects=HALO)

# --- test cases (CPU markers + virtual GPU-improved points) -----------
LBL = {                       # offsets in points from the CPU marker
    "Le Teil":   (-10,  5, "right"),
    "HEMEW":     (-10,  0, "right"),
    "Volvi":     (-10,  2, "right"),
    "Niigata":   ( 14, 15, "left"),
    "Argostoli": ( 27,  4, "left"),
    "Cadarache": ( 27, -6, "left"),
}
for c in CASES:
    dx, dy, ha = LBL[c["short"]]
    y = c["vs"]
    f_cpu = c["f"]
    f_gpu = f_cpu * f_factor

    # Virtual GPU reach: horizontal arrow in f_max (x1.35) to shifted GPU contour
    ax.annotate("", xy=(f_gpu, y), xytext=(f_cpu, y), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=GPU_COL, lw=1.4,
                                shrinkA=3.5, shrinkB=2.5, mutation_scale=9))
    ax.plot([f_gpu], [y], marker="D", ms=4.8, mfc=GPU_COL,
            mec="white", mew=0.9, zorder=6)

    # Also show the equivalent Vs_min drop (x0.67) on Le Teil (the x1 reference contour)
    if c["short"] == "Le Teil":
        y_gpu = y * v_factor
        ax.annotate("", xy=(f_cpu, y_gpu), xytext=(f_cpu, y), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=GPU_COL, lw=1.2,
                                    ls=":", shrinkA=3.5, shrinkB=2.5, mutation_scale=8))
        ax.plot([f_cpu], [y_gpu], marker="D", ms=4.4, mfc="white",
                mec=GPU_COL, mew=1.2, zorder=6)

    # CPU marker
    ax.plot([f_cpu], [y], marker="o", ms=7.2, mfc=MARK,
            mec="white", mew=1.5, zorder=6)
    sub = c["machine"]
    ax.annotate(c["name"], xy=(f_cpu, y), xytext=(dx, dy + 3.6),
                textcoords="offset points", fontsize=7.4, color=INK,
                fontweight="bold", ha=ha, va="center", zorder=7,
                path_effects=HALO)
    ax.annotate(sub, xy=(f_cpu, y), xytext=(dx, dy - 4.6),
                textcoords="offset points", fontsize=6.2, color=MUTED,
                ha=ha, va="center", zorder=7,
                path_effects=HALO)

# Highlight annotation on Le Teil virtual shift
ax.text(5.88, 2400,
        r"$+35\%$ $f_{\max}$",
        fontsize=6.4, color=GPU_COL, fontweight="bold", ha="center", va="bottom",
        zorder=7, path_effects=HALO)
ax.text(5.25, 1460,
        r"$-33\%$ $V_s^{\min}$",
        fontsize=6.3, color=GPU_COL, fontweight="bold", ha="left", va="center",
        zorder=7, path_effects=HALO)

# --- compact legend inside lower-left ---------------------------------
legend_handles = [
    Line2D([0], [0], color=CPU_LINE, lw=1.3, label="CPU iso-cost"),
    Line2D([0], [0], color=GPU_COL, lw=1.2, ls=(0, (3.2, 2.0)),
           label=r"GPU $\times3.34$ iso-cost ($+0.52$ dec)"),
    Line2D([0], [0], marker="o", color="none", mfc=MARK, mec="white",
           ms=6.0, label="CPU test case"),
    Line2D([0], [0], marker="D", color=GPU_COL, mfc=GPU_COL, mec="white",
           lw=1.1, ms=4.5, label=r"virtual GPU reach ($\times1.35\,f_{\max}$)"),
]
leg = ax.legend(handles=legend_handles, loc="lower left", frameon=True,
                facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
                fontsize=6.0, handlelength=1.8, borderpad=0.35, labelspacing=0.22)
leg.set_zorder(8)

# --- axes -------------------------------------------------------------
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(FLO, FHI); ax.set_ylim(VLO, VHI)
ax.set_xticks([2, 3, 5, 7, 10, 15, 20, 30, 40])
ax.set_yticks([150, 200, 300, 500, 1000, 2000, 3000])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.minorticks_off()
ax.tick_params(labelsize=7.2, colors=MUTED, length=3, width=0.7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.7)
ax.set_xlabel(r"$f_{\max}$  [Hz]", fontsize=8.0, color=INK, labelpad=2)
ax.set_ylabel(r"$V_s^{\min}$  [m/s]", fontsize=8.0, color=INK, labelpad=2)
ax.text(0.0, 1.025,
        r"worst-case demand $f_{\max}^{4}/(V_s^{\min})^{3}$ per unit $L^{3}T$"
        r"  —  Le Teil $=\times1$",
        transform=ax.transAxes, fontsize=7.4, color=INK, va="bottom")

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=340, facecolor="white", bbox_inches="tight",
            pad_inches=0.03)
print("\nwrote", OUT)
