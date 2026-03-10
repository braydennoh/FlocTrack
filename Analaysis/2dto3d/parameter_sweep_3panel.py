#!/usr/bin/env python3
"""
3-panel parameter sweep: (a) shear, (b) OM, (c) 2D→3D conversion comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

RES = "/Users/braydennoh/Research/Floc/results"
OUT = f"{RES}/2dto3d"

# ── Shear-velocity sweep (a) ──
shear_x = [0.02, 0.03, 0.04]

agg_shear   = np.array([1.976, 1.940, 1.996])
agg_sh_lo   = np.array([1.883, 1.819, 1.889])
agg_sh_hi   = np.array([2.069, 2.061, 2.103])

str_shear   = np.array([1.766, 1.951, 2.115])
str_sh_lo   = np.array([1.714, 1.917, 2.073])
str_sh_hi   = np.array([1.958, 1.995, 2.166])

# ── Organic-matter sweep (b) ──
om_x = [1, 2, 3]

agg_om      = np.array([1.797, 1.738, 1.746])
agg_om_lo   = np.array([1.560, 1.558, 1.617])
agg_om_hi   = np.array([2.039, 1.917, 1.876])

str_om      = np.array([1.840, 1.658, 1.627])
str_om_lo   = np.array([1.765, 1.552, 1.533])
str_om_hi   = np.array([1.909, 1.759, 1.735])

# ── colours ──
c_strat = "#FF6C0C"
c_agg   = "#2E4057"

# ── load per-bin scatter data from all 6 datasets ──
FD_BINS = np.arange(1.0, 2.1, 0.1)
bin_mids = 0.5 * (FD_BINS[:-1] + FD_BINS[1:])

datasets = [
    ("LOWSHEAR", "lowshear", {7}),   ("MIDSHEAR", "midshear", {0}),
    ("HIGHSHEAR", "highshear", set()), ("OM1", "om1", set()),
    ("OM2", "om2", {0, 1}),           ("OM3", "om3", {6}),
]

all_nf2d, all_nf3d_med, all_nf3d_lo, all_nf3d_hi = [], [], [], []
for name, sub, excl in datasets:
    d = np.load(f"{RES}/{sub}/{name}_bin_posteriors.npz", allow_pickle=True)
    for k in sorted(int(k) for k in d.keys()):
        if k in excl:
            continue
        slopes = d[str(k)][:, 0]
        nf3d = slopes + 1
        all_nf2d.append(bin_mids[k])
        all_nf3d_med.append(np.median(nf3d))
        all_nf3d_lo.append(np.percentile(nf3d, 2.5))
        all_nf3d_hi.append(np.percentile(nf3d, 97.5))

all_nf2d    = np.array(all_nf2d)
all_nf3d_med = np.array(all_nf3d_med)
all_nf3d_lo  = np.array(all_nf3d_lo)
all_nf3d_hi  = np.array(all_nf3d_hi)

# ── figure ──
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.5, 1.9))
fig.subplots_adjust(wspace=0.45)

# ── helper for panels (a) and (b) ──
def plot_panel(ax, x, agg, agg_lo, agg_hi, strt, strt_lo, strt_hi, xlabel):
    agg_yerr = np.array([agg - agg_lo, agg_hi - agg])
    str_yerr = np.array([strt - strt_lo, strt_hi - strt])

    ax.plot(x, strt, '-o', color=c_strat, ms=7, mec=c_strat,
            mew=1.0, mfc=(*plt.matplotlib.colors.to_rgb(c_strat), 0.8),
            lw=1.8, label="Stratified", zorder=10)
    ax.plot(x, agg, '-o', color=c_agg, ms=7, mec=c_agg,
            mew=1.0, mfc=(*plt.matplotlib.colors.to_rgb(c_agg), 0.8),
            lw=1.8, label="Aggregated", zorder=10)

    ax.errorbar(x, strt, yerr=str_yerr, fmt='none',
                ecolor=c_strat, capsize=3, capthick=1.2, lw=1.2, zorder=9)
    ax.errorbar(x, agg,  yerr=agg_yerr, fmt='none',
                ecolor=c_agg,   capsize=3, capthick=1.2, lw=1.2, zorder=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"3D Fractal Dimension $n_f$")
    ax.set_ylim(1.4, 2.2)
    ax.set_aspect("auto")
    ax.legend(fontsize=6, frameon=False, loc="upper left")

# Panel (a): shear
plot_panel(ax_a, shear_x, agg_shear, agg_sh_lo, agg_sh_hi,
           str_shear, str_sh_lo, str_sh_hi, "Shear Velocity (m/s)")
ax_a.set_xticks(shear_x)

# best-fit equations
m_sh_s, b_sh_s = np.polyfit(shear_x, str_shear, 1)
m_sh_a, b_sh_a = np.polyfit(shear_x, agg_shear, 1)
ax_a.text(0.03, 0.12,
          f"$n_f$ = {m_sh_s:.1f}$\\,u_*$ + {b_sh_s:.2f}",
          transform=ax_a.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_strat)
ax_a.text(0.03, 0.04,
          f"$n_f$ = {m_sh_a:.1f}$\\,u_*$ + {b_sh_a:.2f}",
          transform=ax_a.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_agg)

# Panel (b): OM
plot_panel(ax_b, om_x, agg_om, agg_om_lo, agg_om_hi,
           str_om, str_om_lo, str_om_hi, "Organic Matter (%)")
ax_b.set_xticks(om_x)

m_om_s, b_om_s = np.polyfit(om_x, str_om, 1)
m_om_a, b_om_a = np.polyfit(om_x, agg_om, 1)
ax_b.text(0.03, 0.12,
          f"$n_f$ = {m_om_s:.2f}$\\,$OM + {b_om_s:.2f}",
          transform=ax_b.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_strat)
ax_b.text(0.03, 0.04,
          f"$n_f$ = {m_om_a:.2f}$\\,$OM + {b_om_a:.2f}",
          transform=ax_b.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_agg)

# ── Panel (c): 2D → 3D conversions ──
x2d = np.linspace(0.01, 2.0, 200)

# theoretical / empirical models
ax_c.plot(x2d, x2d + 1, 'k--', lw=1.5, label=r"$n_f = n_f^{(\mathrm{2D})} + 1$")
ax_c.plot(x2d, 0.2015 * x2d**4.079, color="#FF6C0C", ls="--", lw=1.3,
          label="Wang et al. (2022) PL")
ax_c.plot(x2d, 0.8118 * x2d**1.8054, color="#FF6C0C", ls="-", lw=1.3,
          label="Wang et al. (2022) BC")

# scatter data from all 6 datasets
ax_c.plot(all_nf2d, all_nf3d_med, 'o', ms=4, color=c_agg, mec=c_agg,
          mew=0.5, alpha=0.8, zorder=10, label="This study")

ax_c.set_xlabel(r"$n_f^{(\mathrm{2D})}$")
ax_c.set_ylabel(r"3D Fractal Dimension $n_f$")
ax_c.set_xlim(0, 2.0)
ax_c.set_ylim(0, 3.0)
ax_c.legend(fontsize=5.5, frameon=False, loc="upper left")

# ── save ──
fig.savefig(f"{OUT}/parameter_sweep_3panel.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/parameter_sweep_3panel.svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/parameter_sweep_3panel.png")
print(f"Saved → {OUT}/parameter_sweep_3panel.svg")
