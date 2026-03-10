#!/usr/bin/env python3
"""
Parameter-sweep comparison: 3D fractal dimension (Aggregated vs Stratified)
with uncertainty from MCMC analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# ── Shear-velocity sweep (a) ──
shear_x = [0.02, 0.03, 0.04]  # m/s

# Global / "Aggregated"  — median [95% CI lo, hi]
#   LOWSHEAR: unchanged,  MIDSHEAR: unchanged,  HIGHSHEAR: unchanged
agg_shear   = np.array([1.976, 1.940, 1.996])
agg_sh_lo   = np.array([1.883, 1.819, 1.889])
agg_sh_hi   = np.array([2.069, 2.061, 2.103])

# Stratified — mean [25th, 75th percentile]
#   LOWSHEAR: excl bin 7,  MIDSHEAR: excl bin 0,  HIGHSHEAR: no excl
str_shear   = np.array([1.766, 1.951, 2.115])
str_sh_lo   = np.array([1.714, 1.917, 2.073])
str_sh_hi   = np.array([1.958, 1.995, 2.166])

# ── Organic-matter sweep (b) ──
om_x = [1, 2, 3]  # %

#   OM1: no excl,  OM2: excl bins 0,1,  OM3: excl bin 6
agg_om      = np.array([1.797, 1.738, 1.746])
agg_om_lo   = np.array([1.560, 1.558, 1.617])
agg_om_hi   = np.array([2.039, 1.917, 1.876])

str_om      = np.array([1.840, 1.658, 1.627])
str_om_lo   = np.array([1.765, 1.552, 1.533])
str_om_hi   = np.array([1.909, 1.759, 1.735])

# ── colours (match reference) ──
c_strat = "#7BC8A4"   # light green
c_agg   = "#2E4057"   # dark navy

# ── figure ──
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(4.2, 2.0),
                                  constrained_layout=True)

def plot_panel(ax, x, agg, agg_lo, agg_hi, strt, strt_lo, strt_hi,
               xlabel):
    # error bars: [point - lo, hi - point]
    agg_yerr = np.array([agg - agg_lo, agg_hi - agg])
    str_yerr = np.array([strt - strt_lo, strt_hi - strt])

    # uncertainty bands
    ax.fill_between(x, strt_lo, strt_hi, color=c_strat, alpha=0.20,
                    linewidth=0)
    ax.fill_between(x, agg_lo, agg_hi, color=c_agg, alpha=0.20,
                    linewidth=0)

    # lines
    ax.plot(x, strt, '-o', color=c_strat, ms=7, mec='none',
            lw=1.8, label="Stratified", zorder=10)
    ax.plot(x, agg,  '-o', color=c_agg,   ms=7, mec='none',
            lw=1.8, label="Aggregated", zorder=10)

    # error bars
    ax.errorbar(x, strt, yerr=str_yerr, fmt='none',
                ecolor=c_strat, capsize=3, capthick=1.2, lw=1.2, zorder=9)
    ax.errorbar(x, agg,  yerr=agg_yerr, fmt='none',
                ecolor=c_agg,   capsize=3, capthick=1.2, lw=1.2, zorder=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"3D Fractal Dimension $n_f^{(3D)}$")
    ax.set_ylim(1.3, 2.3)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

plot_panel(ax_a, shear_x, agg_shear, agg_sh_lo, agg_sh_hi,
           str_shear, str_sh_lo, str_sh_hi,
           "Shear Velocity (m/s)")
ax_a.set_xticks(shear_x)

plot_panel(ax_b, om_x, agg_om, agg_om_lo, agg_om_hi,
           str_om, str_om_lo, str_om_hi,
           "Organic Matter (%)")
ax_b.set_xticks(om_x)

# ── best-fit slopes ──
# Stratified
m_sh_s = np.polyfit(shear_x, str_shear, 1)[0]
m_om_s = np.polyfit(om_x, str_om, 1)[0]
# Aggregated
m_sh_a = np.polyfit(shear_x, agg_shear, 1)[0]
m_om_a = np.polyfit(om_x, agg_om, 1)[0]

# Intercepts
b_sh_s = np.polyfit(shear_x, str_shear, 1)[1]
b_sh_a = np.polyfit(shear_x, agg_shear, 1)[1]
b_om_s = np.polyfit(om_x, str_om, 1)[1]
b_om_a = np.polyfit(om_x, agg_om, 1)[1]

ax_a.text(0.03, 0.12,
          f"$n_f^{{3D}}$ = {m_sh_s:.1f}$\\,u_*$ + {b_sh_s:.2f}",
          transform=ax_a.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_strat)
ax_a.text(0.03, 0.04,
          f"$n_f^{{3D}}$ = {m_sh_a:.1f}$\\,u_*$ + {b_sh_a:.2f}",
          transform=ax_a.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_agg)
ax_b.text(0.03, 0.12,
          f"$n_f^{{3D}}$ = {m_om_s:.2f}$\\,$OM + {b_om_s:.2f}",
          transform=ax_b.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_strat)
ax_b.text(0.03, 0.04,
          f"$n_f^{{3D}}$ = {m_om_a:.2f}$\\,$OM + {b_om_a:.2f}",
          transform=ax_b.transAxes, fontsize=6, ha="left", va="bottom",
          color=c_agg)

out = "/Users/braydennoh/Research/Floc/results/parameter_sweep.png"
fig.savefig(out, dpi=600)
out2 = "/Users/braydennoh/Research/Floc/results/parameter_sweep.svg"
fig.savefig(out2)
plt.close(fig)
print(f"Saved → {out}")
print(f"Saved → {out2}")
