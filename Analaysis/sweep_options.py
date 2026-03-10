#!/usr/bin/env python3
"""Generate 4 style options for the parameter sweep comparison."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# ── data ──
shear_x = np.array([0.02, 0.03, 0.04])
agg_shear   = np.array([1.976, 1.940, 1.996])
agg_sh_lo   = np.array([1.883, 1.819, 1.889])
agg_sh_hi   = np.array([2.069, 2.061, 2.103])
str_shear   = np.array([1.766, 1.951, 2.115])
str_sh_lo   = np.array([1.714, 1.917, 2.073])
str_sh_hi   = np.array([1.958, 1.995, 2.166])

om_x = np.array([1, 2, 3])
agg_om    = np.array([1.797, 1.738, 1.746])
agg_om_lo = np.array([1.560, 1.558, 1.617])
agg_om_hi = np.array([2.039, 1.917, 1.876])
str_om    = np.array([1.840, 1.658, 1.627])
str_om_lo = np.array([1.765, 1.552, 1.533])
str_om_hi = np.array([1.909, 1.759, 1.735])

c_strat = "#7BC8A4"
c_agg   = "#2E4057"
OUT = "/Users/braydennoh/Research/Floc/results"


# ═══════════════════════════════════════════════════════════════
# OPTION A — Grouped bar chart
# ═══════════════════════════════════════════════════════════════
def option_a():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                    constrained_layout=True)
    w = 0.003  # bar half-width for shear
    for ax, x, agg, alo, ahi, st, slo, shi, xl, xw, lab in [
        (ax1, shear_x, agg_shear, agg_sh_lo, agg_sh_hi,
         str_shear, str_sh_lo, str_sh_hi, "Shear Velocity (m/s)", w, "(a)"),
        (ax2, om_x, agg_om, agg_om_lo, agg_om_hi,
         str_om, str_om_lo, str_om_hi, "Organic Matter (%)", 0.15, "(b)")]:

        ax.bar(x - xw, st, 2*xw, color=c_strat, alpha=0.85,
               label="Stratified", zorder=5)
        ax.bar(x + xw, agg, 2*xw, color=c_agg, alpha=0.85,
               label="Aggregated", zorder=5)
        ax.errorbar(x - xw, st,
                    yerr=[st - slo, shi - st],
                    fmt='none', ecolor='k', capsize=3, lw=1, zorder=10)
        ax.errorbar(x + xw, agg,
                    yerr=[agg - alo, ahi - agg],
                    fmt='none', ecolor='k', capsize=3, lw=1, zorder=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"3D Fractal Dimension $n_f^{(3D)}$")
        ax.set_ylim(1.3, 2.3)
        ax.set_xticks(x)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_A_bars.png", dpi=300)
    plt.close(fig)
    print("  A saved")


# ═══════════════════════════════════════════════════════════════
# OPTION B — Dot-and-whisker (dumbbell)
# ═══════════════════════════════════════════════════════════════
def option_b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                    constrained_layout=True)
    for ax, x, agg, alo, ahi, st, slo, shi, xl, lab in [
        (ax1, shear_x, agg_shear, agg_sh_lo, agg_sh_hi,
         str_shear, str_sh_lo, str_sh_hi, "Shear Velocity (m/s)", "(a)"),
        (ax2, om_x, agg_om, agg_om_lo, agg_om_hi,
         str_om, str_om_lo, str_om_hi, "Organic Matter (%)", "(b)")]:

        # connecting segments
        for xi, si, ai in zip(x, st, agg):
            ax.plot([xi, xi], [si, ai], color='grey', lw=1.5,
                    ls='-', zorder=1)

        ax.errorbar(x, st, yerr=[st - slo, shi - st],
                    fmt='o', ms=8, color=c_strat, mec='white', mew=1.2,
                    ecolor=c_strat, capsize=4, capthick=1.2, lw=1.2,
                    label="Stratified", zorder=10)
        ax.errorbar(x, agg, yerr=[agg - alo, ahi - agg],
                    fmt='s', ms=7, color=c_agg, mec='white', mew=1.2,
                    ecolor=c_agg, capsize=4, capthick=1.2, lw=1.2,
                    label="Aggregated", zorder=10)

        ax.set_xlabel(xl)
        ax.set_ylabel(r"3D Fractal Dimension $n_f^{(3D)}$")
        ax.set_ylim(1.3, 2.3)
        ax.set_xticks(x)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_B_dumbbell.png", dpi=300)
    plt.close(fig)
    print("  B saved")


# ═══════════════════════════════════════════════════════════════
# OPTION C — Divergence plot (Agg − Strat)
# ═══════════════════════════════════════════════════════════════
def option_c():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                    constrained_layout=True)
    for ax, x, agg, alo, ahi, st, slo, shi, xl, lab in [
        (ax1, shear_x, agg_shear, agg_sh_lo, agg_sh_hi,
         str_shear, str_sh_lo, str_sh_hi, "Shear Velocity (m/s)", "(a)"),
        (ax2, om_x, agg_om, agg_om_lo, agg_om_hi,
         str_om, str_om_lo, str_om_hi, "Organic Matter (%)", "(b)")]:

        diff = agg - st
        # propagate uncertainty (conservative: add in quadrature)
        diff_lo = np.sqrt((agg - alo)**2 + (st - slo)**2)
        diff_hi = np.sqrt((ahi - agg)**2 + (shi - st)**2)

        ax.axhline(0, color='grey', ls='--', lw=0.8, zorder=0)
        ax.axhspan(-0.05, 0.05, color='grey', alpha=0.1, linewidth=0)
        ax.errorbar(x, diff, yerr=[diff_lo, diff_hi],
                    fmt='o', ms=8, color=c_agg, mec='white', mew=1.2,
                    ecolor=c_agg, capsize=5, capthick=1.5, lw=1.5,
                    zorder=10)
        for xi, di in zip(x, diff):
            ax.plot([xi, xi], [0, di], color=c_agg, lw=2, zorder=5)

        ax.set_xlabel(xl)
        ax.set_ylabel(r"$\Delta n_f^{(3D)}$  (Aggregated $-$ Stratified)")
        ax.set_xticks(x)
        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_C_divergence.png", dpi=300)
    plt.close(fig)
    print("  C saved")


# ═══════════════════════════════════════════════════════════════
# OPTION D — Slope chart (paired comparison)
# ═══════════════════════════════════════════════════════════════
def option_d():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                    constrained_layout=True)
    labels_sh = ["0.02", "0.03", "0.04"]
    labels_om = ["1%", "2%", "3%"]

    for ax, st, slo, shi, agg, alo, ahi, labels, xl, lab in [
        (ax1, str_shear, str_sh_lo, str_sh_hi,
         agg_shear, agg_sh_lo, agg_sh_hi, labels_sh,
         "Shear Velocity (m/s)", "(a)"),
        (ax2, str_om, str_om_lo, str_om_hi,
         agg_om, agg_om_lo, agg_om_hi, labels_om,
         "Organic Matter (%)", "(b)")]:

        x_left, x_right = 0, 1
        for i in range(len(st)):
            col = plt.cm.Set2(i / 3)
            ax.plot([x_left, x_right], [st[i], agg[i]],
                    '-o', color=col, lw=1.8, ms=7, mec='white', mew=1.2,
                    zorder=10, label=labels[i])
            # error bars
            ax.errorbar(x_left, st[i], yerr=[[st[i]-slo[i]], [shi[i]-st[i]]],
                        fmt='none', ecolor=col, capsize=3, lw=1, zorder=9)
            ax.errorbar(x_right, agg[i], yerr=[[agg[i]-alo[i]], [ahi[i]-agg[i]]],
                        fmt='none', ecolor=col, capsize=3, lw=1, zorder=9)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["Stratified", "Aggregated"])
        ax.set_ylabel(r"3D Fractal Dimension $n_f^{(3D)}$")
        ax.set_ylim(1.3, 2.3)
        ax.legend(fontsize=7, frameon=False, title=xl, title_fontsize=7)
        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_D_slope.png", dpi=300)
    plt.close(fig)
    print("  D saved")


option_a()
option_b()
option_c()
option_d()
print("All 4 options saved.")
