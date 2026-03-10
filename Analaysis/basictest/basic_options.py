#!/usr/bin/env python3
"""4 visualization options for basic measured values."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

c_diam = "#2E4057"
c_vel  = "#7BC8A4"

# ── load all data ──
def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    df["vel_mm"] = df["Avg_Velocity"] / 1000.0
    return df

all_data = {}
for name, _ in shear_defs + om_defs:
    all_data[name] = load(name)


def get_stats(name):
    df = all_data[name]
    d, v = df["Avg_Concave_Diameter"].values, df["vel_mm"].values
    return {
        "d_med": np.median(d), "d_q25": np.percentile(d, 25),
        "d_q75": np.percentile(d, 75),
        "v_med": np.median(v), "v_q25": np.percentile(v, 25),
        "v_q75": np.percentile(v, 75),
    }


# ═══════════════════════════════════════════════════════════════
# OPTION A — 1×2 dual-axis panels (diameter + velocity together)
# ═══════════════════════════════════════════════════════════════
def option_a():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                    constrained_layout=True)

    for ax, defs, xlabel, lab in [
        (ax1, shear_defs, "Shear Velocity (m/s)", "(a)"),
        (ax2, om_defs,    "Organic Matter (%)",   "(b)")]:

        xs = np.array([x for _, x in defs])
        ss = [get_stats(n) for n, _ in defs]

        d_med = np.array([s["d_med"] for s in ss])
        d_lo  = d_med - np.array([s["d_q25"] for s in ss])
        d_hi  = np.array([s["d_q75"] for s in ss]) - d_med

        ax.errorbar(xs, d_med, yerr=[d_lo, d_hi],
                    fmt='o-', ms=8, color=c_diam, mec='white', mew=1.2,
                    ecolor=c_diam, capsize=5, capthick=1.3, lw=1.8, zorder=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Floc Diameter ($\mu$m)", color=c_diam)
        ax.tick_params(axis='y', colors=c_diam)
        ax.set_xticks(xs)

        ax2r = ax.twinx()
        v_med = np.array([s["v_med"] for s in ss])
        v_lo  = v_med - np.array([s["v_q25"] for s in ss])
        v_hi  = np.array([s["v_q75"] for s in ss]) - v_med

        ax2r.errorbar(xs, v_med, yerr=[v_lo, v_hi],
                      fmt='s-', ms=7, color=c_vel, mec='white', mew=1.2,
                      ecolor=c_vel, capsize=5, capthick=1.3, lw=1.8, zorder=10)
        ax2r.set_ylabel("Settling Velocity (mm/s)", color=c_vel)
        ax2r.tick_params(axis='y', colors=c_vel)

        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    # shared legend
    handles = [Line2D([], [], marker='o', color=c_diam, ms=7, mec='white',
                       mew=1.2, lw=1.5, label="Diameter"),
               Line2D([], [], marker='s', color=c_vel, ms=6, mec='white',
                       mew=1.2, lw=1.5, label="Velocity")]
    fig.legend(handles=handles, fontsize=7, frameon=False,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(f"{OUT}/option_A_dual_axis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  A saved")


# ═══════════════════════════════════════════════════════════════
# OPTION B — Violin plots showing full distributions
# ═══════════════════════════════════════════════════════════════
def option_b():
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8),
                              constrained_layout=True)

    for col, (defs, xlabel) in enumerate([
        (shear_defs, "Shear Velocity (m/s)"),
        (om_defs,    "Organic Matter (%)")]):

        xs = [x for _, x in defs]
        x_pos = np.arange(len(defs))
        labels = [str(x) for x in xs]

        # diameter row
        ax_d = axes[0, col]
        diam_data = [all_data[n]["Avg_Concave_Diameter"].values for n, _ in defs]
        parts = ax_d.violinplot(diam_data, positions=x_pos, showmedians=True,
                                showextrema=False, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor(c_diam)
            pc.set_alpha(0.35)
        parts["cmedians"].set_color(c_diam)
        parts["cmedians"].set_linewidth(2)
        # overlay IQR bars
        for i, d in enumerate(diam_data):
            q25, med, q75 = np.percentile(d, [25, 50, 75])
            ax_d.vlines(x_pos[i], q25, q75, color=c_diam, lw=3, zorder=10)
            ax_d.scatter(x_pos[i], med, color=c_diam, s=30, zorder=11,
                         edgecolors="white", linewidths=1)
        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(labels)
        ax_d.set_xlabel(xlabel)
        ax_d.set_ylabel(r"Floc Diameter ($\mu$m)")

        # velocity row
        ax_v = axes[1, col]
        vel_data = [all_data[n]["vel_mm"].values for n, _ in defs]
        parts = ax_v.violinplot(vel_data, positions=x_pos, showmedians=True,
                                showextrema=False, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor(c_vel)
            pc.set_alpha(0.35)
        parts["cmedians"].set_color(c_vel)
        parts["cmedians"].set_linewidth(2)
        for i, v in enumerate(vel_data):
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            ax_v.vlines(x_pos[i], q25, q75, color=c_vel, lw=3, zorder=10)
            ax_v.scatter(x_pos[i], med, color=c_vel, s=30, zorder=11,
                         edgecolors="white", linewidths=1)
        ax_v.set_xticks(x_pos)
        ax_v.set_xticklabels(labels)
        ax_v.set_xlabel(xlabel)
        ax_v.set_ylabel("Settling Velocity (mm/s)")

    for i, lab in enumerate(["(a)", "(b)", "(c)", "(d)"]):
        axes.flat[i].set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_B_violin.png", dpi=300)
    plt.close(fig)
    print("  B saved")


# ═══════════════════════════════════════════════════════════════
# OPTION C — Box + strip (jittered points) overlay
# ═══════════════════════════════════════════════════════════════
def option_c():
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8),
                              constrained_layout=True)

    for col, (defs, xlabel) in enumerate([
        (shear_defs, "Shear Velocity (m/s)"),
        (om_defs,    "Organic Matter (%)")]):

        xs = [x for _, x in defs]
        x_pos = np.arange(len(defs))
        labels = [str(x) for x in xs]

        # diameter
        ax_d = axes[0, col]
        diam_data = [all_data[n]["Avg_Concave_Diameter"].values for n, _ in defs]
        bp = ax_d.boxplot(diam_data, positions=x_pos, widths=0.4,
                          patch_artist=True, showfliers=False, zorder=5,
                          medianprops=dict(color="white", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(c_diam)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(c_diam)
        # jittered strip
        for i, d in enumerate(diam_data):
            jitter = np.random.default_rng(42).normal(0, 0.06, len(d))
            ax_d.scatter(x_pos[i] + jitter, d, s=1.5, color=c_diam,
                         alpha=0.15, zorder=1)
        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(labels)
        ax_d.set_xlabel(xlabel)
        ax_d.set_ylabel(r"Floc Diameter ($\mu$m)")

        # velocity
        ax_v = axes[1, col]
        vel_data = [all_data[n]["vel_mm"].values for n, _ in defs]
        bp = ax_v.boxplot(vel_data, positions=x_pos, widths=0.4,
                          patch_artist=True, showfliers=False, zorder=5,
                          medianprops=dict(color="white", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(c_vel)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(c_vel)
        for i, v in enumerate(vel_data):
            jitter = np.random.default_rng(43).normal(0, 0.06, len(v))
            ax_v.scatter(x_pos[i] + jitter, v, s=1.5, color=c_vel,
                         alpha=0.15, zorder=1)
        ax_v.set_xticks(x_pos)
        ax_v.set_xticklabels(labels)
        ax_v.set_xlabel(xlabel)
        ax_v.set_ylabel("Settling Velocity (mm/s)")

    for i, lab in enumerate(["(a)", "(b)", "(c)", "(d)"]):
        axes.flat[i].set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_C_boxstrip.png", dpi=300)
    plt.close(fig)
    print("  C saved")


# ═══════════════════════════════════════════════════════════════
# OPTION D — Compact 1×2 dumbbell: diameter ↔ velocity paired
# ═══════════════════════════════════════════════════════════════
def option_d():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 3.0),
                                    constrained_layout=True)

    for ax, defs, xlabel, lab in [
        (ax1, shear_defs, "Shear Velocity (m/s)", "(a)"),
        (ax2, om_defs,    "Organic Matter (%)",   "(b)")]:

        xs = np.array([x for _, x in defs])
        ss = [get_stats(n) for n, _ in defs]

        # normalize both to [0,1] range for visual comparison
        d_med = np.array([s["d_med"] for s in ss])
        v_med = np.array([s["v_med"] for s in ss])

        y_pos = np.arange(len(defs))

        # horizontal layout: conditions on y-axis
        for i in range(len(defs)):
            ax.plot([d_med[i], d_med[i]], [y_pos[i] - 0.15, y_pos[i] + 0.15],
                    color=c_diam, lw=0)  # placeholder
            ax.plot([v_med[i]*60, v_med[i]*60],
                    [y_pos[i] - 0.15, y_pos[i] + 0.15],
                    color=c_vel, lw=0)
            # connecting line
            ax.plot([d_med[i], v_med[i]*60], [y_pos[i], y_pos[i]],
                    color='grey', lw=1.2, ls='-', zorder=1)

        # diameter IQR
        d_lo = d_med - np.array([s["d_q25"] for s in ss])
        d_hi = np.array([s["d_q75"] for s in ss]) - d_med
        ax.errorbar(d_med, y_pos, xerr=[d_lo, d_hi],
                    fmt='o', ms=9, color=c_diam, mec='white', mew=1.2,
                    ecolor=c_diam, capsize=4, capthick=1.3, lw=1.3,
                    label=r"Diameter ($\mu$m)", zorder=10)

        # velocity (×60 to put on comparable scale with diameter)
        v_lo = v_med - np.array([s["v_q25"] for s in ss])
        v_hi = np.array([s["v_q75"] for s in ss]) - v_med
        ax.errorbar(v_med * 60, y_pos, xerr=[v_lo * 60, v_hi * 60],
                    fmt='s', ms=7, color=c_vel, mec='white', mew=1.2,
                    ecolor=c_vel, capsize=4, capthick=1.3, lw=1.3,
                    label="Velocity (mm/s × 60)", zorder=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(x) for x in xs])
        ax.set_ylabel(xlabel)
        ax.set_xlabel(r"$\mu$m  /  mm/s × 60")
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_title(lab, fontsize=11, loc="left", fontweight="bold")

    fig.savefig(f"{OUT}/option_D_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  D saved")


option_a()
option_b()
option_c()
option_d()
print("All 4 options saved.")
