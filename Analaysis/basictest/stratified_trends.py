#!/usr/bin/env python3
"""
Stratified basic measured values: within each fractal-dimension bin,
show how diameter and velocity change with shear / OM.
This isolates the effect of the environmental parameter by holding
n_f approximately constant.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmocean

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

fd_edges = np.arange(1.0, 2.05, 0.1)
fd_mids  = (fd_edges[:-1] + fd_edges[1:]) / 2

# ── load ──
def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    df["vel_mm"] = df["Avg_Velocity"] / 1000.0
    df["fd_bin"] = np.digitize(df["Avg_Perimeter_Fractal_Dimension"],
                                fd_edges) - 1  # 0-indexed
    return df

all_data = {n: load(n) for n, _ in shear_defs + om_defs}

# ── per-bin stats ──
def bin_stats(name, bin_idx):
    """Return median, q25, q75, N for a specific fd bin."""
    df = all_data[name]
    sub = df[df["fd_bin"] == bin_idx]
    n = len(sub)
    if n < 10:
        return None
    d = sub["Avg_Concave_Diameter"].values
    v = sub["vel_mm"].values
    return {
        "n": n,
        "d_med": np.median(d), "d_q25": np.percentile(d, 25),
        "d_q75": np.percentile(d, 75),
        "v_med": np.median(v), "v_q25": np.percentile(v, 25),
        "v_q75": np.percentile(v, 75),
    }

# ── identify bins with enough data across all 3 conditions ──
def usable_bins(defs):
    good = []
    for b in range(10):
        counts = []
        for name, _ in defs:
            s = bin_stats(name, b)
            counts.append(s["n"] if s else 0)
        if all(c >= 10 for c in counts):
            good.append(b)
    return good

shear_bins = usable_bins(shear_defs)
om_bins    = usable_bins(om_defs)

print("Usable bins for shear sweep:", [f"[{fd_edges[b]:.1f},{fd_edges[b+1]:.1f})" for b in shear_bins])
print("Usable bins for OM sweep:   ", [f"[{fd_edges[b]:.1f},{fd_edges[b+1]:.1f})" for b in om_bins])

# Print population table
print("\nPopulation per bin:")
print(f"{'Bin':<12}", end="")
for n, _ in shear_defs + om_defs:
    print(f"{n:>10}", end="")
print()
for b in range(10):
    print(f"[{fd_edges[b]:.1f},{fd_edges[b+1]:.1f})", end="  ")
    for name, _ in shear_defs + om_defs:
        s = bin_stats(name, b)
        print(f"{s['n'] if s else 0:>10}", end="")
    print()

# ── colormap for bins ──
cmap = cmocean.cm.deep_r
# use only usable bins common to both sweeps for consistent coloring
all_usable = sorted(set(shear_bins) | set(om_bins))
bin_colors = {b: cmap((b - min(all_usable)) / max(1, max(all_usable) - min(all_usable)))
              for b in all_usable}

# ═══════════════════════════════════════════════════════════════
# Figure: 2×2 — stratified by fractal dimension
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8), constrained_layout=True)

def plot_stratified(ax, defs, bins, key_med, key_lo, key_hi, ylabel,
                    xlabel, connect=True):
    xs = np.array([x for _, x in defs])
    for b in bins:
        col = bin_colors[b]
        ss = [bin_stats(n, b) for n, _ in defs]
        meds = np.array([s[key_med] for s in ss])
        lo   = meds - np.array([s[key_lo] for s in ss])
        hi   = np.array([s[key_hi] for s in ss]) - meds

        lbl = f"$n_f^{{2D}}$ [{fd_edges[b]:.1f},{fd_edges[b+1]:.1f})"
        style = '-o' if connect else 'o'
        ax.errorbar(xs, meds, yerr=[lo, hi],
                    fmt=style, ms=6, color=col, mec='white', mew=0.8,
                    ecolor=col, capsize=3, capthick=1, lw=1.4,
                    label=lbl, zorder=10 - b, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)

# (a) Diameter vs Shear — stratified
plot_stratified(axes[0, 0], shear_defs, shear_bins,
                "d_med", "d_q25", "d_q75",
                r"Floc Diameter ($\mu$m)", "Shear Velocity (m/s)")
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")

# (b) Diameter vs OM — stratified
plot_stratified(axes[0, 1], om_defs, om_bins,
                "d_med", "d_q25", "d_q75",
                r"Floc Diameter ($\mu$m)", "Organic Matter (%)")
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")

# (c) Velocity vs Shear — stratified
plot_stratified(axes[1, 0], shear_defs, shear_bins,
                "v_med", "v_q25", "v_q75",
                "Settling Velocity (mm/s)", "Shear Velocity (m/s)")
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")

# (d) Velocity vs OM — stratified
plot_stratified(axes[1, 1], om_defs, om_bins,
                "v_med", "v_q25", "v_q75",
                "Settling Velocity (mm/s)", "Organic Matter (%)")
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")

# shared legend at bottom
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6, frameon=False,
           loc="lower center", ncol=len(shear_bins),
           bbox_to_anchor=(0.5, -0.06))

fig.savefig(f"{OUT}/stratified_trends.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/stratified_trends.svg", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}/stratified_trends.png")

# ═══════════════════════════════════════════════════════════════
# Side-by-side: unstratified vs stratified (compact 1×2 for velocity)
# ═══════════════════════════════════════════════════════════════
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8),
                                 constrained_layout=True)

# Unstratified velocity vs shear
c_bulk = "#888888"
for ax, defs, xlabel, lab, bins in [
    (ax1, shear_defs, "Shear Velocity (m/s)", "(a) Shear sweep", shear_bins),
    (ax2, om_defs,    "Organic Matter (%)",   "(b) OM sweep",    om_bins)]:

    xs = np.array([x for _, x in defs])

    # bulk (grey, thick)
    bulk_meds, bulk_lo, bulk_hi = [], [], []
    for name, _ in defs:
        df = all_data[name]
        v = df["vel_mm"].values
        med = np.median(v)
        bulk_meds.append(med)
        bulk_lo.append(med - np.percentile(v, 25))
        bulk_hi.append(np.percentile(v, 75) - med)

    ax.errorbar(xs, bulk_meds, yerr=[bulk_lo, bulk_hi],
                fmt='D-', ms=9, color=c_bulk, mec='white', mew=1.2,
                ecolor=c_bulk, capsize=5, capthick=1.5, lw=2,
                label="Bulk (all $n_f$)", zorder=5, alpha=0.5)

    # stratified lines
    for b in bins:
        col = bin_colors[b]
        ss = [bin_stats(n, b) for n, _ in defs]
        meds = np.array([s["v_med"] for s in ss])
        lo   = meds - np.array([s["v_q25"] for s in ss])
        hi   = np.array([s["v_q75"] for s in ss]) - meds
        lbl = f"$n_f^{{2D}}$ [{fd_edges[b]:.1f},{fd_edges[b+1]:.1f})"
        ax.errorbar(xs, meds, yerr=[lo, hi],
                    fmt='-o', ms=5, color=col, mec='white', mew=0.8,
                    ecolor=col, capsize=3, capthick=1, lw=1.2,
                    label=lbl, zorder=10, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Settling Velocity (mm/s)")
    ax.set_xticks(xs)
    ax.set_title(lab, fontsize=10, loc="left", fontweight="bold")

handles, labels = ax1.get_legend_handles_labels()
fig2.legend(handles, labels, fontsize=5.5, frameon=False,
            loc="lower center", ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, -0.10))

fig2.savefig(f"{OUT}/stratified_vs_bulk_velocity.png", dpi=300,
             bbox_inches="tight")
fig2.savefig(f"{OUT}/stratified_vs_bulk_velocity.svg", bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {OUT}/stratified_vs_bulk_velocity.png")
