#!/usr/bin/env python3
"""
Stratified measured values (binned by nf_3D) with Nghiem et al. (2022)
model predictions.

Each particle's nf_2D is converted to nf_3D using its dataset's
empirical 2D→3D regression, then binned by nf_3D for cross-experiment
comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

# ── empirical 2D→3D regressions (from MCMC analysis) ──
regressions = {
    "LOWSHEAR":  (1.421, -0.391),
    "MIDSHEAR":  (1.069,  0.315),
    "HIGHSHEAR": (0.936,  0.730),
    "OM1":       (1.433, -0.332),
    "OM2":       (0.661,  0.685),
    "OM3":       (0.904,  0.347),
}

# ── nf_3D bins (width 0.2 for sufficient population) ──
nf3d_edges = np.arange(1.4, 2.65, 0.2)

# ── load all data, convert nf_2D → nf_3D ──
def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    df["vel_mm"] = df["Avg_Velocity"] / 1000.0
    m, b = regressions[name]
    df["nf3d"] = m * df["Avg_Perimeter_Fractal_Dimension"] + b
    df["nf3d_bin"] = np.digitize(df["nf3d"], nf3d_edges) - 1
    return df

all_data = {n: load(n) for n, _ in shear_defs + om_defs}

def bin_stats(name, b):
    sub = all_data[name][all_data[name]["nf3d_bin"] == b]
    n = len(sub)
    if n < 15:
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

# ── usable bins ──
def usable_bins(defs):
    good = []
    n_bins = len(nf3d_edges) - 1
    for b in range(n_bins):
        if all(bin_stats(n, b) is not None for n, _ in defs):
            good.append(b)
    return good

shear_bins = usable_bins(shear_defs)
om_bins    = usable_bins(om_defs)

# ── print population table ──
print("nf_3D bin populations:")
print(f"{'Bin':<16}", end="")
for n, _ in shear_defs + om_defs:
    print(f"{n:>10}", end="")
print()
n_bins = len(nf3d_edges) - 1
for b in range(n_bins):
    lo, hi = nf3d_edges[b], nf3d_edges[b+1]
    print(f"[{lo:.1f},{hi:.1f})", end="      ")
    for name, _ in shear_defs + om_defs:
        s = bin_stats(name, b)
        print(f"{s['n'] if s else 0:>10}", end="")
    print("  " + ("← SHEAR" if b in shear_bins else "") +
          (" ← OM" if b in om_bins else ""))
print(f"\nUsable shear bins: {len(shear_bins)}")
print(f"Usable OM bins:    {len(om_bins)}")

# ── model scaling ──
theta_factor = (2650 / 1000) * (12e-6)**3 / ((13e-6)**3 - (12e-6)**3)

def om_to_theta(pct):
    return (pct / 100.0) * theta_factor

def model_shear_ratio(u, u_ref):
    return (u_ref / u)**0.5

def model_om_ratio_Df(pct, pct_ref):
    t, t0 = om_to_theta(pct), om_to_theta(pct_ref)
    return ((t**2 * (1 - t)**2)**0.147) / ((t0**2 * (1 - t0)**2)**0.147)

def model_om_ratio_ws(pct, pct_ref):
    t, t0 = om_to_theta(pct), om_to_theta(pct_ref)
    return ((t**2 * (1 - t)**2)**0.167) / ((t0**2 * (1 - t0)**2)**0.167)

# ── colours ──
c_model = "#FF6C0C"
cmap = cmocean.cm.deep_r

def bin_color(b, bins):
    idx = bins.index(b) if b in bins else 0
    return cmap(0.15 + 0.7 * idx / max(1, len(bins) - 1))

# ── figure: 2×2 ──
fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.2), constrained_layout=True)

u_fine  = np.linspace(0.018, 0.042, 50)
om_fine = np.linspace(0.8, 3.2, 50)

def plot_panel(ax, defs, bins, key_med, key_lo, key_hi,
               ylabel, xlabel, model_fn, model_x_fine):
    xs = np.array([x for _, x in defs])

    for b in bins:
        col = bin_color(b, bins)
        ss = [bin_stats(n, b) for n, _ in defs]
        if any(s is None for s in ss):
            continue

        meds = np.array([s[key_med] for s in ss])
        lo   = meds - np.array([s[key_lo] for s in ss])
        hi   = np.array([s[key_hi] for s in ss]) - meds

        lo_3d = nf3d_edges[b]
        hi_3d = nf3d_edges[b + 1]
        lbl = f"[{lo_3d:.1f},{hi_3d:.1f})"
        ax.errorbar(xs, meds, yerr=[lo, hi],
                    fmt='-o', ms=5, color=col, mec='white', mew=0.7,
                    ecolor=col, capsize=3, capthick=0.8, lw=1.3,
                    label=lbl, zorder=10, alpha=0.85)

        # model prediction (orange dashed)
        ref_val = meds[0]
        ref_x   = xs[0]
        y_model = np.array([ref_val * model_fn(xv, ref_x)
                            for xv in model_x_fine])
        ax.plot(model_x_fine, y_model, '--', color=c_model, lw=1.0,
                alpha=0.55, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)


# (a) Diameter vs Shear
plot_panel(axes[0, 0], shear_defs, shear_bins,
           "d_med", "d_q25", "d_q75",
           r"Floc Diameter ($\mu$m)", "Shear Velocity (m/s)",
           model_shear_ratio, u_fine)
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")

# (b) Diameter vs OM
plot_panel(axes[0, 1], om_defs, om_bins,
           "d_med", "d_q25", "d_q75",
           r"Floc Diameter ($\mu$m)", "Organic Matter (%)",
           model_om_ratio_Df, om_fine)
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")

# (c) Velocity vs Shear
plot_panel(axes[1, 0], shear_defs, shear_bins,
           "v_med", "v_q25", "v_q75",
           "Settling Velocity (mm/s)", "Shear Velocity (m/s)",
           model_shear_ratio, u_fine)
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")

# (d) Velocity vs OM
plot_panel(axes[1, 1], om_defs, om_bins,
           "v_med", "v_q25", "v_q75",
           "Settling Velocity (mm/s)", "Organic Matter (%)",
           model_om_ratio_ws, om_fine)
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")

# ── legend ──
model_handle = Line2D([], [], ls='--', color=c_model, lw=1.5,
                       label="Nghiem et al. (2022)")

handles_bins, labels_bins = axes[0, 0].get_legend_handles_labels()
all_handles = handles_bins + [model_handle]
all_labels  = [f"$n_f^{{3D}}$ {l}" for l in labels_bins] + [model_handle.get_label()]

fig.legend(all_handles, all_labels, fontsize=6, frameon=False,
           loc="lower center", ncol=min(len(all_handles), 5),
           bbox_to_anchor=(0.5, -0.06))

fig.savefig(f"{OUT}/stratified_with_model.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/stratified_with_model.svg", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}/stratified_with_model.png")
print(f"Saved → {OUT}/stratified_with_model.svg")
