#!/usr/bin/env python3
"""
Basic measured values: floc diameter and settling velocity
(medians with IQR bars) vs shear velocity and organic matter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

# ── dataset definitions ──
shear_datasets = [
    ("LOWSHEAR",  0.02),
    ("MIDSHEAR",  0.03),
    ("HIGHSHEAR", 0.04),
]
om_datasets = [
    ("OM1", 1),
    ("OM2", 2),
    ("OM3", 3),
]

# ── load & compute stats ──
def stats(csv_name):
    """Return median, q25, q75 for diameter (µm) and velocity (mm/s)."""
    df = pd.read_csv(f"{DATA}/{csv_name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    diam = df["Avg_Concave_Diameter"].values          # µm
    vel  = df["Avg_Velocity"].values / 1000.0          # µm/s → mm/s
    return {
        "d_med": np.median(diam),
        "d_q25": np.percentile(diam, 25),
        "d_q75": np.percentile(diam, 75),
        "v_med": np.median(vel),
        "v_q25": np.percentile(vel, 25),
        "v_q75": np.percentile(vel, 75),
        "n":     len(diam),
    }

# gather results
shear_x, shear_stats = [], []
for name, x in shear_datasets:
    shear_x.append(x)
    shear_stats.append(stats(name))

om_x, om_stats = [], []
for name, x in om_datasets:
    om_x.append(x)
    om_stats.append(stats(name))

shear_x = np.array(shear_x)
om_x    = np.array(om_x)

# ── colours ──
c_diam = "#2E4057"   # dark navy
c_vel  = "#7BC8A4"   # green

# ── figure: 2×2 ──
fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8), constrained_layout=True)

def plot_panel(ax, x_vals, stat_list, key_med, key_lo, key_hi,
               ylabel, xlabel, color, label, marker="o"):
    meds = np.array([s[key_med] for s in stat_list])
    lo   = np.array([s[key_lo]  for s in stat_list])
    hi   = np.array([s[key_hi]  for s in stat_list])
    yerr = np.array([meds - lo, hi - meds])
    ax.errorbar(x_vals, meds, yerr=yerr,
                fmt=marker, ms=8, color=color, mec="white", mew=1.2,
                ecolor=color, capsize=5, capthick=1.5, lw=1.5, zorder=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_vals)

# (a) Diameter vs Shear Velocity
plot_panel(axes[0, 0], shear_x, shear_stats,
           "d_med", "d_q25", "d_q75",
           r"Floc Diameter ($\mu$m)", "Shear Velocity (m/s)",
           c_diam, "Diameter")
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")

# (b) Diameter vs Organic Matter
plot_panel(axes[0, 1], om_x, om_stats,
           "d_med", "d_q25", "d_q75",
           r"Floc Diameter ($\mu$m)", "Organic Matter (%)",
           c_diam, "Diameter")
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")

# (c) Settling Velocity vs Shear Velocity
plot_panel(axes[1, 0], shear_x, shear_stats,
           "v_med", "v_q25", "v_q75",
           "Settling Velocity (mm/s)", "Shear Velocity (m/s)",
           c_vel, "Velocity")
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")

# (d) Settling Velocity vs Organic Matter
plot_panel(axes[1, 1], om_x, om_stats,
           "v_med", "v_q25", "v_q75",
           "Settling Velocity (mm/s)", "Organic Matter (%)",
           c_vel, "Velocity")
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")

fig.savefig(f"{OUT}/basic_measured_values.png", dpi=300)
fig.savefig(f"{OUT}/basic_measured_values.svg")
plt.close(fig)

# ── print summary table ──
print("=" * 72)
print("Basic Measured Values Summary")
print("=" * 72)
print(f"{'Dataset':<12} {'N':>6}  {'D_med (µm)':>11} {'D_IQR':>11}  "
      f"{'ws_med (mm/s)':>13} {'ws_IQR':>11}")
print("-" * 72)
for name, x in shear_datasets + om_datasets:
    s = stats(name)
    d_iqr = s["d_q75"] - s["d_q25"]
    v_iqr = s["v_q75"] - s["v_q25"]
    print(f"{name:<12} {s['n']:>6}  {s['d_med']:>11.2f} {d_iqr:>11.2f}  "
          f"{s['v_med']:>13.3f} {v_iqr:>11.3f}")
print("=" * 72)
print(f"\nSaved → {OUT}/basic_measured_values.png")
print(f"Saved → {OUT}/basic_measured_values.svg")
