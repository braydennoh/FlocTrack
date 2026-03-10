#!/usr/bin/env python3
"""
Basic measured values (medians + IQR) with Nghiem et al. (2022) model
prediction lines overlaid.

Model predictions are normalized via least-squares best fit to all 3
data points (not pinned to any single point).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

# ── load & stats ──
def stats(csv_name):
    df = pd.read_csv(f"{DATA}/{csv_name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    d = df["Avg_Concave_Diameter"].values          # µm
    v = df["Avg_Velocity"].values / 1000.0          # mm/s
    return {
        "d_med": np.median(d), "d_q25": np.percentile(d, 25),
        "d_q75": np.percentile(d, 75),
        "v_med": np.median(v), "v_q25": np.percentile(v, 25),
        "v_q75": np.percentile(v, 75),
    }

shear_x = np.array([x for _, x in shear_defs])
shear_s = [stats(n) for n, _ in shear_defs]
om_x    = np.array([x for _, x in om_defs])
om_s    = [stats(n) for n, _ in om_defs]

# ── θ from %OM using Nghiem Eq. 19 ──
rho_s, rho_om = 2650, 1000
D_p, delta = 12e-6, 1e-6
theta_factor = (rho_s / rho_om) * D_p**3 / ((D_p + delta)**3 - D_p**3)

def om_to_theta(pct_om):
    return (pct_om / 100.0) * theta_factor

# ── model shape functions (un-normalized) ──
def shape_shear(u_star):
    """Model shape: ∝ u*^{−1/2}"""
    return u_star**(-0.5)

def shape_om_Df(pct_om):
    """Model shape: D_f ∝ (θ²(1−θ)²)^{0.147}"""
    t = om_to_theta(pct_om)
    return (t**2 * (1 - t)**2)**0.147

def shape_om_ws(pct_om):
    """Model shape: w_s ∝ (θ²(1−θ)²)^{0.167}"""
    t = om_to_theta(pct_om)
    return (t**2 * (1 - t)**2)**0.167

def best_fit_scale(x_data, y_data, shape_fn):
    """Find C that minimizes Σ(y - C·f(x))² → C = Σ(y·f) / Σ(f²)"""
    f_vals = np.array([shape_fn(x) for x in x_data])
    return np.sum(y_data * f_vals) / np.sum(f_vals**2)

# ── colours ──
c_diam  = "#2E4057"
c_vel   = "#7BC8A4"
c_model = "#FF6C0C"

# ── figure ──
fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.8), constrained_layout=True)

def plot_data(ax, x_vals, stat_list, key_med, key_lo, key_hi,
              ylabel, xlabel, color, marker="o"):
    meds = np.array([s[key_med] for s in stat_list])
    lo   = np.array([s[key_lo]  for s in stat_list])
    hi   = np.array([s[key_hi]  for s in stat_list])
    yerr = np.array([meds - lo, hi - meds])
    ax.errorbar(x_vals, meds, yerr=yerr,
                fmt=marker, ms=8, color=color, mec="white", mew=1.2,
                ecolor=color, capsize=5, capthick=1.5, lw=1.5, zorder=10,
                label="Measured")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_vals)

u_fine  = np.linspace(0.018, 0.042, 50)
om_fine = np.linspace(0.8, 3.2, 50)

# ── (a) Diameter vs Shear ──
plot_data(axes[0, 0], shear_x, shear_s, "d_med", "d_q25", "d_q75",
          r"Floc Diameter ($\mu$m)", "Shear Velocity (m/s)", c_diam)
d_sh = np.array([s["d_med"] for s in shear_s])
C = best_fit_scale(shear_x, d_sh, shape_shear)
axes[0, 0].plot(u_fine, C * np.array([shape_shear(u) for u in u_fine]),
                '--', color=c_model, lw=1.8, zorder=5,
                label="Nghiem et al. (2022)")
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")
axes[0, 0].legend(fontsize=6.5, frameon=False)

# ── (b) Diameter vs OM ──
plot_data(axes[0, 1], om_x, om_s, "d_med", "d_q25", "d_q75",
          r"Floc Diameter ($\mu$m)", "Organic Matter (%)", c_diam)
d_om = np.array([s["d_med"] for s in om_s])
C = best_fit_scale(om_x, d_om, shape_om_Df)
axes[0, 1].plot(om_fine, C * np.array([shape_om_Df(x) for x in om_fine]),
                '--', color=c_model, lw=1.8, zorder=5,
                label="Nghiem et al. (2022)")
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")
axes[0, 1].legend(fontsize=6.5, frameon=False)

# ── (c) Velocity vs Shear ──
plot_data(axes[1, 0], shear_x, shear_s, "v_med", "v_q25", "v_q75",
          "Settling Velocity (mm/s)", "Shear Velocity (m/s)", c_vel)
v_sh = np.array([s["v_med"] for s in shear_s])
C = best_fit_scale(shear_x, v_sh, shape_shear)
axes[1, 0].plot(u_fine, C * np.array([shape_shear(u) for u in u_fine]),
                '--', color=c_model, lw=1.8, zorder=5,
                label="Nghiem et al. (2022)")
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")
axes[1, 0].legend(fontsize=6.5, frameon=False)

# ── (d) Velocity vs OM ──
plot_data(axes[1, 1], om_x, om_s, "v_med", "v_q25", "v_q75",
          "Settling Velocity (mm/s)", "Organic Matter (%)", c_vel)
v_om = np.array([s["v_med"] for s in om_s])
C = best_fit_scale(om_x, v_om, shape_om_ws)
axes[1, 1].plot(om_fine, C * np.array([shape_om_ws(x) for x in om_fine]),
                '--', color=c_model, lw=1.8, zorder=5,
                label="Nghiem et al. (2022)")
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")
axes[1, 1].legend(fontsize=6.5, frameon=False)

fig.savefig(f"{OUT}/basic_with_model.png", dpi=300)
fig.savefig(f"{OUT}/basic_with_model.svg")
plt.close(fig)

# ── print summary ──
print("=" * 72)
print("Model vs Data (best-fit normalization)")
print("=" * 72)
print("\nShear sweep (model shape: u*^{-1/2}):")
C_d = best_fit_scale(shear_x, d_sh, shape_shear)
C_v = best_fit_scale(shear_x, v_sh, shape_shear)
for i, (name, u) in enumerate(shear_defs):
    print(f"  {name:>10}:  D_data={d_sh[i]:.1f}  D_model={C_d*shape_shear(u):.1f}"
          f"  |  ws_data={v_sh[i]:.3f}  ws_model={C_v*shape_shear(u):.3f}")

print(f"\nOM sweep:")
C_d = best_fit_scale(om_x, d_om, shape_om_Df)
C_v = best_fit_scale(om_x, v_om, shape_om_ws)
for i, (name, om) in enumerate(om_defs):
    print(f"  {name:>4}:  D_data={d_om[i]:.1f}  D_model={C_d*shape_om_Df(om):.1f}"
          f"  |  ws_data={v_om[i]:.3f}  ws_model={C_v*shape_om_ws(om):.3f}")

print(f"\nSaved → {OUT}/basic_with_model.png")
print(f"Saved → {OUT}/basic_with_model.svg")
