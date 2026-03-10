#!/usr/bin/env python3
"""
Back-calculate effective floc density from the explicit fractal
settling velocity model (Strom & Keyvani, 2011; Winterwerp, 1998):

  w_s = (g R_s) / (b_1 ν d_p^{n_f-3})  ·  d_f^{n_f-1}       (Eq. 4)

Given measured w_s, D_f, and n_f^{3D}, solve for D_p, then compute:
  φ     = (D_f / D_p)^{n_f - 3}        (solid fraction)
  ρ_floc = ρ_w + (ρ_s - ρ_w) φ          (floc density)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/basictest"

# physical constants
rho_w = 1000.0    # kg/m³
rho_s = 2650.0    # kg/m³
nu    = 1.0e-6    # m²/s
g     = 9.81      # m/s²
R_s   = (rho_s - rho_w) / rho_w   # = 1.65
b_1   = 18.0      # Stokes drag (c_1 = 18, Ω = 1)

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

# 2D→3D regressions
regressions = {
    "LOWSHEAR":  (1.421, -0.391),
    "MIDSHEAR":  (1.069,  0.315),
    "HIGHSHEAR": (0.936,  0.730),
    "OM1":       (1.433, -0.332),
    "OM2":       (0.661,  0.685),
    "OM3":       (0.904,  0.347),
}

def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]

    # convert to SI
    df["Df_m"]  = df["Avg_Concave_Diameter"] * 1e-6     # m
    df["ws_ms"] = df["Avg_Velocity"] * 1e-6              # m/s

    # nf_3D from regression
    m, b = regressions[name]
    df["nf3d"] = m * df["Avg_Perimeter_Fractal_Dimension"] + b

    # Solve Eq. 4 for D_p:
    #   d_p^{n_f-3} = g R_s d_f^{n_f-1} / (b_1 ν w_s)
    #   d_p = (g R_s d_f^{n_f-1} / (b_1 ν w_s))^{1/(n_f-3)}
    nf = df["nf3d"].values
    Df = df["Df_m"].values
    ws = df["ws_ms"].values

    dp_pow = g * R_s * Df**(nf - 1) / (b_1 * nu * ws)
    # d_p^{n_f-3} = dp_pow   →   d_p = dp_pow^{1/(n_f-3)}
    # Note: n_f-3 < 0 always, so dp_pow^{1/(nf-3)} = dp_pow^{negative}
    df["Dp_m"] = dp_pow ** (1.0 / (nf - 3))
    df["Dp_um"] = df["Dp_m"] * 1e6  # µm

    # Solid fraction:  φ = (D_f / D_p)^{n_f - 3}
    df["phi"] = (Df / df["Dp_m"].values) ** (nf - 3)

    # Floc density
    df["rho_floc"]  = rho_w + (rho_s - rho_w) * df["phi"]
    df["delta_rho"] = (rho_s - rho_w) * df["phi"]

    # convenient units
    df["Df_um"]  = df["Avg_Concave_Diameter"]
    df["ws_mms"] = df["Avg_Velocity"] / 1000.0

    return df

all_data = {n: load(n) for n, _ in shear_defs + om_defs}

def get_medians(name):
    df = all_data[name]
    keys = ["Df_um", "ws_mms", "delta_rho", "nf3d", "phi", "Dp_um", "rho_floc"]
    short = ["Df", "ws", "drho", "nf3d", "phi", "Dp", "rho"]
    out = {}
    for k, s in zip(keys, short):
        vals = df[k].dropna()
        out[s]       = np.median(vals)
        out[s+"_25"] = np.percentile(vals, 25)
        out[s+"_75"] = np.percentile(vals, 75)
    return out

# ── print summary ──
print("=" * 105)
print("Floc Properties from Explicit Model (Eq. 4):  w_s = g R_s / (b_1 ν d_p^{nf-3}) · d_f^{nf-1}")
print(f"  R_s = {R_s:.2f},  b_1 = {b_1:.0f} (Stokes, Ω=1)")
print("=" * 105)
print(f"{'Dataset':<12} {'D_f(µm)':>8} {'w_s(mm/s)':>10} {'n_f^3D':>7} "
      f"{'D_p(µm)':>8} {'φ':>6} {'Δρ(kg/m³)':>10} {'ρ_floc':>8}")
print("-" * 105)
for name, x in shear_defs + om_defs:
    s = get_medians(name)
    print(f"{name:<12} {s['Df']:>8.1f} {s['ws']:>10.3f} {s['nf3d']:>7.2f} "
          f"{s['Dp']:>8.2f} {s['phi']:>6.3f} {s['drho']:>10.1f} {s['rho']:>8.0f}")
print("=" * 105)

# ── colours ──
c_diam = "#2E4057"
c_vel  = "#7BC8A4"
c_rho  = "#E8575A"   # warm red for density
c_nf   = "#8B6DAF"   # purple for nf

# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE: 4-row × 2-col showing all variables
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 2, figsize=(5.6, 8.0), constrained_layout=True)

def plot_row(row, defs, key, key_lo, key_hi, ylabel, color, xlabel):
    xs = np.array([x for _, x in defs])
    ss = [get_medians(n) for n, _ in defs]
    meds = np.array([s[key] for s in ss])
    lo   = meds - np.array([s[key_lo] for s in ss])
    hi   = np.array([s[key_hi] for s in ss]) - meds
    for col_idx, (ax, x_arr, xlabel_t) in enumerate([
        (axes[row, 0], xs, "Shear Velocity (m/s)" if row == 3 else ""),
        (axes[row, 1], np.array([x for _, x in om_defs]),
         "Organic Matter (%)" if row == 3 else "")]):

        if col_idx == 0:
            s_list = [get_medians(n) for n, _ in shear_defs]
            x_plot = np.array([x for _, x in shear_defs])
        else:
            s_list = [get_medians(n) for n, _ in om_defs]
            x_plot = np.array([x for _, x in om_defs])

        m = np.array([s[key] for s in s_list])
        l = m - np.array([s[key_lo] for s in s_list])
        h = np.array([s[key_hi] for s in s_list]) - m

        ax.errorbar(x_plot, m, yerr=[l, h],
                    fmt='-o', ms=7, color=color, mec='white', mew=1.0,
                    ecolor=color, capsize=4, capthick=1.2, lw=1.5, zorder=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x_plot)
        if xlabel_t:
            ax.set_xlabel(xlabel_t)
        else:
            ax.set_xticklabels([])

# Row 0: Diameter
plot_row(0, shear_defs, "Df", "Df_25", "Df_75",
         r"Floc Diameter ($\mu$m)", c_diam, "")
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")

# Row 1: Velocity
plot_row(1, shear_defs, "ws", "ws_25", "ws_75",
         "Settling Vel. (mm/s)", c_vel, "")
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")

# Row 2: Excess density
plot_row(2, shear_defs, "drho", "drho_25", "drho_75",
         r"$\Delta\rho$ (kg/m³)", c_rho, "")
axes[2, 0].set_title("(e)", fontsize=11, loc="left", fontweight="bold")
axes[2, 1].set_title("(f)", fontsize=11, loc="left", fontweight="bold")

# Row 3: nf_3D
plot_row(3, shear_defs, "nf3d", "nf3d_25", "nf3d_75",
         r"$n_f^{(3D)}$", c_nf, "Shear Velocity (m/s)")
axes[3, 0].set_title("(g)", fontsize=11, loc="left", fontweight="bold")
axes[3, 1].set_title("(h)", fontsize=11, loc="left", fontweight="bold")

fig.savefig(f"{OUT}/floc_density_4panel.png", dpi=300)
fig.savefig(f"{OUT}/floc_density_4panel.svg")
plt.close(fig)
print(f"\nSaved → {OUT}/floc_density_4panel.png")
print(f"Saved → {OUT}/floc_density_4panel.svg")
