#!/usr/bin/env python3
"""
Figure 6: Median floc diameter and settling velocity vs shear and OM.

Structure:
  - D_f panels (a,b): data + SHARED Nghiem Eq 11/24 trend (normalized).
    D_f prediction is independent of nf (nf cancels in equilibrium).
  - w_s panels (c,d): data + two models that SHARE the same D_f prediction:
      Base:     w_s = k1 × D_f^1                   (Eq 4 with nf=2)
      Modified: w_s = k2 × dp(i)^{3-nf(i)} × D_f^{nf(i)-1}  (Eq 4, variable nf/dp)
    Each model has exactly one free parameter (normalization k).

Shear trend: D_f ∝ η ∝ u*^{-0.75}  (Nghiem Eq 24, other vars fixed)
OM trend:    D_f ∝ [θ²(1-θ)²]^0.147 (Nghiem Eq 24, other vars fixed)
  θ from Nghiem Eq 19, using %OM directly (guar gum ≈ cellulose).

RMSE comparison quantifies which w_s model better reproduces data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
OUT  = "/Users/braydennoh/Research/Floc/results/bulktest"

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

# ── load & stats ──────────────────────────────────────────────────────────
def stats(csv_name):
    df = pd.read_csv(f"{DATA}/{csv_name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    d = df["Avg_Concave_Diameter"].values * 1e-6   # m
    v = df["Avg_Velocity"].values / 1e6             # m/s
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

d_sh = np.array([s["d_med"] for s in shear_s])
v_sh = np.array([s["v_med"] for s in shear_s])
d_om = np.array([s["d_med"] for s in om_s])
v_om = np.array([s["v_med"] for s in om_s])

# ── Measured nf and dp (paper Fig 9, Fig 10) ──────────────────────────────
dp_shear = np.array([21.7e-6, 23.0e-6, 29.8e-6])   # Fig 10d
nf_shear = 17.4 * shear_x + 1.42                     # Fig 9a

dp_om = np.array([34.2e-6, 27.9e-6, 17.6e-6])        # Fig 10h
nf_om = -0.11 * om_x + 1.92                           # Fig 9b

# ── Nghiem Eq 24 D_f trend functions ──────────────────────────────────────
# D_f is independent of nf (nf cancels in equilibrium derivation).
# Shear: D_f ∝ η ∝ u*^{-0.75}   (η = (ν³/ε)^{1/4}, ε ∝ u*³)
# OM:    D_f ∝ [θ²(1-θ)²]^0.147  (Nghiem calibrated Eq 24)

def df_trend_shear(u_star):
    return u_star ** (-0.75)

# θ from Nghiem Eq 19
RHO_S, RHO_OM, DELTA = 2650.0, 1000.0, 1e-6
D_P_THETA = 20e-6   # representative dp for θ calculation

def theta(om_pct):
    dp3   = D_P_THETA ** 3
    shell = (D_P_THETA + DELTA) ** 3 - dp3
    return np.clip((om_pct / 100.0) * (RHO_S / RHO_OM) * dp3 / shell, 0, 1)

def df_trend_om(om_pct):
    th = theta(om_pct)
    return (th**2 * (1 - th)**2) ** 0.147

# ── Normalization helper ──────────────────────────────────────────────────
def fit_k(y_data, trend_at_data):
    """Least-squares scalar: k = argmin Σ(y - k*t)²."""
    return np.dot(y_data, trend_at_data) / np.dot(trend_at_data, trend_at_data)

# ── w_s models (both use the SAME normalized D_f prediction) ──────────────
# Base (Eq 4 with nf=2):       w_s = k_base × D_f^1
# Modified (Eq 4, meas nf,dp): w_s = k_mod × dp_i^{3-nf_i} × D_f^{nf_i - 1}
# Each has exactly 1 free parameter (its normalization k).

def ws_trend_base(df_pred):
    """Base model: w_s ∝ D_f^1 (nf=2, dp absorbed into k)."""
    return df_pred ** 1.0

def ws_trend_mod_shear(df_pred, u_star_arr):
    """Modified model for shear sweep: uses interpolated nf(u*), dp(u*)."""
    nf = 17.4 * u_star_arr + 1.42
    dp = np.interp(u_star_arr, shear_x, dp_shear)
    return dp ** (3 - nf) * df_pred ** (nf - 1)

def ws_trend_mod_om(df_pred, om_arr):
    """Modified model for OM sweep: uses interpolated nf(OM), dp(OM)."""
    nf = -0.11 * om_arr + 1.92
    dp = np.interp(om_arr, om_x, dp_om)
    return dp ** (3 - nf) * df_pred ** (nf - 1)

# ── Fine grids ────────────────────────────────────────────────────────────
u_fine = np.linspace(0.018, 0.042, 200)
om_fine = np.linspace(0.8, 3.2, 200)

# ── Compute D_f trends (normalized to data) ───────────────────────────────
# Shear
df_trend_sh_at_data = df_trend_shear(shear_x)
k_df_sh = fit_k(d_sh, df_trend_sh_at_data)
df_pred_sh_fine = k_df_sh * df_trend_shear(u_fine)
df_pred_sh_data = k_df_sh * df_trend_sh_at_data  # at the 3 data points

# OM
df_trend_om_at_data = df_trend_om(om_x)
k_df_om = fit_k(d_om, df_trend_om_at_data)
df_pred_om_fine = k_df_om * df_trend_om(om_fine)
df_pred_om_data = k_df_om * df_trend_om_at_data

# ── Compute w_s predictions ───────────────────────────────────────────────
# SHEAR — base
ws_base_sh_trend_data = ws_trend_base(df_pred_sh_data)
k_ws_base_sh = fit_k(v_sh, ws_base_sh_trend_data)
ws_base_sh_fine = k_ws_base_sh * ws_trend_base(df_pred_sh_fine)
ws_base_sh_pred = k_ws_base_sh * ws_base_sh_trend_data

# SHEAR — modified
ws_mod_sh_trend_data = ws_trend_mod_shear(df_pred_sh_data, shear_x)
k_ws_mod_sh = fit_k(v_sh, ws_mod_sh_trend_data)
ws_mod_sh_fine = k_ws_mod_sh * ws_trend_mod_shear(df_pred_sh_fine, u_fine)
ws_mod_sh_pred = k_ws_mod_sh * ws_mod_sh_trend_data

# OM — base
ws_base_om_trend_data = ws_trend_base(df_pred_om_data)
k_ws_base_om = fit_k(v_om, ws_base_om_trend_data)
ws_base_om_fine = k_ws_base_om * ws_trend_base(df_pred_om_fine)
ws_base_om_pred = k_ws_base_om * ws_base_om_trend_data

# OM — modified
ws_mod_om_trend_data = ws_trend_mod_om(df_pred_om_data, om_x)
k_ws_mod_om = fit_k(v_om, ws_mod_om_trend_data)
ws_mod_om_fine = k_ws_mod_om * ws_trend_mod_om(df_pred_om_fine, om_fine)
ws_mod_om_pred = k_ws_mod_om * ws_mod_om_trend_data

# ── RMSE ──────────────────────────────────────────────────────────────────
def rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs)**2))

rmse_base_sh = rmse(ws_base_sh_pred, v_sh)
rmse_mod_sh  = rmse(ws_mod_sh_pred, v_sh)
rmse_base_om = rmse(ws_base_om_pred, v_om)
rmse_mod_om  = rmse(ws_mod_om_pred, v_om)

# Combined RMSE across all 6 experiments
rmse_base_all = rmse(np.concatenate([ws_base_sh_pred, ws_base_om_pred]),
                     np.concatenate([v_sh, v_om]))
rmse_mod_all  = rmse(np.concatenate([ws_mod_sh_pred, ws_mod_om_pred]),
                     np.concatenate([v_sh, v_om]))

# ── colours ───────────────────────────────────────────────────────────────
c_data = "#2E4057"     # navy for all data points
c_base = "#C0392B"     # red for Nghiem / base model
c_mod  = "#27AE60"     # green for modified model

# ── FIGURE ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(5, 4))
fig.subplots_adjust(hspace=0.45, wspace=0.55, top=0.94, bottom=0.10,
                    left=0.12, right=0.97)

def errorbar(ax, x, meds, q25, q75, color, label="Data"):
    yerr = np.array([meds - q25, q75 - meds])
    ax.errorbar(x, meds, yerr=yerr,
                fmt='o', ms=6, color=color, mec=color, mew=1.0,
                mfc=(*plt.matplotlib.colors.to_rgb(color), 0.8),
                ecolor=color, capsize=3, capthick=1.0, lw=1.5, zorder=10,
                label=label)

def fmt_axis(ax, ylabel, xlabel, x_ticks, scale_exp=None, unit=None):
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_ticks)
    if scale_exp is not None:
        factor = 10 ** (-scale_exp)
        ax.set_ylabel(ylabel + f"\n({unit})" + rf" $\times 10^{{{scale_exp}}}$")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _, f=factor: f"{x * f:.1f}"))
    else:
        ax.set_ylabel(ylabel)

panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for i, ax in enumerate(axes.flat):
    ax.set_title(panel_labels[i], fontsize=12, fontweight="normal", loc="left")

# ── (a) D_f vs Shear ─────────────────────────────────────────────────────
d_sh_lo = np.array([s["d_q25"] for s in shear_s])
d_sh_hi = np.array([s["d_q75"] for s in shear_s])
errorbar(axes[0, 0], shear_x, d_sh, d_sh_lo, d_sh_hi, c_data)
axes[0, 0].plot(u_fine, df_pred_sh_fine, '--', color=c_base, lw=1.5, zorder=5,
                label=r"Nghiem Eq 24 ($D_f \propto \eta$)")
fmt_axis(axes[0, 0], r"Floc Diameter $d_f$", "Shear Velocity (m/s)",
         shear_x, scale_exp=-5, unit="m")
axes[0, 0].legend(fontsize=5.5, loc="upper right", frameon=False)
ymin_a = min(d_sh_lo.min(), df_pred_sh_fine.min()) * 0.85
ymax_a = max(d_sh_hi.max(), df_pred_sh_fine.max()) * 1.15
axes[0, 0].set_ylim(ymin_a, ymax_a)

# ── (b) D_f vs OM ────────────────────────────────────────────────────────
d_om_lo = np.array([s["d_q25"] for s in om_s])
d_om_hi = np.array([s["d_q75"] for s in om_s])
errorbar(axes[0, 1], om_x, d_om, d_om_lo, d_om_hi, c_data)
axes[0, 1].plot(om_fine, df_pred_om_fine, '--', color=c_base, lw=1.5, zorder=5,
                label=r"Nghiem Eq 24 ($D_f \propto [\theta^2(1-\theta)^2]^{0.147}$)")
fmt_axis(axes[0, 1], r"Floc Diameter $d_f$", "Organic Matter (%)",
         om_x, scale_exp=-5, unit="m")
axes[0, 1].legend(fontsize=5, loc="upper right", frameon=False)
ymin_b = min(d_om_lo.min(), df_pred_om_fine.min()) * 0.85
ymax_b = max(d_om_hi.max(), df_pred_om_fine.max()) * 1.15
axes[0, 1].set_ylim(ymin_b, ymax_b)

# ── (c) w_s vs Shear ─────────────────────────────────────────────────────
v_sh_lo = np.array([s["v_q25"] for s in shear_s])
v_sh_hi = np.array([s["v_q75"] for s in shear_s])
errorbar(axes[1, 0], shear_x, v_sh, v_sh_lo, v_sh_hi, c_data)
axes[1, 0].plot(u_fine, ws_base_sh_fine, '--', color=c_base, lw=1.5, zorder=5,
                label=r"Base ($n_f=2$)")
axes[1, 0].plot(u_fine, ws_mod_sh_fine, '-', color=c_mod, lw=2.0, zorder=6,
                label=r"Modified (Measured $n_f,d_p$)")
fmt_axis(axes[1, 0], r"Settling Velocity $w_s$", "Shear Velocity (m/s)",
         shear_x, scale_exp=-4, unit="m/s")
axes[1, 0].legend(fontsize=5.5, loc="upper left", frameon=False)
ymin_c = min(v_sh_lo.min(), ws_base_sh_fine.min(), ws_mod_sh_fine.min()) * 0.85
ymax_c = max(v_sh_hi.max(), ws_base_sh_fine.max(), ws_mod_sh_fine.max()) * 1.15
axes[1, 0].set_ylim(ymin_c, ymax_c)

# ── (d) w_s vs OM ────────────────────────────────────────────────────────
v_om_lo = np.array([s["v_q25"] for s in om_s])
v_om_hi = np.array([s["v_q75"] for s in om_s])
errorbar(axes[1, 1], om_x, v_om, v_om_lo, v_om_hi, c_data)
axes[1, 1].plot(om_fine, ws_base_om_fine, '--', color=c_base, lw=1.5, zorder=5,
                label=r"Base ($n_f=2$)")
axes[1, 1].plot(om_fine, ws_mod_om_fine, '-', color=c_mod, lw=2.0, zorder=6,
                label=r"Modified (Measured $n_f,d_p$)")
fmt_axis(axes[1, 1], r"Settling Velocity $w_s$", "Organic Matter (%)",
         om_x, scale_exp=-4, unit="m/s")
axes[1, 1].legend(fontsize=5.5, loc="upper right", frameon=False)
ymin_d = min(v_om_lo.min(), ws_base_om_fine.min(), ws_mod_om_fine.min()) * 0.85
ymax_d = max(v_om_hi.max(), ws_base_om_fine.max(), ws_mod_om_fine.max()) * 1.15
axes[1, 1].set_ylim(ymin_d, ymax_d)

fig.savefig(f"{OUT}/basic_with_model.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────
print("=" * 65)
print("Figure 6 — Normalized Nghiem D_f + base vs modified w_s")
print("=" * 65)
print(f"\nD_f normalization constants:")
print(f"  Shear: k = {k_df_sh:.4e}")
print(f"  OM:    k = {k_df_om:.4e}")
print(f"\nw_s normalization constants:")
print(f"  Shear base: k = {k_ws_base_sh:.4e}   mod: k = {k_ws_mod_sh:.4e}")
print(f"  OM    base: k = {k_ws_base_om:.4e}   mod: k = {k_ws_mod_om:.4e}")

print(f"\n{'':>12} {'ws_data':>10} {'ws_base':>10} {'ws_mod':>10}")
print("-" * 45)
for i, (name, _) in enumerate(shear_defs):
    print(f"  {name:>10} {v_sh[i]:.2e}   {ws_base_sh_pred[i]:.2e}   {ws_mod_sh_pred[i]:.2e}")
for i, (name, _) in enumerate(om_defs):
    print(f"  {name:>10} {v_om[i]:.2e}   {ws_base_om_pred[i]:.2e}   {ws_mod_om_pred[i]:.2e}")

print(f"\nRMSE (shear):  base = {rmse_base_sh:.2e}   mod = {rmse_mod_sh:.2e}")
print(f"RMSE (OM):     base = {rmse_base_om:.2e}   mod = {rmse_mod_om:.2e}")
print(f"RMSE (all 6):  base = {rmse_base_all:.2e}   mod = {rmse_mod_all:.2e}")
print(f"\nImprovement:   {(1 - rmse_mod_all/rmse_base_all)*100:.0f}% reduction in RMSE")
print(f"\nSaved → {OUT}/basic_with_model.pdf")
