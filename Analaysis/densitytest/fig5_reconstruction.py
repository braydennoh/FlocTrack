#!/usr/bin/env python3
"""
Figure 5: Settling velocity reconstruction.

Compare measured w_s trends against:
  (1) Nghiem et al. (2022) model (constant n_f ≈ 2)
  (2) Reconstructed from measured n_f and D_p trends via Eq. 4:
      w_s = g R_s / (b_1 ν) · D_p^{3-n_f} · D_f^{n_f-1}

Shows that variable n_f + D_p resolves the model–data discrepancy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
RES  = "/Users/braydennoh/Research/Floc/results"
OUT  = f"{RES}/densitytest"

# ── physical constants ──
rho_w = 1000.0
rho_s = 2650.0
nu    = 1.0e-6
g     = 9.81
R_s   = (rho_s - rho_w) / rho_w
b_1   = 18.0

# ── dataset definitions ──
shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

regressions = {
    "LOWSHEAR":  (1.421, -0.391), "MIDSHEAR": (1.069, 0.315),
    "HIGHSHEAR": (0.936,  0.730), "OM1":      (1.433, -0.332),
    "OM2":       (0.661,  0.685), "OM3":      (0.904,  0.347),
}

# ── load D_p medians (metres) ──
dp_med = {}
for name, sub in [("LOWSHEAR","lowshear"),("MIDSHEAR","midshear"),
                   ("HIGHSHEAR","highshear"),("OM1","om1"),
                   ("OM2","om2"),("OM3","om3")]:
    dp_med[name] = np.median(np.load(f"{RES}/{sub}/{name}_dp_samples.npy"))

# ── load measured data ──
def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    Df = df["Avg_Concave_Diameter"].values * 1e-6   # m
    ws = df["Avg_Velocity"].values * 1e-6            # m/s
    m_r, b_r = regressions[name]
    nf3d = m_r * df["Avg_Perimeter_Fractal_Dimension"].values + b_r
    return Df, ws, nf3d

all_data = {n: load(n) for n, _ in shear_defs + om_defs}

# ── per-dataset median measured values ──
def measured_meds(name):
    Df, ws, nf3d = all_data[name]
    return {
        "Df": np.median(Df),
        "ws": np.median(ws),
        "nf3d": np.median(nf3d),
        "ws_q25": np.percentile(ws, 25),
        "ws_q75": np.percentile(ws, 75),
    }

# ── Eq. 4 prediction ──
def ws_eq4(Df, Dp, nf):
    """w_s = g R_s / (b_1 ν) · D_p^{3-n_f} · D_f^{n_f-1}"""
    return (g * R_s / (b_1 * nu)) * Dp**(3 - nf) * Df**(nf - 1)

# ── Nghiem model shape (constant n_f ≈ 2) ──
# For shear: w_s ∝ u*^{-0.5}
# For OM: w_s ∝ (θ²(1-θ)²)^{0.167}
theta_factor = (rho_s / 1000) * (12e-6)**3 / ((13e-6)**3 - (12e-6)**3)

def nghiem_om_shape(pct):
    t = (pct / 100.0) * theta_factor
    return (t**2 * (1 - t)**2)**0.167

def nghiem_shear_shape(u):
    return u**(-0.5)

def best_fit_scale(x_data, y_data, shape_fn):
    f = np.array([shape_fn(x) for x in x_data])
    return np.sum(y_data * f) / np.sum(f**2)

# ── compute everything ──
shear_x = np.array([x for _, x in shear_defs])
om_x    = np.array([x for _, x in om_defs])

# Measured
sh_meas = [measured_meds(n) for n, _ in shear_defs]
om_meas = [measured_meds(n) for n, _ in om_defs]

ws_sh_meas = np.array([s["ws"] for s in sh_meas]) * 1e3  # mm/s
ws_sh_lo   = ws_sh_meas - np.array([s["ws_q25"] for s in sh_meas]) * 1e3
ws_sh_hi   = np.array([s["ws_q75"] for s in sh_meas]) * 1e3 - ws_sh_meas

ws_om_meas = np.array([s["ws"] for s in om_meas]) * 1e3
ws_om_lo   = ws_om_meas - np.array([s["ws_q25"] for s in om_meas]) * 1e3
ws_om_hi   = np.array([s["ws_q75"] for s in om_meas]) * 1e3 - ws_om_meas

# Reconstructed (Eq. 4 with measured median D_f, derived n_f and D_p)
ws_sh_recon = []
for name, _ in shear_defs:
    s = measured_meds(name)
    ws_sh_recon.append(ws_eq4(s["Df"], dp_med[name], s["nf3d"]) * 1e3)
ws_sh_recon = np.array(ws_sh_recon)

ws_om_recon = []
for name, _ in om_defs:
    s = measured_meds(name)
    ws_om_recon.append(ws_eq4(s["Df"], dp_med[name], s["nf3d"]) * 1e3)
ws_om_recon = np.array(ws_om_recon)

# Nghiem (best-fit normalized to measured)
C_sh = best_fit_scale(shear_x, ws_sh_meas,  nghiem_shear_shape)
C_om = best_fit_scale(om_x,    ws_om_meas,  nghiem_om_shape)

u_fine  = np.linspace(0.018, 0.042, 50)
om_fine = np.linspace(0.8, 3.2, 50)

ws_sh_nghiem = C_sh * np.array([nghiem_shear_shape(u) for u in u_fine])
ws_om_nghiem = C_om * np.array([nghiem_om_shape(x) for x in om_fine])

# ── colours ──
c_meas   = "#2E4057"
c_recon  = "#7BC8A4"
c_nghiem = "#FF6C0C"

# ── figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.4), constrained_layout=True)

# (a) Shear
ax1.errorbar(shear_x, ws_sh_meas, yerr=[ws_sh_lo, ws_sh_hi],
             fmt='-o', ms=7, color=c_meas, mec='white', mew=1.0,
             ecolor=c_meas, capsize=4, capthick=1.2, lw=1.5, zorder=10,
             label="Measured")
ax1.plot(shear_x, ws_sh_recon, 's--', ms=7, color=c_recon, mec='white',
         mew=1.0, lw=1.5, zorder=10, label="Eq. 4 (variable $n_f$, $D_p$)")
ax1.plot(u_fine, ws_sh_nghiem, '--', color=c_nghiem, lw=1.2, alpha=0.8,
         label="Nghiem (const. $n_f$)")
ax1.set_xlabel("Shear Velocity (m/s)")
ax1.set_ylabel("Settling Velocity (mm/s)")
ax1.set_xticks(shear_x)
ax1.legend(fontsize=5.5, frameon=False)

# (b) OM
ax2.errorbar(om_x, ws_om_meas, yerr=[ws_om_lo, ws_om_hi],
             fmt='-o', ms=7, color=c_meas, mec='white', mew=1.0,
             ecolor=c_meas, capsize=4, capthick=1.2, lw=1.5, zorder=10,
             label="Measured")
ax2.plot(om_x, ws_om_recon, 's--', ms=7, color=c_recon, mec='white',
         mew=1.0, lw=1.5, zorder=10, label="Eq. 4 (variable $n_f$, $D_p$)")
ax2.plot(om_fine, ws_om_nghiem, '--', color=c_nghiem, lw=1.2, alpha=0.8,
         label="Nghiem (const. $n_f$)")
ax2.set_xlabel("Organic Matter (%)")
ax2.set_ylabel("Settling Velocity (mm/s)")
ax2.set_xticks(om_x)
ax2.legend(fontsize=5.5, frameon=False)

fig.savefig(f"{OUT}/fig5_reconstruction.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_reconstruction.svg", bbox_inches="tight")
plt.close(fig)

# ── print comparison ──
print("Shear sweep (mm/s):")
for i, (name, u) in enumerate(shear_defs):
    print(f"  {name:>10}: measured={ws_sh_meas[i]:.3f}  "
          f"recon={ws_sh_recon[i]:.3f}  "
          f"nghiem={C_sh*nghiem_shear_shape(u):.3f}")
print("\nOM sweep (mm/s):")
for i, (name, om) in enumerate(om_defs):
    print(f"  {name:>4}: measured={ws_om_meas[i]:.3f}  "
          f"recon={ws_om_recon[i]:.3f}  "
          f"nghiem={C_om*nghiem_om_shape(om):.3f}")

print(f"\nSaved → {OUT}/fig5_reconstruction.png")
