#!/usr/bin/env python3
"""
5-row × 2-col: D_f, D_p, n_f^{3D}, φ, w_s
across shear (left) and OM (right).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
RES  = "/Users/braydennoh/Research/Floc/results"
OUT  = f"{RES}/densitytest"

regressions = {
    "LOWSHEAR":  (1.421, -0.391), "MIDSHEAR": (1.069, 0.315),
    "HIGHSHEAR": (0.936,  0.730), "OM1":      (1.433, -0.332),
    "OM2":       (0.661,  0.685), "OM3":      (0.904,  0.347),
}

dp_samples = {}
for name, sub in [("LOWSHEAR","lowshear"),("MIDSHEAR","midshear"),
                   ("HIGHSHEAR","highshear"),("OM1","om1"),
                   ("OM2","om2"),("OM3","om3")]:
    dp_samples[name] = np.load(f"{RES}/{sub}/{name}_dp_samples.npy") * 1e6

shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]
    Df   = df["Avg_Concave_Diameter"].values           # µm
    ws   = df["Avg_Velocity"].values / 1000.0           # mm/s
    Df_m = Df * 1e-6
    m_r, b_r = regressions[name]
    nf3d = m_r * df["Avg_Perimeter_Fractal_Dimension"].values + b_r
    Dp_med = np.median(dp_samples[name]) * 1e-6
    phi = (Df_m / Dp_med) ** (nf3d - 3.0)
    ok = (phi > 0) & (phi < 1)
    return {"Df": Df[ok], "ws": ws[ok], "nf3d": nf3d[ok], "phi": phi[ok]}

all_data = {n: load(n) for n, _ in shear_defs + om_defs}

def get_meds(defs, key):
    meds, lo, hi = [], [], []
    for name, _ in defs:
        arr = all_data[name][key]
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        meds.append(med); lo.append(med - q25); hi.append(q75 - med)
    return np.array(meds), np.array(lo), np.array(hi)

def get_dp_meds(defs):
    meds, lo, hi = [], [], []
    for name, _ in defs:
        arr = dp_samples[name]
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        meds.append(med); lo.append(med - q25); hi.append(q75 - med)
    return np.array(meds), np.array(lo), np.array(hi)

# ── colours ──
c_Df = "#2E4057"
c_Dp = "#FF6C0C"
c_nf = "#8B6DAF"
c_phi = "#7BC8A4"
c_ws = "#E8575A"

shear_x = np.array([x for _, x in shear_defs])
om_x    = np.array([x for _, x in om_defs])

fig, axes = plt.subplots(5, 2, figsize=(5.6, 10.0), constrained_layout=False)
fig.subplots_adjust(left=0.12, right=0.97, hspace=0.35, wspace=0.35,
                    bottom=0.06, top=0.92)

fig.text(0.5, 0.96,
         r"$\varphi = \left(\dfrac{D_f}{D_p}\right)^{\!n_f - 3}$",
         fontsize=15, ha="center", va="center")

rows = [
    ("Df",   c_Df,  r"$D_f$ ($\mu$m)"),
    ("Dp",   c_Dp,  r"$D_p$ ($\mu$m)"),
    ("nf3d", c_nf,  r"$n_f^{(3D)}$"),
    ("phi",  c_phi, r"$\varphi$ (solid fraction)"),
    ("ws",   c_ws,  r"$w_s$ (mm/s)"),
]

labels = iter("abcdefghij")

for row, (key, color, ylabel) in enumerate(rows):
    for col, (defs, x_arr, xlabel) in enumerate([
        (shear_defs, shear_x, "Shear Velocity (m/s)"),
        (om_defs,    om_x,    "Organic Matter (%)")]):

        ax = axes[row, col]

        if key == "Dp":
            meds, lo, hi = get_dp_meds(defs)
        else:
            meds, lo, hi = get_meds(defs, key)

        ax.errorbar(x_arr, meds, yerr=[lo, hi],
                    fmt='-o', ms=7, color=color, mec='white', mew=1.0,
                    ecolor=color, capsize=4, capthick=1.2, lw=1.5, zorder=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x_arr)
        ax.set_title(f"({next(labels)})", fontsize=11, loc="left",
                     fontweight="bold")

        if row == 4:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])

fig.savefig(f"{OUT}/distributions_5row.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/distributions_5row.svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/distributions_5row.png")
