#!/usr/bin/env python3
"""
Scatter: φ_Stokes vs φ_fractal for all particles.

  φ_Stokes  = 18 μ w_s / [g D_f² (ρ_s − ρ_w)]   (from force balance)
  φ_fractal = (D_f / D_p)^{n_f − 3}              (from geometry)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
RES  = "/Users/braydennoh/Research/Floc/results"
OUT  = f"{RES}/densitytest"

rho_w, rho_s = 1000.0, 2650.0
mu, g = 1.0e-3, 9.81

regressions = {
    "LOWSHEAR":  (1.421, -0.391), "MIDSHEAR": (1.069, 0.315),
    "HIGHSHEAR": (0.936,  0.730), "OM1":      (1.433, -0.332),
    "OM2":       (0.661,  0.685), "OM3":      (0.904,  0.347),
}

dp_med = {}
for name, sub in [("LOWSHEAR","lowshear"),("MIDSHEAR","midshear"),
                   ("HIGHSHEAR","highshear"),("OM1","om1"),
                   ("OM2","om2"),("OM3","om3")]:
    dp_med[name] = np.median(np.load(f"{RES}/{sub}/{name}_dp_samples.npy"))

shear_defs = [("LOWSHEAR",0.02),("MIDSHEAR",0.03),("HIGHSHEAR",0.04)]
om_defs    = [("OM1",1),("OM2",2),("OM3",3)]

def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter","Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"]>0)&(df["Avg_Velocity"]>0)]
    Df = df["Avg_Concave_Diameter"].values * 1e-6
    ws = df["Avg_Velocity"].values * 1e-6
    m, b = regressions[name]
    nf = m * df["Avg_Perimeter_Fractal_Dimension"].values + b
    Dp = dp_med[name]
    phi_stokes  = 18*mu*ws / (g * Df**2 * (rho_s - rho_w))
    phi_fractal = (Df / Dp) ** (nf - 3.0)
    ok = (phi_stokes>0)&(phi_stokes<1)&(phi_fractal>0)&(phi_fractal<1)
    return phi_stokes[ok], phi_fractal[ok]

# ── single scatter ──
fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)

colors = {"LOWSHEAR":"#4477AA","MIDSHEAR":"#66CCEE","HIGHSHEAR":"#228833",
          "OM1":"#CCBB44","OM2":"#EE6677","OM3":"#AA3377"}

for name, _ in shear_defs + om_defs:
    ps, pf = load(name)
    ax.scatter(pf, ps, s=1.5, alpha=0.2, color=colors[name],
               label=name, rasterized=True)

ax.plot([0,1],[0,1],'k--',lw=0.8, label="1 : 1")
ax.set_xlabel(r"$\varphi_{\mathrm{fractal}}$ = $(D_f/D_p)^{n_f-3}$")
ax.set_ylabel(r"$\varphi_{\mathrm{Stokes}}$ = $18\mu w_s\,/\,[g\,D_f^2\,(\rho_s-\rho_w)]$")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_aspect("equal")
ax.legend(fontsize=6, frameon=False, markerscale=4)

fig.savefig(f"{OUT}/phi_scatter.png", dpi=300)
fig.savefig(f"{OUT}/phi_scatter.svg")
plt.close(fig)
print(f"Saved → {OUT}/phi_scatter.png")
