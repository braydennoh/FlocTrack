#!/usr/bin/env python3
"""
3-across × 2-down: input sediment distribution vs D_p distribution.
Top row: shear experiments.  Bottom row: OM experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "Times New Roman"

RES = "/Users/braydennoh/Research/Floc/results"
OUT = f"{RES}/densitytest"
SED = "/Users/braydennoh/Research/Floc/inputseddata"

# ── load input sediment ──
mont = np.loadtxt(f"{SED}/montinput.txt")
sil  = np.loadtxt(f"{SED}/silicainput.txt")

grain_m  = mont[:, 0]              # metres (shared axis)
grain_um = grain_m * 1e6           # µm

# normalise each to pdf (sum to 1)
mont_f = mont[:, 1] / mont[:, 1].sum()
sil_f  = sil[:, 1] / sil[:, 1].sum()

# 50:50 volume-weighted mixture
mix_f = 0.5 * mont_f + 0.5 * sil_f

# ── load D_p samples ──
dp = {}
for name, sub in [("LOWSHEAR","lowshear"),("MIDSHEAR","midshear"),
                   ("HIGHSHEAR","highshear"),("OM1","om1"),
                   ("OM2","om2"),("OM3","om3")]:
    dp[name] = np.load(f"{RES}/{sub}/{name}_dp_samples.npy") * 1e6  # µm

# ── layout ──
top_row = [("LOWSHEAR", r"$u_*$ = 0.02 m/s"),
           ("MIDSHEAR", r"$u_*$ = 0.03 m/s"),
           ("HIGHSHEAR", r"$u_*$ = 0.04 m/s")]
bot_row = [("OM1", "OM = 1%"), ("OM2", "OM = 2%"), ("OM3", "OM = 3%")]

labels = iter("abcdef")

fig, axes = plt.subplots(2, 3, figsize=(5.6, 3.2), constrained_layout=True)

c_input = "#2E4057"
c_dp    = "#7BC8A4"

for row, defs in enumerate([top_row, bot_row]):
    for col, (name, title) in enumerate(defs):
        ax = axes[row, col]

        # input mixture (log-x so plot vs log bins)
        ax.fill_between(grain_um, mix_f, color=c_input, alpha=0.3,
                        label="Input sediment", linewidth=0)
        ax.plot(grain_um, mix_f, color=c_input, lw=1)

        # D_p distribution as KDE
        samples = dp[name]
        # clip to reasonable range for KDE
        samples = samples[(samples > 0.5) & (samples < 200)]
        x_kde = np.linspace(0.5, 200, 500)
        kde = gaussian_kde(samples)
        pdf = kde(x_kde)
        # scale KDE to match input curve height
        pdf_scaled = pdf * (mix_f.max() / pdf.max())
        ax.fill_between(x_kde, pdf_scaled, color=c_dp, alpha=0.3,
                        linewidth=0, label=r"$D_p$ posterior")
        ax.plot(x_kde, pdf_scaled, color=c_dp, lw=1.2)

        # median line
        med = np.median(samples)
        ax.axvline(med, color=c_dp, ls="--", lw=1, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlim(0.1, 300)
        ax.set_ylim(bottom=0)
        ax.set_yticklabels([])
        ax.set_title(f"({next(labels)})  {title}", fontsize=9,
                     loc="left", fontweight="bold")

        if row == 1:
            ax.set_xlabel(r"Grain size ($\mu$m)")
        if col == 0:
            ax.set_ylabel("Density (scaled)")

# shared legend
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, fontsize=7, frameon=False,
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))

fig.savefig(f"{OUT}/input_vs_dp.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/input_vs_dp.svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/input_vs_dp.png")
