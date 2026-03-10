#!/usr/bin/env python3
"""
2×4 layout: columns 1-3 show input sediment vs D_p distributions,
column 4 shows D_p median trend vs shear (top) and OM (bottom).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

plt.rcParams["font.family"] = "Times New Roman"

RES = "/Users/braydennoh/Research/Floc/results"
OUT = f"{RES}/dpdata"
SED = "/Users/braydennoh/Research/Floc/inputseddata"

# ── load input sediment ──
mont = np.loadtxt(f"{SED}/montinput.txt")
sil  = np.loadtxt(f"{SED}/silicainput.txt")

grain = mont[:, 0]  # already in metres

mont_f = mont[:, 1] / mont[:, 1].sum()
sil_f  = sil[:, 1] / sil[:, 1].sum()
mix_f = 0.5 * mont_f + 0.5 * sil_f

# ── input sediment weighted percentiles ──
mix_cdf = np.cumsum(mix_f) / np.sum(mix_f)
input_q25 = grain[np.searchsorted(mix_cdf, 0.25)]
input_med = grain[np.searchsorted(mix_cdf, 0.50)]
input_q75 = grain[np.searchsorted(mix_cdf, 0.75)]

# ── load D_p samples ──
dp = {}
for name, sub in [("LOWSHEAR","lowshear"),("MIDSHEAR","midshear"),
                   ("HIGHSHEAR","highshear"),("OM1","om1"),
                   ("OM2","om2"),("OM3","om3")]:
    dp[name] = np.load(f"{RES}/{sub}/{name}_dp_samples.npy")  # m

# ── layout ──
top_row = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
bot_row = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

c_input = "#2E4057"   # navy
c_dp    = "#FF6C0C"   # Caltech orange

fig = plt.figure(figsize=(6.5, 2.8))
outer = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.22)
gs_left  = outer[0].subgridspec(2, 3, wspace=0.35, hspace=0.55)
gs_right = outer[1].subgridspec(2, 1, hspace=0.55)
axes = np.empty((2, 4), dtype=object)
for r in range(2):
    for c in range(3):
        axes[r, c] = fig.add_subplot(gs_left[r, c])
    axes[r, 3] = fig.add_subplot(gs_right[r, 0])

# ── CDF panels (cols 0-2) ──
for row, defs in enumerate([top_row, bot_row]):
    for col, (name, _) in enumerate(defs):
        ax = axes[row, col]

        # input sediment CDF
        ax.plot(grain, mix_cdf, color=c_input, lw=1, label="Input sediment")

        # D_p posterior empirical CDF
        samples = dp[name]
        samples = samples[(samples > 0.5e-6) & (samples < 200e-6)]
        s_sorted = np.sort(samples)
        s_cdf = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        ax.plot(s_sorted, s_cdf, color=c_dp, lw=1.2, label=r"$D_p$ posterior")

        # KS statistic
        ks_stat, _ = ks_2samp(
            np.random.choice(grain, size=10000, p=mix_f), samples
        )
        ax.text(0.95, 0.05, f"KS = {ks_stat:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

        # median lines
        ax.axvline(input_med, color=c_input, ls="--", lw=1, alpha=0.5)
        med = np.median(samples)
        ax.axvline(med, color=c_dp, ls="--", lw=1, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlim(1e-7, 1e-3)
        ax.set_xticks([1e-7, 1e-5, 1e-3])
        ax.set_ylim(0, 1)

        if row == 1:
            ax.set_xlabel("Grain size (m)")
        if col == 0:
            ax.set_ylabel("Cumulative fraction")

# ── trend panels (col 3) ──
# Top: D_p vs shear
shear_x = np.array([0.02, 0.03, 0.04])
dp_sh_med = np.array([np.median(dp[n]) for n, _ in top_row])
dp_sh_q25 = np.array([np.percentile(dp[n], 25) for n, _ in top_row])
dp_sh_q75 = np.array([np.percentile(dp[n], 75) for n, _ in top_row])

ax_t = axes[0, 3]
input_yerr = np.array([[input_med - input_q25], [input_q75 - input_med]])
ax_t.errorbar(shear_x, np.full_like(shear_x, input_med),
              yerr=np.tile(input_yerr, len(shear_x)),
              fmt='o', ms=7, color=c_input, mec=c_input, mew=1.0,
              ecolor=c_input, capsize=3, capthick=1.2, lw=1.2,
              alpha=0.5, zorder=5, label="Input sediment")
ax_t.errorbar(shear_x, dp_sh_med,
              yerr=[dp_sh_med - dp_sh_q25, dp_sh_q75 - dp_sh_med],
              fmt='-o', ms=7, color=c_dp, mec=c_dp, mew=1.0,
              ecolor=c_dp, capsize=3, capthick=1.2, lw=1.8,
              alpha=0.8, zorder=10, label=r"$D_p$ posterior")
ax_t.set_xlabel("Shear Velocity (m/s)")
ax_t.set_ylabel(r"$D_p$ (m) $\times 10^{-5}$")
ax_t.set_xticks(shear_x)
ax_t.set_ylim(0, 50e-6)
ax_t.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e5:.0f}"))

# Bottom: D_p vs OM
om_x = np.array([1, 2, 3])
dp_om_med = np.array([np.median(dp[n]) for n, _ in bot_row])
dp_om_q25 = np.array([np.percentile(dp[n], 25) for n, _ in bot_row])
dp_om_q75 = np.array([np.percentile(dp[n], 75) for n, _ in bot_row])

ax_b = axes[1, 3]
ax_b.errorbar(om_x, np.full_like(om_x, input_med, dtype=float),
              yerr=np.tile(input_yerr, len(om_x)),
              fmt='o', ms=7, color=c_input, mec=c_input, mew=1.0,
              ecolor=c_input, capsize=3, capthick=1.2, lw=1.2,
              alpha=0.5, zorder=5)
ax_b.errorbar(om_x, dp_om_med,
              yerr=[dp_om_med - dp_om_q25, dp_om_q75 - dp_om_med],
              fmt='-o', ms=7, color=c_dp, mec=c_dp, mew=1.0,
              ecolor=c_dp, capsize=3, capthick=1.2, lw=1.8,
              alpha=0.8, zorder=10)
ax_b.set_xlabel("Organic Matter (%)")
ax_b.set_ylabel(r"$D_p$ (m) $\times 10^{-5}$")
ax_b.set_xticks(om_x)
ax_b.set_ylim(0, 50e-6)
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e5:.0f}"))


fig.savefig(f"{OUT}/input_vs_dp.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/input_vs_dp.svg", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/input_vs_dp.png")
