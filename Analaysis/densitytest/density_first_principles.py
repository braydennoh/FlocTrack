#!/usr/bin/env python3
"""
First-principles floc density analysis.

Three independent paths to characterise floc density, each requiring
minimal physical assumptions:

  Path 1 — Stokes force balance (assumption: low-Re drag)
      Δρ = 18 μ w_s / (g D_f²)
    Uses only measured w_s and D_f.

  Path 2 — Fractal geometry (assumption: self-similar packing)
      φ = (D_f / D_p)^{n_f - 3}
    Uses measured D_f, derived n_f^{3D}, and D_p from regression
    intersections.  Gives solid volume fraction, independent of any
    force balance or mineral-density value.

  Path 3 — Consistency cross-check (zero extra assumptions)
      ρ_s,eff = ρ_w + Δρ_Stokes / φ
    Should recover ~2650 kg/m³ (kaolinite) if both paths are
    internally consistent.  Deviations flag non-Stokes effects,
    non-ideal packing, or organic coating.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"

DATA = "/Users/braydennoh/Research/Floc/experimentaldata"
RES  = "/Users/braydennoh/Research/Floc/results"
OUT  = f"{RES}/densitytest"

# ── physical constants (only fluid properties — no mineral assumptions) ──
rho_w = 1000.0      # kg/m³
mu    = 1.0e-3       # Pa·s
nu    = mu / rho_w   # m²/s
g     = 9.81         # m/s²

# mineral density — used ONLY in the cross-check, not in Paths 1 or 2
rho_s_kaolinite = 2650.0  # kg/m³

# ── dataset definitions ──
shear_defs = [("LOWSHEAR", 0.02), ("MIDSHEAR", 0.03), ("HIGHSHEAR", 0.04)]
om_defs    = [("OM1", 1), ("OM2", 2), ("OM3", 3)]

# 2D→3D regressions (from MCMC analysis)
regressions = {
    "LOWSHEAR":  (1.421, -0.391),
    "MIDSHEAR":  (1.069,  0.315),
    "HIGHSHEAR": (0.936,  0.730),
    "OM1":       (1.433, -0.332),
    "OM2":       (0.661,  0.685),
    "OM3":       (0.904,  0.347),
}

# D_p median from intersection analysis (loaded from saved samples)
dp_medians = {}
for name, subdir in [("LOWSHEAR", "lowshear"), ("MIDSHEAR", "midshear"),
                      ("HIGHSHEAR", "highshear"), ("OM1", "om1"),
                      ("OM2", "om2"), ("OM3", "om3")]:
    try:
        arr = np.load(f"{RES}/{subdir}/{name}_dp_samples.npy")
        dp_medians[name] = np.median(arr)  # metres
    except FileNotFoundError:
        print(f"  Warning: no D_p samples for {name}")
        dp_medians[name] = None


def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df.dropna(subset=["Avg_Concave_Diameter", "Avg_Velocity",
                            "Avg_Perimeter_Fractal_Dimension"])
    df = df[(df["Avg_Concave_Diameter"] > 0) & (df["Avg_Velocity"] > 0)]

    # SI units
    Df = df["Avg_Concave_Diameter"].values * 1e-6   # m
    ws = df["Avg_Velocity"].values * 1e-6            # m/s

    # Per-particle n_f^{3D} from regression
    m_reg, b_reg = regressions[name]
    nf2d = df["Avg_Perimeter_Fractal_Dimension"].values
    nf3d = m_reg * nf2d + b_reg

    # ── Path 1: Stokes excess density ──
    # Δρ = 18 μ w_s / (g D_f²)
    delta_rho = 18.0 * mu * ws / (g * Df**2)

    # ── Path 2: Fractal solid fraction ──
    Dp = dp_medians[name]
    if Dp is not None and Dp > 0:
        phi = (Df / Dp) ** (nf3d - 3.0)
    else:
        phi = np.full_like(Df, np.nan)

    # ── Path 3: Effective mineral density ──
    # ρ_s,eff = ρ_w + Δρ / φ
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_s_eff = rho_w + delta_rho / phi

    # Filter: keep physically plausible values
    good = (delta_rho > 0) & (delta_rho < 2000) & np.isfinite(phi) & (phi > 0) & (phi < 1)
    good &= np.isfinite(rho_s_eff) & (rho_s_eff > 500) & (rho_s_eff < 10000)

    return {
        "Df": Df[good] * 1e6,       # µm for display
        "ws": ws[good] * 1e3,        # mm/s
        "nf3d": nf3d[good],
        "delta_rho": delta_rho[good],
        "phi": phi[good],
        "rho_s_eff": rho_s_eff[good],
        "rho_floc": rho_w + delta_rho[good],
        "N_total": len(Df),
        "N_good": int(np.sum(good)),
        "Dp_um": Dp * 1e6 if Dp else None,
    }


all_data = {}
for name, _ in shear_defs + om_defs:
    all_data[name] = load(name)


def pct(arr, qs=(25, 50, 75)):
    return [np.percentile(arr, q) for q in qs]


# ═══════════════════════════════════════════════════════════════
# Print summary
# ═══════════════════════════════════════════════════════════════
print("=" * 100)
print("First-Principles Floc Density Analysis")
print("=" * 100)
print(f"\n{'Dataset':<12} {'N':>5}  {'D_p(µm)':>8}  "
      f"{'Δρ_Stokes':>12}  {'φ':>10}  {'ρ_s,eff':>12}  {'ρ_floc':>10}")
print(f"{'':12} {'':>5}  {'':>8}  "
      f"{'med [IQR]':>12}  {'med [IQR]':>10}  {'med [IQR]':>12}  {'med [IQR]':>10}")
print("-" * 100)
for name, _ in shear_defs + om_defs:
    d = all_data[name]
    dr_q = pct(d["delta_rho"])
    phi_q = pct(d["phi"])
    rs_q = pct(d["rho_s_eff"])
    rf_q = pct(d["rho_floc"])
    dp_str = f"{d['Dp_um']:.1f}" if d['Dp_um'] else "—"
    print(f"{name:<12} {d['N_good']:>5}  {dp_str:>8}  "
          f"{dr_q[1]:6.0f} [{dr_q[0]:.0f},{dr_q[2]:.0f}]  "
          f"{phi_q[1]:5.3f} [{phi_q[0]:.3f},{phi_q[2]:.3f}]  "
          f"{rs_q[1]:6.0f} [{rs_q[0]:.0f},{rs_q[2]:.0f}]  "
          f"{rf_q[1]:6.0f} [{rf_q[0]:.0f},{rf_q[2]:.0f}]")
print("=" * 100)
print("\nPath 1: Δρ = 18μw_s/(gD_f²)  — Stokes only, no structural model")
print("Path 2: φ = (D_f/D_p)^(n_f-3) — fractal geometry only, no force balance")
print("Path 3: ρ_s,eff = ρ_w + Δρ/φ  — should ≈ 2650 kg/m³ (kaolinite) if consistent")


# ═══════════════════════════════════════════════════════════════
# Helper for median + IQR arrays
# ═══════════════════════════════════════════════════════════════
def get_medians(defs, key):
    meds, lo, hi = [], [], []
    for name, _ in defs:
        arr = all_data[name][key]
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        meds.append(med)
        lo.append(med - q25)
        hi.append(q75 - med)
    return np.array(meds), np.array(lo), np.array(hi)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: 3-row × 2-col — the three independent paths
# ═══════════════════════════════════════════════════════════════
c_stokes  = "#2E4057"   # navy — Stokes
c_phi     = "#7BC8A4"   # green — solid fraction
c_check   = "#E8575A"   # red — consistency check
c_ref     = "#888888"   # grey

fig, axes = plt.subplots(3, 2, figsize=(5.6, 6.4), constrained_layout=True)

shear_x = np.array([x for _, x in shear_defs])
om_x    = np.array([x for _, x in om_defs])

def plot_row(row, key, color, ylabel, ref_line=None, ref_label=None):
    for col, (defs, x_arr, xlabel) in enumerate([
        (shear_defs, shear_x, "Shear Velocity (m/s)"),
        (om_defs,    om_x,    "Organic Matter (%)")]):

        ax = axes[row, col]
        meds, lo, hi = get_medians(defs, key)

        ax.errorbar(x_arr, meds, yerr=[lo, hi],
                    fmt='-o', ms=7, color=color, mec='white', mew=1.0,
                    ecolor=color, capsize=4, capthick=1.2, lw=1.5, zorder=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x_arr)

        if ref_line is not None:
            ax.axhline(ref_line, color=c_ref, ls='--', lw=1, alpha=0.6,
                       zorder=1, label=ref_label)
            ax.legend(fontsize=6, frameon=False)

        if row == 2:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])

# Row 0: Δρ from Stokes (Path 1)
plot_row(0, "delta_rho", c_stokes, r"$\Delta\rho$ (kg/m³)" + "\n[Stokes]")
axes[0, 0].set_title("(a)", fontsize=11, loc="left", fontweight="bold")
axes[0, 1].set_title("(b)", fontsize=11, loc="left", fontweight="bold")

# Row 1: φ from fractal geometry (Path 2)
plot_row(1, "phi", c_phi, r"$\varphi$ (solid fraction)" + "\n[Fractal geometry]")
axes[1, 0].set_title("(c)", fontsize=11, loc="left", fontweight="bold")
axes[1, 1].set_title("(d)", fontsize=11, loc="left", fontweight="bold")

# Row 2: ρ_s,eff consistency check (Path 3)
plot_row(2, "rho_s_eff", c_check,
         r"$\rho_{s,\mathrm{eff}}$ (kg/m³)" + "\n[Cross-check]",
         ref_line=rho_s_kaolinite, ref_label="Kaolinite (2650)")
axes[2, 0].set_title("(e)", fontsize=11, loc="left", fontweight="bold")
axes[2, 1].set_title("(f)", fontsize=11, loc="left", fontweight="bold")

fig.savefig(f"{OUT}/density_3paths.png", dpi=300)
fig.savefig(f"{OUT}/density_3paths.svg")
plt.close(fig)
print(f"\nSaved → {OUT}/density_3paths.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Δρ_Stokes vs Δρ_fractal scatter (direct comparison)
# ═══════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(5.6, 2.8), constrained_layout=True)

for col, (defs, xlabel_grp) in enumerate([
    (shear_defs, "Shear sweep"), (om_defs, "OM sweep")]):

    ax = axes2[col]
    for name, val in defs:
        d = all_data[name]
        drho_stokes = d["delta_rho"]
        drho_fractal = (rho_s_kaolinite - rho_w) * d["phi"]

        ax.scatter(drho_fractal, drho_stokes, s=1, alpha=0.15,
                   label=name, zorder=5)

    lim_max = ax.get_xlim()[1]
    ax.plot([0, lim_max], [0, lim_max], 'k--', lw=0.8, alpha=0.5,
            label="1:1 line")
    ax.set_xlabel(r"$\Delta\rho_{\mathrm{fractal}}$ = $(\rho_s - \rho_w)\varphi$  (kg/m³)")
    ax.set_ylabel(r"$\Delta\rho_{\mathrm{Stokes}}$ = $18\mu w_s / (g D_f^2)$  (kg/m³)")
    ax.legend(fontsize=5.5, frameon=False, markerscale=4)
    ax.set_title(f"({'a' if col == 0 else 'b'})  {xlabel_grp}",
                 fontsize=10, loc="left", fontweight="bold")

fig2.savefig(f"{OUT}/density_stokes_vs_fractal.png", dpi=300)
fig2.savefig(f"{OUT}/density_stokes_vs_fractal.svg")
plt.close(fig2)
print(f"Saved → {OUT}/density_stokes_vs_fractal.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: ρ_floc (absolute density) — simplest summary
# ═══════════════════════════════════════════════════════════════
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(5.6, 2.6),
                                   constrained_layout=True)

for ax, defs, x_arr, xlabel in [
    (ax3a, shear_defs, shear_x, "Shear Velocity (m/s)"),
    (ax3b, om_defs,    om_x,    "Organic Matter (%)")]:

    meds, lo, hi = get_medians(defs, "rho_floc")
    ax.errorbar(x_arr, meds, yerr=[lo, hi],
                fmt='-o', ms=8, color=c_stokes, mec='white', mew=1.2,
                ecolor=c_stokes, capsize=5, capthick=1.3, lw=1.8, zorder=10)
    ax.axhline(rho_w, color=c_ref, ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\rho_{\mathrm{floc}}$ (kg/m³)")
    ax.set_xticks(x_arr)

ax3a.set_title("(a)", fontsize=11, loc="left", fontweight="bold")
ax3b.set_title("(b)", fontsize=11, loc="left", fontweight="bold")

fig3.savefig(f"{OUT}/density_rhofloc.png", dpi=300)
fig3.savefig(f"{OUT}/density_rhofloc.svg")
plt.close(fig3)
print(f"Saved → {OUT}/density_rhofloc.png")


# ═══════════════════════════════════════════════════════════════
# Write detailed report
# ═══════════════════════════════════════════════════════════════
with open(f"{OUT}/density_report.txt", "w") as f:
    f.write("First-Principles Floc Density Analysis\n")
    f.write("=" * 70 + "\n\n")

    f.write("METHODOLOGY\n")
    f.write("-" * 70 + "\n")
    f.write("Path 1 (Stokes): Δρ = 18μw_s / (gD_f²)\n")
    f.write("  Assumptions: Stokes drag (Re << 1)\n")
    f.write("  Uses: w_s (measured), D_f (measured)\n\n")
    f.write("Path 2 (Fractal): φ = (D_f / D_p)^{n_f - 3}\n")
    f.write("  Assumptions: Self-similar fractal packing\n")
    f.write("  Uses: D_f (measured), D_p (from intersections), n_f^3D (from MCMC)\n\n")
    f.write("Path 3 (Cross-check): ρ_s,eff = ρ_w + Δρ / φ\n")
    f.write("  Assumptions: None beyond Paths 1 + 2\n")
    f.write("  Expected: ≈ 2650 kg/m³ (kaolinite) if consistent\n\n")

    f.write("RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Dataset':<12} {'N':>5}  {'D_p':>6}  "
            f"{'Δρ med':>7}  {'φ med':>7}  {'ρ_s,eff':>8}  {'ρ_floc':>8}\n")
    f.write(f"{'':12} {'':>5}  {'(µm)':>6}  "
            f"{'(kg/m³)':>7}  {'':>7}  {'(kg/m³)':>8}  {'(kg/m³)':>8}\n")
    for name, _ in shear_defs + om_defs:
        d = all_data[name]
        dp_str = f"{d['Dp_um']:.1f}" if d['Dp_um'] else "—"
        f.write(f"{name:<12} {d['N_good']:>5}  {dp_str:>6}  "
                f"{np.median(d['delta_rho']):>7.0f}  "
                f"{np.median(d['phi']):>7.4f}  "
                f"{np.median(d['rho_s_eff']):>8.0f}  "
                f"{np.median(d['rho_floc']):>8.0f}\n")

    f.write(f"\n\nKEY FINDINGS\n")
    f.write("-" * 70 + "\n")

    # Shear trend
    drho_shear = [np.median(all_data[n]["delta_rho"]) for n, _ in shear_defs]
    phi_shear = [np.median(all_data[n]["phi"]) for n, _ in shear_defs]
    f.write(f"\nShear sweep (u* = 0.02 → 0.04 m/s):\n")
    f.write(f"  Δρ: {drho_shear[0]:.0f} → {drho_shear[-1]:.0f} kg/m³\n")
    f.write(f"  φ:  {phi_shear[0]:.4f} → {phi_shear[-1]:.4f}\n")

    # OM trend
    drho_om = [np.median(all_data[n]["delta_rho"]) for n, _ in om_defs]
    phi_om = [np.median(all_data[n]["phi"]) for n, _ in om_defs]
    f.write(f"\nOM sweep (1% → 3%):\n")
    f.write(f"  Δρ: {drho_om[0]:.0f} → {drho_om[-1]:.0f} kg/m³\n")
    f.write(f"  φ:  {phi_om[0]:.4f} → {phi_om[-1]:.4f}\n")

    # Consistency
    rs_all = np.concatenate([all_data[n]["rho_s_eff"] for n, _ in shear_defs + om_defs])
    f.write(f"\nConsistency check (ρ_s,eff across all datasets):\n")
    f.write(f"  Overall median = {np.median(rs_all):.0f} kg/m³\n")
    f.write(f"  IQR = [{np.percentile(rs_all, 25):.0f}, {np.percentile(rs_all, 75):.0f}]\n")
    f.write(f"  Expected (kaolinite) = 2650 kg/m³\n")

print(f"Saved → {OUT}/density_report.txt")
