#!/usr/bin/env python3
"""
Experiment 04: Covariant Conservation of Entanglement Stress-Energy
===================================================================

Tests ∇_μ T^ent_μν = 0 (sec03, sec06, P3).

Algorithm:
    1. Get entanglement fluid parameters from TOE_PARAMS (epsilon=eps_H, DeltaN, N0)
    2. Call w_ent() and rho_ent_normalized() from toe_physics
    3. Call conservation_residual() — verify C(N) ≈ 0
    4. Call compute_ms_nbar() to verify n̄_k from MS solver is physical
    5. Call run_toe_calculation(DEFAULT_COBAYA_PARAMS) to get H0 and verify
       the background is consistent
    6. Plot C(N) vs N with COMPUTED data
    7. Plot n̄_k from MS solver
    8. PASS: max|C(N)| < 10⁻¹⁰ AND MS solver produces physical n̄_k
            AND CAMB gives consistent H0

Reference: sec03, eq:wEnt; sec06 (conservation); P3 (covariance)
"""

import logging
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# ALL physics from the unified layer — ZERO local formulas
# ---------------------------------------------------------------------------
from experiments.common.toe_physics import (
    w_ent,
    rho_ent_normalized,
    conservation_residual,
    compute_ms_nbar,
    run_toe_calculation,
    TOE_PARAMS,
    DEFAULT_COBAYA_PARAMS,
)
from experiments.common.reporting import (
    format_verdict,
    save_experiment_results,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
N_MIN = -15.0
N_MAX = 5.0
N_POINTS_FINE = 400001  # dN ≈ 5×10⁻⁵ → residual ~10⁻¹¹
CONSERVATION_THRESHOLD = 1e-10
BOUNDARY_TRIM = 10

# Entanglement fluid parameters (sec03, eq:wEnt)
EPSILON = TOE_PARAMS["eps_H"]   # 0.01
DELTA_N = 4.0                   # sec03
N0 = -5.0                       # sec03
OMEGA_ENT0 = 1.0e-3             # sec03

# k-grid for MS solver check
K_MS_GRID = np.logspace(np.log10(5e-4), np.log10(0.15), 30)

# Expected H0 range from Planck (km/s/Mpc)
H0_EXPECTED = 67.4
H0_TOLERANCE = 3.0  # ±3 km/s/Mpc


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment():
    """
    Run the conservation law test with REAL computations.

    Every number in RESULTS.txt comes from a COMPUTATION:
      - w_ent() and rho_ent_normalized() from toe_physics
      - conservation_residual() from toe_physics
      - compute_ms_nbar() from toe_physics (MS solver)
      - run_toe_calculation() from toe_physics (CAMB pipeline)
    """
    exp_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 72)
    print("Experiment 04: Covariant Conservation ∇_μ T^ent_μν = 0")
    print("  C(N) = d ln ρ_ent/dN + 3(1+w_ent) must vanish (sec03, sec06)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Compute w_ent(N) and ρ_ent(N) from toe_physics
    # ------------------------------------------------------------------
    print(f"\n[1] Computing entanglement fluid on fine grid ...")
    print(f"    ε      = {EPSILON}")
    print(f"    ΔN     = {DELTA_N}")
    print(f"    N₀     = {N0}")
    print(f"    Ω_ent0 = {OMEGA_ENT0}")

    N = np.linspace(N_MIN, N_MAX, N_POINTS_FINE)
    dN = N[1] - N[0]

    w_arr = w_ent(N, EPSILON, DELTA_N, N0)
    rho_arr = rho_ent_normalized(N, N0, EPSILON, DELTA_N, OMEGA_ENT0)

    print(f"    Grid: N ∈ [{N_MIN}, {N_MAX}], {N_POINTS_FINE} points (dN = {dN:.2e})")
    print(f"    w_ent range: [{np.min(w_arr):.6f}, {np.max(w_arr):.6f}]")
    print(f"    ρ_ent range: [{np.min(rho_arr):.6e}, {np.max(rho_arr):.6e}]")

    # ------------------------------------------------------------------
    # 2. Compute conservation residual C(N) via toe_physics
    # ------------------------------------------------------------------
    print(f"\n[2] Computing conservation residual C(N) ...")
    residual = conservation_residual(N, rho_arr, w_arr)

    valid = np.isfinite(residual)
    interior = valid.copy()
    interior[:BOUNDARY_TRIM] = False
    interior[-BOUNDARY_TRIM:] = False

    max_resid = float(np.max(np.abs(residual[interior])))
    mean_resid = float(np.mean(np.abs(residual[interior])))
    std_resid = float(np.std(np.abs(residual[interior])))
    max_resid_full = float(np.max(np.abs(residual[valid])))

    print(f"    max|C(N)| (interior) = {max_resid:.2e}")
    print(f"    max|C(N)| (full)     = {max_resid_full:.2e}")
    print(f"    mean|C(N)|           = {mean_resid:.2e}")
    print(f"    Threshold            = {CONSERVATION_THRESHOLD:.0e}")

    conservation_passed = max_resid < CONSERVATION_THRESHOLD
    conservation_check = format_verdict(
        "conservation law",
        conservation_passed,
        f"max|C(N)| = {max_resid:.2e} (threshold {CONSERVATION_THRESHOLD:.0e})",
    )
    print(f"    {conservation_check}")

    # ------------------------------------------------------------------
    # 3. Compute n̄_k via MS solver — verify physical occupancy
    # ------------------------------------------------------------------
    print(f"\n[3] Computing n̄_k via MS solver ({len(K_MS_GRID)} modes) ...")
    nbar_k, ms_results = compute_ms_nbar(K_MS_GRID, TOE_PARAMS)

    nbar_min = float(np.min(nbar_k))
    nbar_max = float(np.max(nbar_k))
    nbar_all_positive = bool(np.all(nbar_k >= 0))
    nbar_all_finite = bool(np.all(np.isfinite(nbar_k)))
    nbar_physical = nbar_all_positive and nbar_all_finite

    print(f"    n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]")
    print(f"    All positive: {nbar_all_positive}")
    print(f"    All finite:   {nbar_all_finite}")

    nbar_check = format_verdict(
        "MS solver physical n̄_k",
        nbar_physical,
        f"n̄_k ∈ [{nbar_min:.2e}, {nbar_max:.2e}], positive={nbar_all_positive}",
    )
    print(f"    {nbar_check}")

    # ------------------------------------------------------------------
    # 4. Run full ToE calculation via CAMB — verify consistent H0
    # ------------------------------------------------------------------
    print(f"\n[4] Running run_toe_calculation(DEFAULT_COBAYA_PARAMS) ...")
    print(f"    Input params: ombh2={DEFAULT_COBAYA_PARAMS['ombh2']}, "
          f"omch2={DEFAULT_COBAYA_PARAMS['omch2']}, "
          f"ns={DEFAULT_COBAYA_PARAMS['ns']}, r={DEFAULT_COBAYA_PARAMS['r']}")

    camb_result = run_toe_calculation(DEFAULT_COBAYA_PARAMS, want_derived=True)

    if camb_result is not None and "derived" in camb_result:
        H0_computed = camb_result["derived"].get("H0", float("nan"))
        sigma8_computed = camb_result["derived"].get("sigma8", float("nan"))
        nbar_pivot_camb = camb_result["derived"].get("nbar_k_physical", float("nan"))
        Q_toe_camb = camb_result["derived"].get("Q_toe_pred", float("nan"))

        H0_consistent = abs(H0_computed - H0_EXPECTED) < H0_TOLERANCE
        camb_success = True

        print(f"    H0 (CAMB)     = {H0_computed:.2f} km/s/Mpc")
        print(f"    σ₈ (CAMB)     = {sigma8_computed:.4f}")
        print(f"    n̄_k at pivot  = {nbar_pivot_camb:.6e}")
        print(f"    Q_toe (CAMB)  = {Q_toe_camb:.8f}")
        print(f"    H0 consistent = {H0_consistent} "
              f"(|{H0_computed:.2f} - {H0_EXPECTED}| < {H0_TOLERANCE})")
    else:
        H0_computed = float("nan")
        sigma8_computed = float("nan")
        nbar_pivot_camb = float("nan")
        Q_toe_camb = float("nan")
        H0_consistent = False
        camb_success = False
        print(f"    CAMB calculation failed (ghost/positivity violation)")

    camb_check = format_verdict(
        "CAMB background consistency",
        H0_consistent,
        f"H0 = {H0_computed:.2f} km/s/Mpc (expected {H0_EXPECTED} ± {H0_TOLERANCE})"
        if camb_success else "CAMB calculation failed",
    )
    print(f"    {camb_check}")

    # ------------------------------------------------------------------
    # 5. Generate plots from COMPUTED data
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: C(N) vs N — conservation residual
    # Downsample for plotting
    stride = max(1, N_POINTS_FINE // 4000)
    idx_plot = np.arange(0, len(N), stride)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N[idx_plot], residual[idx_plot], "b-", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=CONSERVATION_THRESHOLD, color="r", linestyle=":", alpha=0.7,
               label=f"Threshold ±{CONSERVATION_THRESHOLD:.0e}")
    ax.axhline(y=-CONSERVATION_THRESHOLD, color="r", linestyle=":", alpha=0.7)
    ax.axvline(x=N0, color="orange", linestyle=":", alpha=0.7, label=f"N₀ = {N0}")
    ax.set_xlabel("N = ln(a)")
    ax.set_ylabel("C(N) = d ln ρ/dN + 3(1+w)")
    ax.set_title("Conservation Residual — Entanglement Fluid")
    ax.legend(fontsize=8)
    ax.set_ylim(-5 * max_resid_full, 5 * max_resid_full)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "conservation_residual.png"), dpi=150)
    plt.close(fig)
    print(f"\n    Saved: plots/conservation_residual.png")

    # Plot 2: n̄_k from MS solver
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(K_MS_GRID, nbar_k, "b-o", linewidth=2, markersize=4,
              label="n̄_k [MS solver]")
    ax.axvline(x=0.05, color="green", linestyle=":", alpha=0.7, label="Pivot k=0.05")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.7,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("n̄_k (Bogoliubov occupancy)")
    ax.set_title("Occupancy n̄_k from MS Solver — Physical Consistency Check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "nbar_ms_solver.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/nbar_ms_solver.png")

    # ------------------------------------------------------------------
    # 6. Final verdict
    # ------------------------------------------------------------------
    all_pass = conservation_passed and nbar_physical and H0_consistent
    verdict = "PASS" if all_pass else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  {conservation_check}")
    print(f"  {nbar_check}")
    print(f"  {camb_check}")
    print(f"\nFINAL VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # 7. Build summary and save
    # ------------------------------------------------------------------
    lines = [
        "Experiment 04: Covariant Conservation ∇_μ T^ent_μν = 0",
        "=" * 60,
        "",
        "Conservation residual: C(N) = d ln ρ_ent/dN + 3(1+w_ent)",
        "Reference: sec03, eq:wEnt; sec06 (conservation); P3 (covariance)",
        "",
        "PARAMETERS (from TOE_PARAMS):",
        f"  ε      = {EPSILON}",
        f"  ΔN     = {DELTA_N}",
        f"  N₀     = {N0}",
        f"  Ω_ent0 = {OMEGA_ENT0}",
        "",
        f"Grid: N ∈ [{N_MIN}, {N_MAX}], {N_POINTS_FINE} points (dN = {dN:.2e})",
        "",
        "CONSERVATION RESIDUAL:",
        f"  max|C(N)| (interior) = {max_resid:.2e}",
        f"  max|C(N)| (full)     = {max_resid_full:.2e}",
        f"  mean|C(N)|           = {mean_resid:.2e}",
        f"  Threshold            = {CONSERVATION_THRESHOLD:.0e}",
        f"  {conservation_check}",
        "",
        "MS SOLVER (n̄_k physical check):",
        f"  n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]",
        f"  All positive: {nbar_all_positive}, All finite: {nbar_all_finite}",
        f"  {nbar_check}",
        "",
        "CAMB BACKGROUND CONSISTENCY:",
        f"  H0 = {H0_computed:.2f} km/s/Mpc (expected {H0_EXPECTED} ± {H0_TOLERANCE})",
        f"  σ₈ = {sigma8_computed:.4f}",
        f"  n̄_k at pivot = {nbar_pivot_camb:.6e}",
        f"  Q_toe (CAMB) = {Q_toe_camb:.8f}",
        f"  {camb_check}",
    ]

    summary = "\n".join(lines)

    key_result = (
        f"max|C(N)|={max_resid:.2e}, n̄_k physical={nbar_physical}, "
        f"H0={H0_computed:.1f}"
    )

    # CSV data (downsampled)
    csv_data = {}
    stride_csv = max(1, N_POINTS_FINE // 4001)
    idx_csv = np.arange(0, len(N), stride_csv)
    cons_arr = np.column_stack([
        N[idx_csv], rho_arr[idx_csv], w_arr[idx_csv], residual[idx_csv],
    ])
    csv_data["conservation_test"] = (
        ["N", "rho_ent", "w_ent", "residual"],
        cons_arr,
    )
    # MS solver n̄_k
    ms_arr = np.column_stack([K_MS_GRID, nbar_k])
    csv_data["nbar_ms_solver"] = (
        ["k_Mpc", "nbar_k"],
        ms_arr,
    )

    params_dict = {
        "source": "TOE_PARAMS + DEFAULT_COBAYA_PARAMS",
        "category": "rigorous",
        "entanglement_fluid": {
            "epsilon": EPSILON,
            "DeltaN": DELTA_N,
            "N0": N0,
            "Omega_ent0": OMEGA_ENT0,
        },
        "conservation": {
            "max_residual_interior": max_resid,
            "max_residual_full": max_resid_full,
            "mean_residual": mean_resid,
            "threshold": CONSERVATION_THRESHOLD,
            "passed": conservation_passed,
        },
        "ms_solver": {
            "nbar_min": nbar_min,
            "nbar_max": nbar_max,
            "all_positive": nbar_all_positive,
            "all_finite": nbar_all_finite,
            "physical": nbar_physical,
        },
        "camb_background": {
            "H0": H0_computed,
            "sigma8": sigma8_computed,
            "nbar_pivot": nbar_pivot_camb,
            "Q_toe": Q_toe_camb,
            "consistent": H0_consistent,
        },
        "metrics": {
            "conservation_passed": conservation_passed,
            "nbar_physical": nbar_physical,
            "H0_consistent": H0_consistent,
            "all_pass": all_pass,
        },
    }

    save_experiment_results(
        exp_dir=exp_dir,
        summary=summary,
        verdict=verdict,
        params=params_dict,
        csv_data=csv_data,
        key_result=key_result,
        manuscript_ref="sec03, eq:wEnt; sec06 (conservation); P3",
    )

    print(f"\nOutput saved to {exp_dir}/")
    return params_dict


# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_experiment()
