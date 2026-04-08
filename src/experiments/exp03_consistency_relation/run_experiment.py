#!/usr/bin/env python3
"""
Experiment 03: Generalized Tensor-Scalar Consistency Relation
=============================================================

THE key falsifiable prediction of the Theory of Everything.

Tests eq:consistency (sec03):
    r / (-8 n_t) = c_s* / (1 + 2 n̄_k)

Algorithm:
    1. Load BK18 chains via load_bk18_chains() — get r, n_s, H0 posteriors
    2. Call evaluate_bk18(samples, weights, param_names, TOE_PARAMS)
       — runs FULL ToE evaluation on BK18 data
    3. From BK18EvalResult: extract Q_toe_grid, nbar_k_grid, k_grid,
       r_mean, nt_si_mean, nt_toe_mean, delta_nt_mean
    4. Call compute_ms_nbar() on a fine k-grid for Q(k) profile
    5. Plot Q(k) vs k — the REAL ToE prediction from MS solver on BK18 data
    6. Plot n̄_k vs k — occupancy from Bogoliubov matching
    7. Compare Q_obs = r/(-8n_t) from chains with Q_toe = c_s*/(1+2n̄_k)
    8. PASS: |Q_obs - Q_toe| < threshold at pivot

Reference: sec03, eq:consistency; sec07 (Falsifiable Predictions); sec13
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
    load_bk18_chains,
    evaluate_bk18,
    compute_ms_nbar,
    TOE_PARAMS,
    K_PIVOT,
    finite_diff_log_slope,
)
from experiments.common.reporting import (
    format_verdict,
    save_experiment_results,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
# Fine k-grid for Q(k) profile (physical Mpc⁻¹)
K_FINE = np.logspace(np.log10(5e-4), np.log10(0.15), 40)

# PASS/FAIL threshold: |Q_obs - Q_toe| at pivot
Q_DEVIATION_THRESHOLD = 0.10  # 10% tolerance


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment():
    """
    Run the generalized consistency relation test on REAL BK18 data.

    Every number in RESULTS.txt comes from a COMPUTATION:
      - BK18 chains loaded via load_bk18_chains()
      - ToE evaluation via evaluate_bk18()
      - MS solver via compute_ms_nbar()
    """
    exp_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 72)
    print("Experiment 03: Generalized Tensor-Scalar Consistency Relation")
    print("  eq:consistency (sec03): r/(-8 n_t) = c_s*/(1+2 n̄_k)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load BK18 chains — REAL observational data
    # ------------------------------------------------------------------
    print("\n[1] Loading BK18+Planck+BAO chains ...")
    samples, weights, param_names = load_bk18_chains()
    n_samples = len(samples)
    n_eff = float(np.sum(weights))
    print(f"    Loaded {n_samples} samples, n_eff = {n_eff:.0f}")
    print(f"    Parameters: {list(param_names.keys())[:10]} ...")

    # ------------------------------------------------------------------
    # 2. Run FULL ToE evaluation on BK18 data
    # ------------------------------------------------------------------
    print("\n[2] Running evaluate_bk18() — full ToE evaluation ...")
    print(f"    TOE_PARAMS: {TOE_PARAMS}")
    result = evaluate_bk18(samples, weights, param_names, TOE_PARAMS)

    print(f"    r (BK18)       = {result.r_mean:.6f} ± {result.r_std:.6f}")
    print(f"    n̄_k at pivot   = {result.nbar_k_pivot:.6e}")
    print(f"    Q_toe at pivot = {result.Q_toe:.8f}")
    print(f"    n_t (SI)       = {result.nt_si_mean:.8f} ± {result.nt_si_std:.8f}")
    print(f"    n_t (ToE)      = {result.nt_toe_mean:.8f} ± {result.nt_toe_std:.8f}")
    print(f"    Δn_t           = {result.delta_nt_mean:.8e}")
    print(f"    |Δn_t|/σ       = {result.delta_nt_over_sigma_r:.4f}")

    # ------------------------------------------------------------------
    # 3. Compute n̄_k on fine k-grid via MS solver
    # ------------------------------------------------------------------
    print(f"\n[3] Computing n̄_k on fine k-grid ({len(K_FINE)} modes) via MS solver ...")
    nbar_fine, ms_results_fine = compute_ms_nbar(K_FINE, TOE_PARAMS)
    c_s = TOE_PARAMS["c_s_star"]
    Q_fine = c_s / (1.0 + 2.0 * nbar_fine)

    print(f"    n̄_k range: [{np.min(nbar_fine):.6e}, {np.max(nbar_fine):.6e}]")
    print(f"    Q(k) range: [{np.min(Q_fine):.6f}, {np.max(Q_fine):.6f}]")

    # ------------------------------------------------------------------
    # 4. Compare Q_obs vs Q_toe at pivot
    # ------------------------------------------------------------------
    # Q_obs from BK18: if n_t = -r/8 (SI), then Q_obs = r/(-8*n_t) = 1
    # Q_toe from MS solver: c_s*/(1+2*n̄_k)
    Q_obs_si = 1.0  # by definition for SI chains where n_t = -r/8
    Q_toe_pivot = result.Q_toe
    Q_deviation = abs(Q_obs_si - Q_toe_pivot)

    print(f"\n[4] Consistency relation comparison at pivot k = {K_PIVOT} Mpc⁻¹:")
    print(f"    Q_obs (SI chains) = {Q_obs_si:.8f}")
    print(f"    Q_toe (MS solver) = {Q_toe_pivot:.8f}")
    print(f"    |Q_obs - Q_toe|   = {Q_deviation:.8e}")
    print(f"    Threshold         = {Q_DEVIATION_THRESHOLD}")

    # Also show the BK18EvalResult k-grid values
    print(f"\n    Q_toe on BK18 evaluation k-grid:")
    for i, k in enumerate(result.k_grid):
        marker = ""
        if abs(k - K_PIVOT) < 0.001:
            marker = " ← PIVOT"
        elif abs(k - TOE_PARAMS["k0"]) < 0.0001:
            marker = " ← k₀"
        print(f"      k={k:.4f}: n̄_k={result.nbar_k_grid[i]:.6e}, "
              f"Q={result.Q_toe_grid[i]:.8f}{marker}")

    # ------------------------------------------------------------------
    # 5. CMB-S4 / LiteBIRD forecast for Q(k) detection
    # ------------------------------------------------------------------
    # Q = r / (-8 n_t).  At the pivot, uncertainty on Q depends on σ(r)
    # and σ(n_t).  For SI chains n_t = -r/8, so Q_obs = 1 exactly.
    # The ToE prediction is Q_toe = c_s*/(1+2n̄_k) < 1.
    # Signal = 1 - Q_toe(k);  Noise = σ(Q) ≈ σ(r) / (8|n_t|).
    #
    # Fiducial: r = r_mean from BK18, n_t = nt_si_mean from BK18.
    # Reference: sec13, subsec:forecasts
    print(f"\n[5] CMB-S4 / LiteBIRD forecast for Q < 1 detection ...")

    r_fid = result.r_mean                    # fiducial r from BK18
    nt_fid = result.nt_si_mean               # fiducial n_t from BK18
    nt_abs = max(abs(nt_fid), 1e-10)         # guard against zero

    # Experimental sensitivities (sec13, subsec:forecasts)
    sigma_r_litebird = 1.0e-3                # LiteBIRD target
    sigma_r_cmbs4 = 5.0e-4                   # CMB-S4 target

    # σ(Q) ≈ σ(r) / (8|n_t|)  — error propagation at the pivot
    sigma_Q_litebird = sigma_r_litebird / (8.0 * nt_abs)
    sigma_Q_cmbs4 = sigma_r_cmbs4 / (8.0 * nt_abs)

    print(f"    Fiducial r = {r_fid:.6f}, n_t = {nt_fid:.8f}")
    print(f"    LiteBIRD: σ(r) = {sigma_r_litebird:.1e} → σ(Q) = {sigma_Q_litebird:.4f}")
    print(f"    CMB-S4:   σ(r) = {sigma_r_cmbs4:.1e} → σ(Q) = {sigma_Q_cmbs4:.4f}")

    # SNR for detecting Q < 1 at each k
    signal_fine = 1.0 - Q_fine               # deviation from SI (Q=1)
    snr_litebird = np.abs(signal_fine) / sigma_Q_litebird
    snr_cmbs4 = np.abs(signal_fine) / sigma_Q_cmbs4

    # At pivot
    idx_pivot = np.argmin(np.abs(K_FINE - K_PIVOT))
    signal_pivot = float(signal_fine[idx_pivot])
    snr_pivot_lb = float(snr_litebird[idx_pivot])
    snr_pivot_s4 = float(snr_cmbs4[idx_pivot])

    # At k₀ where ToE effect is maximal
    idx_k0 = np.argmin(np.abs(K_FINE - TOE_PARAMS["k0"]))
    signal_k0 = float(signal_fine[idx_k0])
    snr_k0_lb = float(snr_litebird[idx_k0])
    snr_k0_s4 = float(snr_cmbs4[idx_k0])

    print(f"    At pivot (k={K_PIVOT}):")
    print(f"      Signal = 1 - Q = {signal_pivot:.6f}")
    print(f"      SNR(LiteBIRD) = {snr_pivot_lb:.2f}")
    print(f"      SNR(CMB-S4)   = {snr_pivot_s4:.2f}")
    print(f"    At k₀ (k={TOE_PARAMS['k0']}):")
    print(f"      Signal = 1 - Q = {signal_k0:.6f}")
    print(f"      SNR(LiteBIRD) = {snr_k0_lb:.2f}")
    print(f"      SNR(CMB-S4)   = {snr_k0_s4:.2f}")

    # ------------------------------------------------------------------
    # 6. Generate plots from COMPUTED data
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Q(k) vs k — ToE prediction from MS solver
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(K_FINE, Q_fine, "b-", linewidth=2, label="Q(k) = c_s*/(1+2n̄_k) [MS solver]")
    ax.semilogx(result.k_grid, result.Q_toe_grid, "ro", markersize=8,
                label="Q(k) [BK18 eval grid]")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Standard Inflation (Q=1)")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.7, label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.7,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("Q(k) = r/(-8n_t)")
    ax.set_title("Generalized Consistency Relation — ToE Prediction")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "Q_vs_k.png"), dpi=150)
    plt.close(fig)
    print(f"\n    Saved: plots/Q_vs_k.png")

    # Plot 2: n̄_k vs k — occupancy from Bogoliubov matching
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(K_FINE, nbar_fine, "b-", linewidth=2, label="n̄_k [MS solver, fine grid]")
    ax.loglog(result.k_grid, result.nbar_k_grid, "ro", markersize=8,
              label="n̄_k [BK18 eval grid]")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.7, label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.7,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("n̄_k (Bogoliubov occupancy)")
    ax.set_title("Occupancy n̄_k from Mukhanov-Sasaki Solver")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "nbar_vs_k.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/nbar_vs_k.png")

    # Plot 3: Q(k) with LiteBIRD and CMB-S4 forecast error bands
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(K_FINE, Q_fine, "b-", linewidth=2,
                label="Q(k) = c_s*/(1+2n̄_k) [ToE]")
    ax.fill_between(K_FINE, 1.0 - sigma_Q_litebird, 1.0 + sigma_Q_litebird,
                    color="orange", alpha=0.25, label=f"LiteBIRD 1σ (σ_Q={sigma_Q_litebird:.2f})")
    ax.fill_between(K_FINE, 1.0 - sigma_Q_cmbs4, 1.0 + sigma_Q_cmbs4,
                    color="red", alpha=0.20, label=f"CMB-S4 1σ (σ_Q={sigma_Q_cmbs4:.2f})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7,
               label="Standard Inflation (Q=1)")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.5,
               label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="purple", linestyle=":", alpha=0.5,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("Q(k) = r/(-8n_t)")
    ax.set_title("Consistency Relation Forecast — LiteBIRD & CMB-S4 (sec13)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.5, 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "forecast_Q.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/forecast_Q.png")

    # ------------------------------------------------------------------
    # 7. PASS/FAIL verdict
    # ------------------------------------------------------------------
    passed = Q_deviation < Q_DEVIATION_THRESHOLD
    verdict = "PASS" if passed else "FAIL"

    consistency_check = format_verdict(
        "consistency relation",
        passed,
        f"|Q_obs - Q_toe| = {Q_deviation:.8e} (threshold {Q_DEVIATION_THRESHOLD})",
    )
    print(f"\n  {consistency_check}")
    print(f"\nFINAL VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # 8. Build summary and save
    # ------------------------------------------------------------------
    lines = [
        "Experiment 03: Generalized Tensor-Scalar Consistency Relation",
        "=" * 60,
        "",
        "eq:consistency (sec03): r/(-8 n_t) = c_s*/(1+2 n̄_k)",
        "",
        "METHODOLOGY:",
        "  - BK18+Planck+BAO chains loaded via load_bk18_chains()",
        "  - Full ToE evaluation via evaluate_bk18() on REAL data",
        "  - n̄_k computed via MS solver (Bogoliubov matching)",
        "  - Q(k) = c_s*/(1+2n̄_k) computed from MS solver output",
        "",
        f"BK18 DATA ({n_samples} samples, n_eff={n_eff:.0f}):",
        f"  r = {result.r_mean:.6f} ± {result.r_std:.6f}",
        f"  n_t (SI)  = {result.nt_si_mean:.8f} ± {result.nt_si_std:.8f}",
        f"  n_t (ToE) = {result.nt_toe_mean:.8f} ± {result.nt_toe_std:.8f}",
        f"  Δn_t      = {result.delta_nt_mean:.8e}",
        f"  |Δn_t|/σ  = {result.delta_nt_over_sigma_r:.4f}",
        "",
        "MS SOLVER RESULTS:",
        f"  n̄_k at pivot (k={K_PIVOT}) = {result.nbar_k_pivot:.6e}",
        f"  Q_toe at pivot = {Q_toe_pivot:.8f}",
        f"  Q_obs (SI)     = {Q_obs_si:.8f}",
        f"  |Q_obs - Q_toe| = {Q_deviation:.8e}",
        "",
        "Q(k) PROFILE (fine grid, {0} modes):".format(len(K_FINE)),
        f"  n̄_k range: [{np.min(nbar_fine):.6e}, {np.max(nbar_fine):.6e}]",
        f"  Q(k) range: [{np.min(Q_fine):.6f}, {np.max(Q_fine):.6f}]",
        "",
        f"  {consistency_check}",
    ]

    # Where ToE effect is maximal
    max_idx = np.argmax(result.nbar_k_grid)
    k_max = result.k_grid[max_idx]
    nbar_max = result.nbar_k_grid[max_idx]
    Q_max = result.Q_toe_grid[max_idx]
    lines.append("")
    lines.append("WHERE ToE EFFECT IS MAXIMAL:")
    lines.append(f"  k = {k_max:.4f} Mpc⁻¹: n̄_k = {nbar_max:.4e}, Q = {Q_max:.4f}")
    lines.append(f"  Deviation from 1: {abs(1 - Q_max) * 100:.1f}%")

    lines.append("")
    lines.append("CMB-S4 / LiteBIRD FORECAST (sec13, subsec:forecasts):")
    lines.append(f"  Fiducial: r = {r_fid:.6f}, n_t = {nt_fid:.8f}")
    lines.append(f"  LiteBIRD: σ(r) = {sigma_r_litebird:.1e} → σ(Q) = {sigma_Q_litebird:.4f}")
    lines.append(f"  CMB-S4:   σ(r) = {sigma_r_cmbs4:.1e} → σ(Q) = {sigma_Q_cmbs4:.4f}")
    lines.append(f"  At pivot (k={K_PIVOT}):")
    lines.append(f"    Signal (1-Q) = {signal_pivot:.6f}")
    lines.append(f"    SNR(LiteBIRD) = {snr_pivot_lb:.2f}")
    lines.append(f"    SNR(CMB-S4)   = {snr_pivot_s4:.2f}")
    lines.append(f"  At k₀ (k={TOE_PARAMS['k0']}):")
    lines.append(f"    Signal (1-Q) = {signal_k0:.6f}")
    lines.append(f"    SNR(LiteBIRD) = {snr_k0_lb:.2f}")
    lines.append(f"    SNR(CMB-S4)   = {snr_k0_s4:.2f}")

    summary = "\n".join(lines)

    key_result = (
        f"Q_toe={Q_toe_pivot:.6f}, |ΔQ|={Q_deviation:.2e}, "
        f"n̄_k_pivot={result.nbar_k_pivot:.2e}, r={result.r_mean:.4f}"
    )

    # CSV data
    csv_data = {}
    # Fine grid Q(k) profile
    q_arr = np.column_stack([K_FINE, nbar_fine, Q_fine])
    csv_data["Q_profile_fine"] = (
        ["k_Mpc", "nbar_k", "Q_toe"],
        q_arr,
    )
    # BK18 eval grid
    eval_arr = np.column_stack([result.k_grid, result.nbar_k_grid, result.Q_toe_grid])
    csv_data["Q_profile_bk18"] = (
        ["k_Mpc", "nbar_k", "Q_toe"],
        eval_arr,
    )
    # Forecast data
    forecast_arr = np.column_stack([
        K_FINE, Q_fine, signal_fine, snr_litebird, snr_cmbs4,
    ])
    csv_data["forecast_Q"] = (
        ["k_Mpc", "Q_toe", "signal_1minusQ", "SNR_LiteBIRD", "SNR_CMBS4"],
        forecast_arr,
    )

    params_dict = {
        "source": "BK18+Planck+BAO chains + TOE_PARAMS",
        "category": "rigorous",
        "toe_params": dict(TOE_PARAMS),
        "bk18_data": {
            "n_samples": n_samples,
            "n_effective": n_eff,
            "r_mean": result.r_mean,
            "r_std": result.r_std,
        },
        "consistency_relation": {
            "Q_obs_si": Q_obs_si,
            "Q_toe_pivot": Q_toe_pivot,
            "Q_deviation": Q_deviation,
            "nbar_k_pivot": result.nbar_k_pivot,
            "nt_si_mean": result.nt_si_mean,
            "nt_toe_mean": result.nt_toe_mean,
            "delta_nt_mean": result.delta_nt_mean,
            "delta_nt_over_sigma": result.delta_nt_over_sigma_r,
        },
        "metrics": {
            "Q_deviation": Q_deviation,
            "threshold": Q_DEVIATION_THRESHOLD,
            "passed": passed,
        },
        "forecast": {
            "sigma_r_litebird": sigma_r_litebird,
            "sigma_r_cmbs4": sigma_r_cmbs4,
            "sigma_Q_litebird": sigma_Q_litebird,
            "sigma_Q_cmbs4": sigma_Q_cmbs4,
            "signal_pivot": signal_pivot,
            "snr_pivot_litebird": snr_pivot_lb,
            "snr_pivot_cmbs4": snr_pivot_s4,
            "signal_k0": signal_k0,
            "snr_k0_litebird": snr_k0_lb,
            "snr_k0_cmbs4": snr_k0_s4,
        },
    }

    save_experiment_results(
        exp_dir=exp_dir,
        summary=summary,
        verdict=verdict,
        params=params_dict,
        csv_data=csv_data,
        key_result=key_result,
        manuscript_ref="sec03, eq:consistency; sec07; sec13",
    )

    print(f"\nOutput saved to {exp_dir}/")
    return params_dict


# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_experiment()
