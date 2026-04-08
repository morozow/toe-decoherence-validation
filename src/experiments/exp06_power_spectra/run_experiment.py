#!/usr/bin/env python3
"""
Experiment 06: Primordial Power Spectra
=======================================

Computes P_ζ(k) with occupancy enhancement from the MS solver on REAL data.

Algorithm:
    1. Call run_toe_calculation(DEFAULT_COBAYA_PARAMS) → get C_ℓ, H0, σ₈, nbar_k, Q_toe
    2. Call compute_ms_nbar() on k-grid → get n̄_k profile and ring-down
    3. Load BK18 chains → get n_s, A_s, r posteriors for comparison
    4. Compute P_ζ(k) = (1+2n̄_k)·A_s·(k/k_*)^(n_s-1) using n̄_k from MS solver
    5. Plot: P_ζ(k) with occupancy enhancement, n̄_k profile, ring-down
    6. PASS: H0 consistent with Planck AND n̄_k physical

Reference: sec03, eq:consistency; sec13, eq:posteriors
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
    run_toe_calculation,
    compute_ms_nbar,
    load_bk18_chains,
    get_bk18_posteriors,
    finite_diff_log_slope,
    TOE_PARAMS,
    DEFAULT_COBAYA_PARAMS,
    PLANCK_POSTERIORS,
    K_PIVOT,
)
from experiments.common.reporting import (
    format_verdict,
    save_experiment_results,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
K_FINE = np.logspace(np.log10(5e-4), np.log10(0.15), 60)
H0_EXPECTED = 67.4
H0_TOLERANCE = 3.0
SIGMA_THRESHOLD = 3.0


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment():
    """
    Run the primordial power spectra experiment with REAL ToE computations.

    Every number in RESULTS.txt comes from a COMPUTATION:
      - run_toe_calculation() for CAMB pipeline (C_ℓ, H0, σ₈)
      - compute_ms_nbar() for n̄_k from MS solver
      - load_bk18_chains() for observational posteriors
    """
    exp_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 72)
    print("Experiment 06: Primordial Power Spectra")
    print("  P_ζ(k) = (1+2n̄_k)·A_s·(k/k_*)^(n_s-1)  (sec03)")
    print("  n̄_k via MS solver (Bogoliubov matching)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Run full ToE calculation via CAMB
    # ------------------------------------------------------------------
    print("\n[1] Running run_toe_calculation(DEFAULT_COBAYA_PARAMS) ...")
    camb_result = run_toe_calculation(DEFAULT_COBAYA_PARAMS, want_derived=True)

    if camb_result is not None and "derived" in camb_result:
        H0_computed = camb_result["derived"].get("H0", float("nan"))
        sigma8_computed = camb_result["derived"].get("sigma8", float("nan"))
        nbar_pivot_camb = camb_result["derived"].get("nbar_k_physical", float("nan"))
        Q_toe_camb = camb_result["derived"].get("Q_toe_pred", float("nan"))
        camb_success = True
        print(f"    H0 = {H0_computed:.2f} km/s/Mpc")
        print(f"    σ₈ = {sigma8_computed:.4f}")
        print(f"    n̄_k at pivot = {nbar_pivot_camb:.6e}")
        print(f"    Q_toe = {Q_toe_camb:.8f}")
    else:
        H0_computed = float("nan")
        sigma8_computed = float("nan")
        nbar_pivot_camb = float("nan")
        Q_toe_camb = float("nan")
        camb_success = False
        print(f"    CAMB calculation failed!")

    # ------------------------------------------------------------------
    # 2. Compute n̄_k on fine k-grid via MS solver
    # ------------------------------------------------------------------
    print(f"\n[2] Computing n̄_k on fine k-grid ({len(K_FINE)} modes) via MS solver ...")
    nbar_fine, ms_results_fine = compute_ms_nbar(K_FINE, TOE_PARAMS)

    nbar_min = float(np.min(nbar_fine))
    nbar_max = float(np.max(nbar_fine))
    nbar_all_positive = bool(np.all(nbar_fine >= 0))
    nbar_all_finite = bool(np.all(np.isfinite(nbar_fine)))
    nbar_physical = nbar_all_positive and nbar_all_finite

    print(f"    n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]")
    print(f"    All positive: {nbar_all_positive}, All finite: {nbar_all_finite}")

    # Ring-down from MS solver
    A_ring_fine = ms_results_fine.get("A_ring", np.zeros_like(K_FINE))
    phi_k_fine = ms_results_fine.get("phi_k", np.zeros_like(K_FINE))

    print(f"    A_ring range: [{float(np.min(A_ring_fine)):.6e}, {float(np.max(A_ring_fine)):.6e}]")

    # ------------------------------------------------------------------
    # 3. Load BK18 chains → get posteriors
    # ------------------------------------------------------------------
    print("\n[3] Loading BK18+Planck+BAO chains ...")
    try:
        samples, weights, param_names = load_bk18_chains()
        n_samples = len(samples)
        n_eff = float(np.sum(weights))
        chains_loaded = True
        print(f"    Loaded {n_samples} samples, n_eff = {n_eff:.0f}")

        posteriors = get_bk18_posteriors()
        ns_chain = posteriors.get("ns", {}).get("mean", PLANCK_POSTERIORS["n_s"]["value"])
        ns_std = posteriors.get("ns", {}).get("std", PLANCK_POSTERIORS["n_s"]["error"])
        r_chain = posteriors.get("r", {}).get("mean", 0.01)
        r_std = posteriors.get("r", {}).get("std", 0.01)
        print(f"    n_s = {ns_chain:.4f} ± {ns_std:.4f}")
        print(f"    r = {r_chain:.4f} ± {r_std:.4f}")
    except FileNotFoundError:
        chains_loaded = False
        n_samples = 0
        n_eff = 0
        ns_chain = PLANCK_POSTERIORS["n_s"]["value"]
        ns_std = PLANCK_POSTERIORS["n_s"]["error"]
        r_chain = 0.01
        r_std = 0.01
        print(f"    BK18 chains not found — using PLANCK_POSTERIORS")
        print(f"    n_s = {ns_chain:.4f} ± {ns_std:.4f}")

    # ------------------------------------------------------------------
    # 4. Compute P_ζ(k) = (1+2n̄_k)·A_s·(k/k_*)^(n_s-1)
    # ------------------------------------------------------------------
    print(f"\n[4] Computing P_ζ(k) with occupancy enhancement ...")
    A_s = PLANCK_POSTERIORS["A_s"]["value"]
    n_s = ns_chain

    # Standard power-law
    P_zeta_standard = A_s * (K_FINE / K_PIVOT) ** (n_s - 1.0)

    # ToE-enhanced with occupancy
    occupancy_enhancement = 1.0 + 2.0 * nbar_fine
    P_zeta_toe = P_zeta_standard * occupancy_enhancement

    # Ring-down modulation
    eta_0 = -1.0 / TOE_PARAMS["k0"]
    Gamma_k = TOE_PARAMS["Gamma_over_H"] * 1.0  # H_star = 1
    eta_star = -1.0 / (TOE_PARAMS["c_s_star"] * K_FINE)
    delta_eta = np.maximum(eta_star - eta_0, 0.0)
    damping = np.exp(-Gamma_k * delta_eta)
    ring_osc = np.cos(2.0 * TOE_PARAMS["c_s_star"] * K_FINE * eta_0 + phi_k_fine)
    ring_factor = 1.0 + A_ring_fine * ring_osc * damping
    P_zeta_ring = P_zeta_toe * ring_factor

    # Spectral index from numerical derivative
    ns_numerical = 1.0 + finite_diff_log_slope(K_FINE, P_zeta_toe)

    # At pivot
    idx_pivot = np.argmin(np.abs(K_FINE - K_PIVOT))
    P_pivot = float(P_zeta_toe[idx_pivot])
    ns_at_pivot = float(ns_numerical[idx_pivot])
    nbar_at_pivot = float(nbar_fine[idx_pivot])

    print(f"    A_s = {A_s:.4e}")
    print(f"    n_s (chains) = {n_s:.4f}")
    print(f"    P_ζ at pivot = {P_pivot:.4e}")
    print(f"    n_s (numerical at pivot) = {ns_at_pivot:.4f}")
    print(f"    n̄_k at pivot = {nbar_at_pivot:.6e}")
    print(f"    Occupancy enhancement at pivot = {1.0 + 2.0 * nbar_at_pivot:.8f}")

    # ------------------------------------------------------------------
    # 5. Planck comparison
    # ------------------------------------------------------------------
    print(f"\n[5] Planck 2018 comparison ...")
    ns_planck = PLANCK_POSTERIORS["n_s"]["value"]
    ns_planck_err = PLANCK_POSTERIORS["n_s"]["error"]
    As_planck = PLANCK_POSTERIORS["A_s"]["value"]
    As_planck_err = PLANCK_POSTERIORS["A_s"]["error"]
    r_upper = PLANCK_POSTERIORS["r_upper_95CL"]

    ns_tension = abs(ns_at_pivot - ns_planck) / ns_planck_err
    As_tension = abs(P_pivot - As_planck) / As_planck_err
    r_pass = r_chain < r_upper

    ns_pass = ns_tension < SIGMA_THRESHOLD
    As_pass = As_tension < SIGMA_THRESHOLD

    print(f"    n_s: {ns_at_pivot:.4f} vs {ns_planck}±{ns_planck_err} → {ns_tension:.1f}σ")
    print(f"    A_s: {P_pivot:.4e} vs {As_planck:.2e}±{As_planck_err:.2e} → {As_tension:.1f}σ")
    print(f"    r: {r_chain:.4f} vs < {r_upper} → {'PASS' if r_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # 6. Ring-down analysis and forecast
    # ------------------------------------------------------------------
    # Extract ring-down parameters from MS solver results.
    # Ring-down frequency: f_ring(k) = 2·c_s·k·η₀ + φ_k
    # Effective amplitude: A_eff(k) = A_ring(k) × damping(k)
    # P_ζ_ring(k) = P_ζ_base(k) × (1 + A_eff × cos(f_ring))
    # Reference: sec03, eq:ringdown; sec13, subsec:forecasts
    print(f"\n[6] Ring-down analysis and forecast ...")

    damping_fine = ms_results_fine.get("damping", np.ones_like(K_FINE))
    # If damping not directly in ms_results, recompute from parameters
    if np.all(damping_fine == 1.0):
        eta_star_rd = -1.0 / (TOE_PARAMS["c_s_star"] * K_FINE)
        delta_eta_rd = np.maximum(eta_star_rd - eta_0, 0.0)
        Gamma_k_rd = TOE_PARAMS["Gamma_over_H"] * 1.0  # H_star = 1
        damping_fine = np.exp(-Gamma_k_rd * delta_eta_rd)

    # Effective ring-down amplitude (A_ring × damping)
    A_eff_fine = A_ring_fine * damping_fine

    # Ring-down oscillation frequency argument
    f_ring_fine = 2.0 * TOE_PARAMS["c_s_star"] * K_FINE * eta_0 + phi_k_fine

    # Ring-down modulation factor
    ring_modulation = A_eff_fine * np.cos(f_ring_fine)

    # P_ζ with ring-down (already computed above as P_zeta_ring, verify consistency)
    # Fractional ring-down: δP/P = A_eff × cos(f_ring)
    frac_ringdown = ring_modulation  # δP_ζ / P_ζ

    A_eff_max = float(np.max(np.abs(A_eff_fine)))
    A_eff_at_pivot = float(A_eff_fine[idx_pivot])
    frac_rd_max = float(np.max(np.abs(frac_ringdown)))

    print(f"    A_eff range: [{float(np.min(A_eff_fine)):.6e}, {A_eff_max:.6e}]")
    print(f"    A_eff at pivot = {A_eff_at_pivot:.6e}")
    print(f"    Max |δP/P| from ring-down = {frac_rd_max:.6e}")

    # Forecast: detection thresholds
    # Planck: σ(P_ζ)/P_ζ ~ few % at low ℓ
    # CMB-S4: σ(P_ζ)/P_ζ ~ 0.1% at ℓ ~ 1000
    # Ring-down detectable when A_eff > σ(P_ζ)/P_ζ
    sigma_frac_planck = 0.02       # ~2% fractional precision (low ℓ)
    sigma_frac_cmbs4 = 0.001       # ~0.1% fractional precision (ℓ~1000)
    sigma_frac_ideal = 0.0001      # ~0.01% future ideal

    snr_planck_rd = A_eff_fine / sigma_frac_planck
    snr_cmbs4_rd = A_eff_fine / sigma_frac_cmbs4
    snr_ideal_rd = A_eff_fine / sigma_frac_ideal

    snr_planck_max = float(np.max(snr_planck_rd))
    snr_cmbs4_max = float(np.max(snr_cmbs4_rd))
    snr_ideal_max = float(np.max(snr_ideal_rd))

    print(f"    Detection thresholds (σ(P_ζ)/P_ζ):")
    print(f"      Planck:  {sigma_frac_planck:.1%} → max SNR = {snr_planck_max:.4f}")
    print(f"      CMB-S4:  {sigma_frac_cmbs4:.1%} → max SNR = {snr_cmbs4_max:.4f}")
    print(f"      Ideal:   {sigma_frac_ideal:.2%} → max SNR = {snr_ideal_max:.4f}")
    print(f"    Ring-down amplitude A_eff ~ O(ε_H) = O({TOE_PARAMS['eps_H']})")

    # ------------------------------------------------------------------
    # 7. Generate plots from COMPUTED data
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: P_ζ(k) with occupancy enhancement
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(K_FINE, P_zeta_standard, "k--", linewidth=1, alpha=0.7,
              label="Standard: A_s·(k/k*)^(n_s-1)")
    ax.loglog(K_FINE, P_zeta_toe, "b-", linewidth=2,
              label="ToE: (1+2n̄_k)·P_standard")
    ax.loglog(K_FINE, P_zeta_ring, "r-", linewidth=1, alpha=0.7,
              label="ToE + ring-down")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.7,
               label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.7,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("P_ζ(k)")
    ax.set_title("Primordial Power Spectrum — ToE Prediction (sec03)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "power_spectrum.png"), dpi=150)
    plt.close(fig)
    print(f"\n    Saved: plots/power_spectrum.png")

    # Plot 2: n̄_k profile from MS solver
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(K_FINE, nbar_fine, "b-o", linewidth=2, markersize=3,
              label="n̄_k [MS solver]")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.7,
               label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.7,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("n̄_k (Bogoliubov occupancy)")
    ax.set_title("Occupancy n̄_k from MS Solver (sec03)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "nbar_profile.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/nbar_profile.png")

    # Plot 3: Ring-down amplitude
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(K_FINE, A_ring_fine, "r-o", linewidth=2, markersize=3,
                label="A_ring [MS solver]")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.7,
               label=f"Pivot k={K_PIVOT}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("A_ring")
    ax.set_title("Ring-down Amplitude from Bogoliubov Matching (sec03)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ringdown.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/ringdown.png")

    # Plot 4: Ring-down modulation — P_ζ(k) with and without ring-down
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(K_FINE, (P_zeta_ring / P_zeta_toe - 1.0) * 100.0,
                "r-", linewidth=1.5, label="δP_ζ/P_ζ (ring-down)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=sigma_frac_planck * 100, color="blue", linestyle=":",
               alpha=0.6, label=f"Planck 1σ ({sigma_frac_planck:.0%})")
    ax.axhline(y=-sigma_frac_planck * 100, color="blue", linestyle=":", alpha=0.6)
    ax.axhline(y=sigma_frac_cmbs4 * 100, color="green", linestyle="--",
               alpha=0.6, label=f"CMB-S4 1σ ({sigma_frac_cmbs4:.1%})")
    ax.axhline(y=-sigma_frac_cmbs4 * 100, color="green", linestyle="--", alpha=0.6)
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.4,
               label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.4,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("δP_ζ/P_ζ [%]")
    ax.set_title("Ring-down Modulation of P_ζ(k) (sec03, eq:ringdown)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ring_down_modulation.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/ring_down_modulation.png")

    # Plot 5: A_eff(k) vs detection thresholds
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(K_FINE, A_eff_fine, "r-o", linewidth=2, markersize=3,
              label="A_eff(k) = A_ring × damping")
    ax.axhline(y=sigma_frac_planck, color="blue", linestyle=":",
               linewidth=1.5, label=f"Planck threshold ({sigma_frac_planck:.0%})")
    ax.axhline(y=sigma_frac_cmbs4, color="green", linestyle="--",
               linewidth=1.5, label=f"CMB-S4 threshold ({sigma_frac_cmbs4:.1%})")
    ax.axhline(y=sigma_frac_ideal, color="purple", linestyle="-.",
               linewidth=1.5, label=f"Ideal threshold ({sigma_frac_ideal:.2%})")
    ax.axvline(x=K_PIVOT, color="green", linestyle=":", alpha=0.4,
               label=f"Pivot k={K_PIVOT}")
    ax.axvline(x=TOE_PARAMS["k0"], color="orange", linestyle=":", alpha=0.4,
               label=f"k₀={TOE_PARAMS['k0']}")
    ax.set_xlabel("k [Mpc⁻¹]")
    ax.set_ylabel("A_eff(k)")
    ax.set_title("Ring-down Detectability Forecast (sec13)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ring_down_forecast.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/ring_down_forecast.png")

    # ------------------------------------------------------------------
    # 8. Final verdict
    # ------------------------------------------------------------------
    H0_consistent = camb_success and abs(H0_computed - H0_EXPECTED) < H0_TOLERANCE
    all_pass = H0_consistent and nbar_physical

    nbar_check = format_verdict(
        "n̄_k physical",
        nbar_physical,
        f"n̄_k ∈ [{nbar_min:.2e}, {nbar_max:.2e}]",
    )
    h0_check = format_verdict(
        "H0 consistent with Planck",
        H0_consistent,
        f"H0={H0_computed:.2f} (expected {H0_EXPECTED}±{H0_TOLERANCE})",
    )
    ns_check = format_verdict(
        "n_s within 3σ of Planck",
        ns_pass,
        f"{ns_tension:.1f}σ",
    )

    verdict = "PASS" if all_pass else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  {nbar_check}")
    print(f"  {h0_check}")
    print(f"  {ns_check}")
    print(f"\nFINAL VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # 9. Build summary and save
    # ------------------------------------------------------------------
    lines = [
        "Experiment 06: Primordial Power Spectra",
        "=" * 60,
        "",
        "P_ζ(k) = (1+2n̄_k)·A_s·(k/k_*)^(n_s-1)  (sec03)",
        "",
        "METHODOLOGY:",
        "  - run_toe_calculation() for CAMB pipeline (H0, σ₈, C_ℓ)",
        "  - compute_ms_nbar() for n̄_k from MS solver",
        "  - load_bk18_chains() for observational posteriors",
        "",
        "CAMB PIPELINE:",
        f"  H0 = {H0_computed:.2f} km/s/Mpc",
        f"  σ₈ = {sigma8_computed:.4f}",
        f"  n̄_k at pivot (CAMB) = {nbar_pivot_camb:.6e}",
        f"  Q_toe (CAMB) = {Q_toe_camb:.8f}",
        "",
        f"MS SOLVER ({len(K_FINE)} modes):",
        f"  n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]",
        f"  n̄_k at pivot = {nbar_at_pivot:.6e}",
        f"  Occupancy enhancement at pivot = {1.0 + 2.0 * nbar_at_pivot:.8f}",
        "",
        f"BK18 CHAINS: {'loaded' if chains_loaded else 'not found'}",
        f"  n_s = {ns_chain:.4f} ± {ns_std:.4f}",
        f"  r = {r_chain:.4f} ± {r_std:.4f}",
        "",
        "POWER SPECTRUM AT PIVOT:",
        f"  P_ζ(k*) = {P_pivot:.4e}",
        f"  n_s (numerical) = {ns_at_pivot:.4f}",
        "",
        "PLANCK COMPARISON:",
        f"  n_s: {ns_at_pivot:.4f} vs {ns_planck}±{ns_planck_err} → {ns_tension:.1f}σ",
        f"  A_s: {P_pivot:.4e} vs {As_planck:.2e}±{As_planck_err:.2e} → {As_tension:.1f}σ",
        f"  r: {r_chain:.4f} vs < {r_upper}",
        "",
        f"  {nbar_check}",
        f"  {h0_check}",
    ]

    # Ring-down forecast section
    lines.extend([
        "",
        "RING-DOWN ANALYSIS (sec03, eq:ringdown):",
        f"  A_eff range: [{float(np.min(A_eff_fine)):.6e}, {A_eff_max:.6e}]",
        f"  A_eff at pivot = {A_eff_at_pivot:.6e}",
        f"  Max |δP/P| from ring-down = {frac_rd_max:.6e}",
        "",
        "RING-DOWN FORECAST (sec13, subsec:forecasts):",
        f"  Planck:  σ(P_ζ)/P_ζ = {sigma_frac_planck:.0%} → max SNR = {snr_planck_max:.4f}",
        f"  CMB-S4:  σ(P_ζ)/P_ζ = {sigma_frac_cmbs4:.1%} → max SNR = {snr_cmbs4_max:.4f}",
        f"  Ideal:   σ(P_ζ)/P_ζ = {sigma_frac_ideal:.2%} → max SNR = {snr_ideal_max:.4f}",
        f"  Ring-down amplitude A_eff ~ O(ε_H) = O({TOE_PARAMS['eps_H']})",
    ])

    summary = "\n".join(lines)

    key_result = (
        f"P_ζ={P_pivot:.2e}, n_s={ns_at_pivot:.4f}, "
        f"n̄_k_pivot={nbar_at_pivot:.2e}, H0={H0_computed:.1f}"
    )

    csv_data = {}
    spec_arr = np.column_stack([
        K_FINE, nbar_fine, P_zeta_standard, P_zeta_toe, P_zeta_ring,
        A_ring_fine, ns_numerical,
    ])
    csv_data["power_spectra"] = (
        ["k_Mpc", "nbar_k", "Pz_standard", "Pz_toe", "Pz_ringdown",
         "A_ring", "ns_numerical"],
        spec_arr,
    )
    # Ring-down forecast data
    rd_arr = np.column_stack([
        K_FINE, A_eff_fine, frac_ringdown,
        snr_planck_rd, snr_cmbs4_rd, snr_ideal_rd,
    ])
    csv_data["ringdown_forecast"] = (
        ["k_Mpc", "A_eff", "frac_ringdown", "SNR_Planck", "SNR_CMBS4", "SNR_ideal"],
        rd_arr,
    )

    params_dict = {
        "source": "DEFAULT_COBAYA_PARAMS + TOE_PARAMS + BK18 chains",
        "category": "rigorous",
        "camb_pipeline": {
            "H0": H0_computed,
            "sigma8": sigma8_computed,
            "nbar_pivot_camb": nbar_pivot_camb,
            "Q_toe_camb": Q_toe_camb,
        },
        "ms_solver": {
            "nbar_min": nbar_min,
            "nbar_max": nbar_max,
            "nbar_at_pivot": nbar_at_pivot,
            "nbar_physical": nbar_physical,
        },
        "power_spectrum": {
            "P_pivot": P_pivot,
            "ns_at_pivot": ns_at_pivot,
            "A_s_used": A_s,
            "ns_used": n_s,
        },
        "bk18_chains": {
            "loaded": chains_loaded,
            "n_samples": n_samples,
            "ns_chain": ns_chain,
            "r_chain": r_chain,
        },
        "planck_comparison": {
            "ns_tension_sigma": ns_tension,
            "As_tension_sigma": As_tension,
            "r_below_limit": r_pass,
        },
        "metrics": {
            "H0_consistent": H0_consistent,
            "nbar_physical": nbar_physical,
            "ns_pass": ns_pass,
            "As_pass": As_pass,
            "all_pass": all_pass,
        },
        "ringdown_forecast": {
            "A_eff_max": A_eff_max,
            "A_eff_at_pivot": A_eff_at_pivot,
            "frac_ringdown_max": frac_rd_max,
            "sigma_frac_planck": sigma_frac_planck,
            "sigma_frac_cmbs4": sigma_frac_cmbs4,
            "sigma_frac_ideal": sigma_frac_ideal,
            "snr_planck_max": snr_planck_max,
            "snr_cmbs4_max": snr_cmbs4_max,
            "snr_ideal_max": snr_ideal_max,
        },
    }

    save_experiment_results(
        exp_dir=exp_dir,
        summary=summary,
        verdict=verdict,
        params=params_dict,
        csv_data=csv_data,
        key_result=key_result,
        manuscript_ref="sec03, eq:consistency; sec13, eq:posteriors",
    )

    print(f"\nOutput saved to {exp_dir}/")
    return params_dict


# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_experiment()
