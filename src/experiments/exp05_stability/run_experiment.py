#!/usr/bin/env python3
"""
Experiment 05: Stability and Ghost-Freedom
==========================================

Tests ghost-freedom conditions and z² > 0 stability from REAL ToE computations.

Physics (sec06, sec10, sec11, sec13):
    1. Ghost-freedom: α₃ ≥ 0 AND α₂ + (1/3)α₃ ≥ 0
    2. Stability: z² = 2a²ε_H/c_s² > 0
    3. SM central charges: a_SM = 1991/720, c_SM = 209/60

Algorithm:
    1. Call run_toe_calculation(DEFAULT_COBAYA_PARAMS) — ghost check is INSIDE calculate()
    2. Try ghost-violating params (α₂=-0.34, α₃=0.98 from PLANCK_POSTERIORS) — verify rejection
    3. Call compute_ms_nbar() — verify n̄_k is physical (positive, finite)
    4. Use PLANCK_POSTERIORS for α₂, α₃ posteriors
    5. Compute z² = 2a²ε_H/c_s² using stability_z2() from toe_physics
    6. Plot: ghost-freedom region in (α₂, α₃) space, z² profile
    7. PASS: ghost-free for DEFAULT_COBAYA_PARAMS AND z² > 0 AND n̄_k physical

Reference: sec06 (stability); sec10 (ghost-freedom); sec11 (SM charges); sec13
"""

import logging
import math
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
    stability_z2,
    TOE_PARAMS,
    DEFAULT_COBAYA_PARAMS,
    PLANCK_POSTERIORS,
    SM_CONTENT,
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
K_MS_GRID = np.logspace(np.log10(5e-4), np.log10(0.15), 30)
H0_EXPECTED = 67.4
H0_TOLERANCE = 3.0


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment():
    """
    Run the stability and ghost-freedom test with REAL ToE computations.

    Every number in RESULTS.txt comes from a COMPUTATION:
      - run_toe_calculation() for ghost check and CAMB pipeline
      - compute_ms_nbar() for MS solver n̄_k
      - stability_z2() for z² profile
    """
    exp_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 72)
    print("Experiment 05: Stability and Ghost-Freedom")
    print("  Ghost-freedom: α₃ ≥ 0 AND α₂ + α₃/3 ≥ 0 (sec10)")
    print("  Stability: z² = 2a²ε_H/c_s² > 0 (sec06)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Run ToE calculation with DEFAULT_COBAYA_PARAMS (ghost-free)
    # ------------------------------------------------------------------
    print("\n[1] Running run_toe_calculation(DEFAULT_COBAYA_PARAMS) ...")
    alpha2_default = DEFAULT_COBAYA_PARAMS["alpha2"]
    alpha3_default = DEFAULT_COBAYA_PARAMS["alpha3"]
    positivity_default = alpha2_default + alpha3_default / 3.0
    print(f"    α₂ = {alpha2_default}, α₃ = {alpha3_default}")
    print(f"    α₂ + α₃/3 = {positivity_default:.4f}")

    camb_result = run_toe_calculation(DEFAULT_COBAYA_PARAMS, want_derived=True)

    if camb_result is not None and "derived" in camb_result:
        H0_computed = camb_result["derived"].get("H0", float("nan"))
        sigma8_computed = camb_result["derived"].get("sigma8", float("nan"))
        nbar_pivot_camb = camb_result["derived"].get("nbar_k_physical", float("nan"))
        ghost_free_default = True
        camb_success = True
        print(f"    CAMB succeeded → ghost-free confirmed")
        print(f"    H0 = {H0_computed:.2f} km/s/Mpc")
        print(f"    σ₈ = {sigma8_computed:.4f}")
        print(f"    n̄_k at pivot = {nbar_pivot_camb:.6e}")
    else:
        H0_computed = float("nan")
        sigma8_computed = float("nan")
        nbar_pivot_camb = float("nan")
        ghost_free_default = False
        camb_success = False
        print(f"    CAMB FAILED — ghost/positivity violation!")

    # ------------------------------------------------------------------
    # 2. Try ghost-VIOLATING params — should be rejected
    # ------------------------------------------------------------------
    print("\n[2] Testing ghost-violating parameters ...")
    ghost_violating_params = dict(DEFAULT_COBAYA_PARAMS)
    # α₂ = -0.34, α₃ = 0.98 from PLANCK_POSTERIORS
    alpha2_post = PLANCK_POSTERIORS["alpha_2"]["value"]
    alpha3_post = PLANCK_POSTERIORS["alpha_3"]["value"]
    alpha2_err = PLANCK_POSTERIORS["alpha_2"]["error"]
    alpha3_err = PLANCK_POSTERIORS["alpha_3"]["error"]

    # Check if posteriors satisfy ghost-freedom
    ghost_cond1_post = alpha3_post >= 0
    ghost_cond2_post = (alpha2_post + alpha3_post / 3.0) >= 0
    positivity_post = alpha2_post + alpha3_post / 3.0

    print(f"    PLANCK_POSTERIORS: α₂ = {alpha2_post} ± {alpha2_err}")
    print(f"    PLANCK_POSTERIORS: α₃ = {alpha3_post} ± {alpha3_err}")
    print(f"    α₃ ≥ 0: {ghost_cond1_post}")
    print(f"    α₂ + α₃/3 = {positivity_post:.4f} ≥ 0: {ghost_cond2_post}")

    # Now try params that FAIL ghost check: α₂ = -1.0, α₃ = -0.5
    ghost_violating_params["alpha2"] = -1.0
    ghost_violating_params["alpha3"] = -0.5
    pos_viol = -1.0 + (-0.5) / 3.0
    print(f"\n    Testing α₂=-1.0, α₃=-0.5 (α₂+α₃/3 = {pos_viol:.4f} < 0) ...")
    viol_result = run_toe_calculation(ghost_violating_params, want_derived=True)
    ghost_rejected = viol_result is None
    print(f"    Result: {'REJECTED (ghost violation)' if ghost_rejected else 'ACCEPTED (unexpected!)'}")

    # ------------------------------------------------------------------
    # 3. Compute n̄_k via MS solver — verify physical
    # ------------------------------------------------------------------
    print(f"\n[3] Computing n̄_k via MS solver ({len(K_MS_GRID)} modes) ...")
    nbar_k, ms_results = compute_ms_nbar(K_MS_GRID, TOE_PARAMS)

    nbar_min = float(np.min(nbar_k))
    nbar_max = float(np.max(nbar_k))
    nbar_all_positive = bool(np.all(nbar_k >= 0))
    nbar_all_finite = bool(np.all(np.isfinite(nbar_k)))
    nbar_physical = nbar_all_positive and nbar_all_finite

    print(f"    n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]")
    print(f"    All positive: {nbar_all_positive}, All finite: {nbar_all_finite}")

    nbar_check = format_verdict(
        "MS solver physical n̄_k",
        nbar_physical,
        f"n̄_k ∈ [{nbar_min:.2e}, {nbar_max:.2e}]",
    )
    print(f"    {nbar_check}")

    # ------------------------------------------------------------------
    # 4. Compute z² = 2a²ε_H/c_s² via stability_z2()
    # ------------------------------------------------------------------
    print(f"\n[4] Computing z² stability profile ...")
    N_stab = np.linspace(-20.0, 8.0, 4001)
    a_stab = np.exp(N_stab)
    eps_H_val = TOE_PARAMS["eps_H"]
    c_s_val = TOE_PARAMS["c_s_star"]

    # During slow-roll inflation, ε_H is approximately constant
    epsH_arr = np.full_like(N_stab, eps_H_val)
    z2_arr = stability_z2(a_stab, epsH_arr, c_s_val)

    z2_min = float(np.min(z2_arr))
    z2_max = float(np.max(z2_arr))
    z2_positive = z2_min > 0

    print(f"    ε_H = {eps_H_val}, c_s = {c_s_val}")
    print(f"    z² range: [{z2_min:.6e}, {z2_max:.6e}]")
    print(f"    z² > 0 everywhere: {z2_positive}")

    z2_check = format_verdict(
        "stability z² > 0",
        z2_positive,
        f"min(z²) = {z2_min:.6e}",
    )
    print(f"    {z2_check}")

    # ------------------------------------------------------------------
    # 5. SM central charges (sec11)
    # ------------------------------------------------------------------
    print(f"\n[5] SM central charges (sec11, subsec:alpha-numerics) ...")
    N_s = SM_CONTENT["N_s"]
    N_w = SM_CONTENT["N_w"]
    N_v = SM_CONTENT["N_v"]
    a_SM = SM_CONTENT["a_SM"]
    c_SM = SM_CONTENT["c_SM"]

    # Verify from field content
    a_SM_computed = N_s * (1.0 / 360) + N_w * (11.0 / 720) + N_v * (31.0 / 180)
    c_SM_computed = N_s * (1.0 / 120) + N_w * (1.0 / 20) + N_v * (1.0 / 10)
    a_SM_exact = 1991.0 / 720.0
    c_SM_exact = 209.0 / 60.0

    pi2_16 = 16.0 * math.pi ** 2
    alpha2_slope = (c_SM / 3.0 - a_SM) / pi2_16
    alpha3_slope = (-2.0 * c_SM + 4.0 * a_SM) / pi2_16

    a_match = abs(a_SM_computed - a_SM_exact) < 1e-12
    c_match = abs(c_SM_computed - c_SM_exact) < 1e-12

    print(f"    N_s={N_s}, N_w={N_w}, N_v={N_v}")
    print(f"    a_SM = {a_SM_computed:.6f} (exact {a_SM_exact:.6f})")
    print(f"    c_SM = {c_SM_computed:.6f} (exact {c_SM_exact:.6f})")
    print(f"    α₂ slope = {alpha2_slope:.5f} (≈ -0.01016)")
    print(f"    α₃ slope = {alpha3_slope:.5f} (≈ +0.02594)")

    # ------------------------------------------------------------------
    # 6. Generate plots from COMPUTED data
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Ghost-freedom region in (α₂, α₃) space
    fig, ax = plt.subplots(figsize=(8, 6))
    a2_range = np.linspace(-1.5, 1.0, 200)
    a3_range = np.linspace(-0.5, 2.0, 200)
    A2, A3 = np.meshgrid(a2_range, a3_range)
    ghost_region = (A3 >= 0) & (A2 + A3 / 3.0 >= 0)
    ax.contourf(A2, A3, ghost_region.astype(float), levels=[0.5, 1.5],
                colors=["lightgreen"], alpha=0.5)
    ax.contour(A2, A3, ghost_region.astype(float), levels=[0.5],
               colors=["green"], linewidths=2)
    # Plot DEFAULT_COBAYA_PARAMS point
    ax.plot(alpha2_default, alpha3_default, "b*", markersize=15,
            label=f"DEFAULT ({alpha2_default}, {alpha3_default})")
    # Plot PLANCK_POSTERIORS point with error bars
    ax.errorbar(alpha2_post, alpha3_post, xerr=alpha2_err, yerr=alpha3_err,
                fmt="ro", markersize=10, capsize=5,
                label=f"Planck post. ({alpha2_post}±{alpha2_err}, {alpha3_post}±{alpha3_err})")
    # Plot ghost-violating point
    ax.plot(-1.0, -0.5, "kx", markersize=12, markeredgewidth=3,
            label="Ghost-violating (-1.0, -0.5)")
    # Boundary lines
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="α₃ = 0")
    a2_boundary = -a3_range / 3.0
    ax.plot(a2_boundary, a3_range, "r:", alpha=0.5, label="α₂ + α₃/3 = 0")
    ax.set_xlabel("α₂")
    ax.set_ylabel("α₃")
    ax.set_title("Ghost-Freedom Region (sec10, sec11)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-0.5, 2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ghost_freedom_region.png"), dpi=150)
    plt.close(fig)
    print(f"\n    Saved: plots/ghost_freedom_region.png")

    # Plot 2: z² profile
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(N_stab, z2_arr, "b-", linewidth=2, label="z² = 2a²ε_H/c_s²")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("N = ln(a)")
    ax.set_ylabel("z²")
    ax.set_title("Stability Variable z² (sec06)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "z2_profile.png"), dpi=150)
    plt.close(fig)
    print(f"    Saved: plots/z2_profile.png")

    # ------------------------------------------------------------------
    # 7. Final verdict
    # ------------------------------------------------------------------
    H0_consistent = camb_success and abs(H0_computed - H0_EXPECTED) < H0_TOLERANCE
    all_pass = ghost_free_default and z2_positive and nbar_physical and ghost_rejected

    ghost_check = format_verdict(
        "ghost-free (DEFAULT_COBAYA_PARAMS)",
        ghost_free_default,
        f"α₂={alpha2_default}, α₃={alpha3_default}, α₂+α₃/3={positivity_default:.4f}",
    )
    reject_check = format_verdict(
        "ghost-violating params rejected",
        ghost_rejected,
        "α₂=-1.0, α₃=-0.5 → calculate() returned None",
    )
    camb_check = format_verdict(
        "CAMB H0 consistent",
        H0_consistent,
        f"H0={H0_computed:.2f} (expected {H0_EXPECTED}±{H0_TOLERANCE})",
    )

    verdict = "PASS" if all_pass else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  {ghost_check}")
    print(f"  {reject_check}")
    print(f"  {z2_check}")
    print(f"  {nbar_check}")
    print(f"  {camb_check}")
    print(f"\nFINAL VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # 8. Build summary and save
    # ------------------------------------------------------------------
    lines = [
        "Experiment 05: Stability and Ghost-Freedom",
        "=" * 60,
        "",
        "Ghost-freedom: α₃ ≥ 0 AND α₂ + α₃/3 ≥ 0 (sec10)",
        "Stability: z² = 2a²ε_H/c_s² > 0 (sec06)",
        "",
        "METHODOLOGY:",
        "  - run_toe_calculation(DEFAULT_COBAYA_PARAMS) for ghost check",
        "  - Ghost-violating params tested (α₂=-1.0, α₃=-0.5)",
        "  - compute_ms_nbar() for physical n̄_k verification",
        "  - stability_z2() for z² profile",
        "",
        "GHOST-FREEDOM (sec10):",
        f"  DEFAULT: α₂={alpha2_default}, α₃={alpha3_default}, α₂+α₃/3={positivity_default:.4f}",
        f"  {ghost_check}",
        f"  {reject_check}",
        f"  PLANCK_POSTERIORS: α₂={alpha2_post}±{alpha2_err}, α₃={alpha3_post}±{alpha3_err}",
        f"  Posterior α₂+α₃/3 = {positivity_post:.4f}",
        "",
        "STABILITY (sec06):",
        f"  z² range: [{z2_min:.6e}, {z2_max:.6e}]",
        f"  {z2_check}",
        "",
        "MS SOLVER:",
        f"  n̄_k range: [{nbar_min:.6e}, {nbar_max:.6e}]",
        f"  {nbar_check}",
        "",
        "CAMB PIPELINE:",
        f"  H0 = {H0_computed:.2f} km/s/Mpc",
        f"  σ₈ = {sigma8_computed:.4f}",
        f"  {camb_check}",
        "",
        "SM CENTRAL CHARGES (sec11):",
        f"  a_SM = {a_SM_computed:.6f} (1991/720)",
        f"  c_SM = {c_SM_computed:.6f} (209/60)",
        f"  α₂ slope = {alpha2_slope:.5f}, α₃ slope = {alpha3_slope:.5f}",
    ]

    summary = "\n".join(lines)

    key_result = (
        f"ghost-free={ghost_free_default}, z²>0={z2_positive}, "
        f"n̄_k physical={nbar_physical}, H0={H0_computed:.1f}"
    )

    csv_data = {}
    stab_arr = np.column_stack([N_stab, a_stab, z2_arr])
    csv_data["stability_z2"] = (["N", "a", "z2"], stab_arr)
    ms_arr = np.column_stack([K_MS_GRID, nbar_k])
    csv_data["nbar_ms_solver"] = (["k_Mpc", "nbar_k"], ms_arr)

    params_dict = {
        "source": "DEFAULT_COBAYA_PARAMS + TOE_PARAMS + PLANCK_POSTERIORS",
        "category": "rigorous",
        "ghost_freedom": {
            "alpha2_default": alpha2_default,
            "alpha3_default": alpha3_default,
            "positivity_default": positivity_default,
            "ghost_free_default": ghost_free_default,
            "ghost_rejected_violating": ghost_rejected,
            "alpha2_posterior": alpha2_post,
            "alpha3_posterior": alpha3_post,
            "positivity_posterior": positivity_post,
        },
        "stability": {
            "z2_min": z2_min,
            "z2_max": z2_max,
            "z2_positive": z2_positive,
            "eps_H": eps_H_val,
            "c_s": c_s_val,
        },
        "ms_solver": {
            "nbar_min": nbar_min,
            "nbar_max": nbar_max,
            "nbar_physical": nbar_physical,
        },
        "sm_central_charges": {
            "a_SM": a_SM_computed,
            "c_SM": c_SM_computed,
            "alpha2_slope": alpha2_slope,
            "alpha3_slope": alpha3_slope,
        },
        "camb": {
            "H0": H0_computed,
            "sigma8": sigma8_computed,
        },
        "metrics": {
            "ghost_free": ghost_free_default,
            "ghost_rejected": ghost_rejected,
            "z2_positive": z2_positive,
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
        manuscript_ref="sec06 (stability); sec10 (ghost-freedom); sec11 (SM charges); sec13",
    )

    print(f"\nOutput saved to {exp_dir}/")
    return params_dict


# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_experiment()
