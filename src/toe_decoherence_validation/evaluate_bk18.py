#!/usr/bin/env python3
"""
ToE Error Evaluation on BK18 Chains (NASA BICEP/Keck 2018 + Planck + BAO)

Takes REAL observational data (BK18 public chains) and applies ToE physics
to measure whether the consistency relation deviates from unity.

WHAT THIS DOES:
  1. Loads BK18 chains (r is free, n_t = -r/8 is fixed = standard inflation)
  2. Computes n̄_k via MS solver using ToETheoryErrorEval._compute_ms_on_sparse_grid
  3. Computes Q_ToE = c_s* / (1 + 2*n̄_k) — ToE prediction
  4. Computes n_t_ToE = -r * (1 + 2*n̄_k) / (8 * c_s*) — ToE tensor tilt
  5. Computes Δn_t = n_t_ToE - n_t_SI — shift from standard inflation

ALL PHYSICS imported from toe_decoherence_validation.toe_theory.ToETheoryErrorEval.
No formulas duplicated.

Usage:
    tar xzf modeling/chains_no_data_files.tar.gz -C /tmp/bk18_chains/
    python toe_error_evaluation/evaluate_bk18.py

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Add project root to path

# Import ToE theory class — ALL physics from here
from toe_decoherence_validation.toe_theory import ToETheoryErrorEval


# =============================================================================
# ToE PARAMETERS FROM MANUSCRIPT (not in BK18 chains)
# =============================================================================

TOE_PARAMS = {
    "k0": 0.002,
    "eps_H": 0.01,
    "eta_H": 0.005,
    "s_cs": 0.0,
    "c_s_star": 1.0,
    "Gamma_over_H": 5.0,
}

K_PIVOT = 0.05


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class BK18EvalResult:
    """Result of ToE evaluation on BK18 chains."""
    r_mean: float
    r_std: float
    n_samples: int
    n_effective: float
    nbar_k_pivot: float
    Q_toe: float
    nt_si_mean: float
    nt_si_std: float
    nt_toe_mean: float
    nt_toe_std: float
    delta_nt_mean: float
    delta_nt_std: float
    k_grid: np.ndarray
    nbar_k_grid: np.ndarray
    Q_toe_grid: np.ndarray
    delta_nt_over_sigma_r: float


# =============================================================================
# CHAIN LOADING
# =============================================================================

def load_bk18_chains(chains_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load BK18 chains. Returns (samples, weights, param_name_to_index)."""
    paramnames_file = None
    for f in Path(chains_dir).glob("*.paramnames"):
        paramnames_file = f
        break
    if paramnames_file is None:
        raise FileNotFoundError(f"No .paramnames file in {chains_dir}")

    param_names = {}
    with open(paramnames_file) as f:
        for i, line in enumerate(f):
            name = line.split()[0].strip("*")
            param_names[name] = i

    prefix = paramnames_file.stem
    chain_files = sorted(Path(chains_dir).glob(f"{prefix}_*.txt"))
    if not chain_files:
        raise FileNotFoundError(f"No chain .txt files in {chains_dir}")

    chains = []
    for cf in chain_files:
        chains.append(np.loadtxt(str(cf)))

    all_data = np.vstack(chains)
    weights = all_data[:, 0]
    samples = all_data[:, 2:]
    return samples, weights, param_names


# =============================================================================
# EVALUATION USING ToETheoryErrorEval
# =============================================================================

def evaluate(
    samples: np.ndarray,
    weights: np.ndarray,
    param_names: Dict[str, int],
    toe_params: dict = None,
) -> BK18EvalResult:
    """
    BK18 data + ToE physics → Q.
    
    Uses ToETheoryErrorEval._compute_ms_on_sparse_grid for n̄_k computation.
    No formulas duplicated — everything from the theory class.
    """
    if toe_params is None:
        toe_params = TOE_PARAMS

    # Instantiate theory class (without Cobaya — just for MS solver access)
    theory = ToETheoryErrorEval.__new__(ToETheoryErrorEval)
    theory.k_pivot = K_PIVOT
    theory.c_s_star = toe_params["c_s_star"]
    theory.n_k_ms = 30

    # Extract r from chains
    r_idx = param_names["r"]
    r_samples = samples[:, r_idx]
    r_mean = float(np.average(r_samples, weights=weights))
    r_std = float(np.sqrt(np.average((r_samples - r_mean)**2, weights=weights)))

    # =========================================================================
    # Compute n̄_k using theory class MS solver (IMPORTED physics)
    # =========================================================================
    k_grid = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    eta_0 = -1.0 / toe_params["k0"]

    print("Computing n̄_k on k-grid via ToETheoryErrorEval._compute_ms_on_sparse_grid...")
    ms_results = theory._compute_ms_on_sparse_grid(
        k_sparse=k_grid,
        eta_0=eta_0,
        c_s=toe_params["c_s_star"],
        eps_H=toe_params["eps_H"],
        eta_H=toe_params["eta_H"],
        s=toe_params["s_cs"],
        Gamma_over_H=toe_params["Gamma_over_H"],
    )

    nbar_k_grid = ms_results["nbar_k"]

    # Q_ToE = c_s* / (1 + 2*n̄_k) — from consistency relation (eq:consistency)
    c_s = toe_params["c_s_star"]
    Q_toe_grid = c_s / (1.0 + 2.0 * nbar_k_grid)

    for i, k in enumerate(k_grid):
        print(f"  k={k:.4f}: n̄_k={nbar_k_grid[i]:.6e}, Q_ToE={Q_toe_grid[i]:.6f}")

    # n̄_k at pivot
    pivot_idx = np.argmin(np.abs(k_grid - K_PIVOT))
    nbar_k_pivot = float(nbar_k_grid[pivot_idx])
    Q_toe = float(Q_toe_grid[pivot_idx])

    # =========================================================================
    # n_t comparison using BK18 r values
    # =========================================================================
    # SI: n_t = -r/8
    nt_si = -r_samples / 8.0
    nt_si_mean = float(np.average(nt_si, weights=weights))
    nt_si_std = float(np.sqrt(np.average((nt_si - nt_si_mean)**2, weights=weights)))

    # ToE: n_t = -r * (1 + 2*n̄_k) / (8 * c_s*)
    # This is the rearranged consistency relation: r/(-8 n_t) = c_s*/(1+2n̄_k)
    nt_toe = -r_samples * (1.0 + 2.0 * nbar_k_pivot) / (8.0 * c_s)
    nt_toe_mean = float(np.average(nt_toe, weights=weights))
    nt_toe_std = float(np.sqrt(np.average((nt_toe - nt_toe_mean)**2, weights=weights)))

    # Shift
    delta_nt = nt_toe - nt_si
    delta_nt_mean = float(np.average(delta_nt, weights=weights))
    delta_nt_std = float(np.sqrt(np.average((delta_nt - delta_nt_mean)**2, weights=weights)))

    delta_nt_over_sigma = abs(delta_nt_mean) / nt_si_std if nt_si_std > 0 else 0.0

    return BK18EvalResult(
        r_mean=r_mean, r_std=r_std,
        n_samples=len(r_samples), n_effective=float(weights.sum()),
        nbar_k_pivot=nbar_k_pivot, Q_toe=Q_toe,
        nt_si_mean=nt_si_mean, nt_si_std=nt_si_std,
        nt_toe_mean=nt_toe_mean, nt_toe_std=nt_toe_std,
        delta_nt_mean=delta_nt_mean, delta_nt_std=delta_nt_std,
        k_grid=k_grid, nbar_k_grid=nbar_k_grid, Q_toe_grid=Q_toe_grid,
        delta_nt_over_sigma_r=delta_nt_over_sigma,
    )


# =============================================================================
# REPORT
# =============================================================================

def print_report(result: BK18EvalResult, toe_params: dict):
    print()
    print("=" * 72)
    print("ToE EVALUATION ON BK18 DATA (NASA BICEP/Keck 2018 + Planck + BAO)")
    print("=" * 72)
    print()

    print("1. BK18 DATA")
    print("-" * 72)
    print(f"  r = {result.r_mean:.6f} ± {result.r_std:.6f}")
    print(f"  Samples: {result.n_samples}, effective: {result.n_effective:.0f}")
    print(f"  n_t (SI) = -r/8 = {result.nt_si_mean:.6f} ± {result.nt_si_std:.6f}")
    print()

    print("2. ToE PARAMETERS (from manuscript)")
    print("-" * 72)
    for k, v in toe_params.items():
        print(f"  {k} = {v}")
    print()

    print("3. n̄_k ON k-GRID (from MS solver via ToETheoryErrorEval)")
    print("-" * 72)
    print(f"  {'k [Mpc⁻¹]':<12} {'n̄_k':<14} {'Q_ToE':<14} {'1-Q_ToE':<14}")
    print("  " + "-" * 54)
    for i, k in enumerate(result.k_grid):
        marker = ""
        if abs(k - K_PIVOT) < 0.001:
            marker = " ← PIVOT"
        elif abs(k - toe_params["k0"]) < 0.0001:
            marker = " ← k₀"
        deviation = 1.0 - result.Q_toe_grid[i]
        print(f"  {k:<12.4f} {result.nbar_k_grid[i]:<14.6e} "
              f"{result.Q_toe_grid[i]:<14.8f} {deviation:<14.8e}{marker}")
    print()

    print("4. CONSISTENCY RELATION")
    print("-" * 72)
    print(f"  Standard Inflation: Q = r/(-8 n_t) = 1 (by definition)")
    print(f"  ToE prediction:     Q = c_s*/(1+2n̄_k) = {result.Q_toe:.8f}")
    print(f"  Deviation from 1:   1 - Q = {1.0 - result.Q_toe:.8e}")
    print()

    print("5. TENSOR TILT COMPARISON")
    print("-" * 72)
    print(f"  n_t (SI)  = {result.nt_si_mean:.8f} ± {result.nt_si_std:.8f}")
    print(f"  n_t (ToE) = {result.nt_toe_mean:.8f} ± {result.nt_toe_std:.8f}")
    print(f"  Δn_t      = {result.delta_nt_mean:.8e} ± {result.delta_nt_std:.8e}")
    print(f"  |Δn_t|/σ  = {result.delta_nt_over_sigma_r:.4f}")
    print()

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print()
    dev = abs(1.0 - result.Q_toe)
    if dev < 1e-10:
        print("  Q_ToE = 1.000... (to machine precision)")
        print("  → ToE INDISTINGUISHABLE from SI at pivot scale.")
        print("    n̄_k ≈ 0 at k=0.05 — decoherence has no effect here.")
    elif dev > 0.01:
        print(f"  Q_ToE = {result.Q_toe:.6f} (deviation {dev*100:.2f}%)")
        print("  → ToE DIFFERS from standard inflation.")
        if result.delta_nt_over_sigma_r > 1.0:
            print(f"    |Δn_t|/σ = {result.delta_nt_over_sigma_r:.2f} > 1 → DETECTABLE")
        else:
            print(f"    |Δn_t|/σ = {result.delta_nt_over_sigma_r:.4f} < 1 → NOT detectable yet")
    else:
        print(f"  Q_ToE = {result.Q_toe:.8f} (deviation {dev*100:.6f}%)")
        print("  → Small ToE effect, below detection threshold.")
        print(f"    |Δn_t|/σ = {result.delta_nt_over_sigma_r:.4f}")

    # Show where ToE IS distinguishable
    print()
    print("  WHERE ToE EFFECT IS MAXIMAL:")
    max_idx = np.argmax(result.nbar_k_grid)
    k_max = result.k_grid[max_idx]
    nbar_max = result.nbar_k_grid[max_idx]
    Q_max = result.Q_toe_grid[max_idx]
    print(f"    k = {k_max:.4f} Mpc⁻¹: n̄_k = {nbar_max:.4f}, Q = {Q_max:.4f}")
    print(f"    Deviation from 1: {abs(1-Q_max)*100:.1f}%")
    print(f"    → Need low-ℓ B-mode experiments (LiteBIRD) to probe this scale")
    print()
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ToE Evaluation on BK18 Chains")
    parser.add_argument(
        "--chains-dir",
        default="/tmp/bk18_chains/BK18_17_BK18lf_freebdust_incP2018_BAO",
        help="Path to extracted BK18 chain directory"
    )
    args = parser.parse_args()

    print("Loading BK18 chains from:", args.chains_dir)
    samples, weights, param_names = load_bk18_chains(args.chains_dir)
    print(f"Loaded {len(samples)} samples, {len(param_names)} parameters")
    print()

    result = evaluate(samples, weights, param_names, TOE_PARAMS)
    print_report(result, TOE_PARAMS)


if __name__ == "__main__":
    main()
