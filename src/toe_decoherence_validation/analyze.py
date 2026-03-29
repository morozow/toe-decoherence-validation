#!/usr/bin/env python3
"""
ToE Error Evaluation — Chain Analysis

Reads MCMC chains and computes posteriors on the KEY quantities:

  Q_obs     = r / (-8 n_t)           — measured consistency ratio
  delta_obs = Q_obs - 1              — deviation from Standard Inflation
  delta_toe = Q_obs - c_s*/(1+2n̄_k) — residual vs ToE prediction

DECISION CRITERIA:
  ┌─────────────────────────────────────────────────────────────────┐
  │ CONFIRMATION of ToE:                                           │
  │   posterior(delta_obs) excludes 0 at >3σ                       │
  │   AND posterior(delta_toe) includes 0 within 2σ                │
  │                                                                │
  │ REFUTATION of ToE:                                             │
  │   posterior(delta_toe) excludes 0 at >3σ                       │
  │                                                                │
  │ INCONCLUSIVE:                                                  │
  │   posterior(delta_obs) includes 0 (can't distinguish ToE/SI)   │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    python toe_error_evaluation/analyze.py [--chains PREFIX]

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ErrorEvalResult:
    """Result of the error evaluation measurement."""
    # Q = r/(-8 n_t)
    Q_mean: float
    Q_std: float
    Q_median: float
    Q_ci_lo: float  # 95% CI lower
    Q_ci_hi: float  # 95% CI upper
    
    # delta_obs = Q - 1 (deviation from SI)
    delta_obs_mean: float
    delta_obs_std: float
    delta_obs_sigma: float  # |mean|/std — significance of deviation from SI
    
    # delta_toe = Q - c_s*/(1+2n̄_k) (residual vs ToE)
    delta_toe_mean: float
    delta_toe_std: float
    delta_toe_sigma: float  # |mean|/std — significance of residual
    
    # ToE prediction
    Q_toe_pred_mean: float
    Q_toe_pred_std: float
    
    # Occupancy
    nbar_k_mean: float
    nbar_k_std: float
    
    # Input parameters
    r_mean: float
    r_std: float
    nt_mean: float
    nt_std: float
    
    # Verdict
    verdict: str
    explanation: str
    
    # Sample counts
    n_total: int
    n_valid: int  # samples where r > 0 and nt < 0


# =============================================================================
# CHAIN LOADING
# =============================================================================

def load_chains(prefix: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load MCMC chains from Cobaya output."""
    try:
        from getdist import loadMCSamples
        samples = loadMCSamples(prefix)
        param_names = {p.name: i for i, p in enumerate(samples.paramNames.names)}
        return samples.samples, param_names
    except ImportError:
        print("GetDist not available, loading raw chains...")
        
        paramnames_file = f"{prefix}.paramnames"
        if not os.path.exists(paramnames_file):
            raise FileNotFoundError(f"Chain files not found: {prefix}.*")
        
        param_names = {}
        with open(paramnames_file) as f:
            for i, line in enumerate(f):
                name = line.split()[0].strip("*")
                param_names[name] = i
        
        chains = []
        for i in range(1, 100):
            chain_file = f"{prefix}.{i}.txt"
            if os.path.exists(chain_file):
                chains.append(np.loadtxt(chain_file))
            else:
                break
        
        if not chains:
            chain_file = f"{prefix}.txt"
            if os.path.exists(chain_file):
                chains.append(np.loadtxt(chain_file))
        
        if not chains:
            raise FileNotFoundError(f"No chain files found: {prefix}.*")
        
        data = np.vstack(chains)
        return data[:, 2:], param_names


# =============================================================================
# CORE MEASUREMENT
# =============================================================================

def measure_consistency_error(
    samples: np.ndarray,
    param_names: Dict[str, int],
    c_s_star: float = 1.0,
) -> ErrorEvalResult:
    """
    THE measurement: compute posteriors on Q, delta_obs, delta_toe.
    
    If Q_obs, delta_obs, delta_toe are already in chains as derived params,
    use them directly. Otherwise compute from r, nt, nbar_k.
    """
    n_total = len(samples)
    
    # Try derived params first (from ToETheoryErrorEval)
    Q_idx = param_names.get("Q_obs")
    delta_obs_idx = param_names.get("delta_obs")
    delta_toe_idx = param_names.get("delta_toe")
    Q_pred_idx = param_names.get("Q_toe_pred")
    nbar_idx = param_names.get("nbar_k_physical")
    r_idx = param_names.get("r")
    nt_idx = param_names.get("nt")
    
    # Get r and nt for statistics
    r_samples = samples[:, r_idx] if r_idx is not None else np.full(n_total, np.nan)
    nt_samples = samples[:, nt_idx] if nt_idx is not None else np.full(n_total, np.nan)
    
    if Q_idx is not None and delta_obs_idx is not None and delta_toe_idx is not None:
        # Use pre-computed derived parameters
        Q_all = samples[:, Q_idx]
        delta_obs_all = samples[:, delta_obs_idx]
        delta_toe_all = samples[:, delta_toe_idx]
        Q_pred_all = samples[:, Q_pred_idx] if Q_pred_idx is not None else np.full(n_total, np.nan)
        nbar_all = samples[:, nbar_idx] if nbar_idx is not None else np.zeros(n_total)
        
        # Filter valid (finite) samples
        valid = np.isfinite(Q_all) & np.isfinite(delta_obs_all)
        
    elif r_idx is not None and nt_idx is not None:
        # Compute from r, nt
        valid = (r_samples > 0) & (nt_samples < 0)
        
        Q_all = np.full(n_total, np.nan)
        Q_all[valid] = r_samples[valid] / (-8.0 * nt_samples[valid])
        
        delta_obs_all = np.full(n_total, np.nan)
        delta_obs_all[valid] = Q_all[valid] - 1.0
        
        nbar_all = samples[:, nbar_idx] if nbar_idx is not None else np.zeros(n_total)
        
        Q_pred_all = np.full(n_total, np.nan)
        Q_pred_all[valid] = c_s_star / (1.0 + 2.0 * nbar_all[valid])
        
        delta_toe_all = np.full(n_total, np.nan)
        delta_toe_all[valid] = Q_all[valid] - Q_pred_all[valid]
    else:
        raise ValueError("Chains must contain either (Q_obs, delta_obs, delta_toe) "
                         "or (r, nt) parameters")
    
    n_valid = int(np.sum(valid))
    
    if n_valid < 10:
        return ErrorEvalResult(
            Q_mean=np.nan, Q_std=np.nan, Q_median=np.nan,
            Q_ci_lo=np.nan, Q_ci_hi=np.nan,
            delta_obs_mean=np.nan, delta_obs_std=np.nan, delta_obs_sigma=np.nan,
            delta_toe_mean=np.nan, delta_toe_std=np.nan, delta_toe_sigma=np.nan,
            Q_toe_pred_mean=np.nan, Q_toe_pred_std=np.nan,
            nbar_k_mean=np.nan, nbar_k_std=np.nan,
            r_mean=np.nan, r_std=np.nan, nt_mean=np.nan, nt_std=np.nan,
            verdict="INSUFFICIENT_DATA",
            explanation=f"Only {n_valid} valid samples (need r > 0, nt < 0)",
            n_total=n_total, n_valid=n_valid,
        )
    
    # Extract valid samples
    Q_v = Q_all[valid]
    dobs_v = delta_obs_all[valid]
    dtoe_v = delta_toe_all[valid]
    Qpred_v = Q_pred_all[valid]
    nbar_v = nbar_all[valid]
    r_v = r_samples[valid] if r_idx is not None else np.full(n_valid, np.nan)
    nt_v = nt_samples[valid] if nt_idx is not None else np.full(n_valid, np.nan)
    
    # Statistics
    Q_mean = float(np.mean(Q_v))
    Q_std = float(np.std(Q_v))
    Q_median = float(np.median(Q_v))
    Q_ci_lo = float(np.percentile(Q_v, 2.5))
    Q_ci_hi = float(np.percentile(Q_v, 97.5))
    
    dobs_mean = float(np.mean(dobs_v))
    dobs_std = float(np.std(dobs_v))
    dobs_sigma = abs(dobs_mean) / dobs_std if dobs_std > 0 else 0.0
    
    dtoe_mean = float(np.mean(dtoe_v))
    dtoe_std = float(np.std(dtoe_v))
    dtoe_sigma = abs(dtoe_mean) / dtoe_std if dtoe_std > 0 else 0.0
    
    Qpred_mean = float(np.nanmean(Qpred_v))
    Qpred_std = float(np.nanstd(Qpred_v))
    
    nbar_mean = float(np.mean(nbar_v))
    nbar_std = float(np.std(nbar_v))
    
    # =========================================================================
    # VERDICT
    # =========================================================================
    if dobs_sigma >= 3.0 and dtoe_sigma < 2.0:
        verdict = "TOE_CONFIRMED"
        explanation = (
            f"Q deviates from 1 at {dobs_sigma:.1f}σ (SI excluded), "
            f"but matches ToE prediction (residual {dtoe_sigma:.1f}σ < 2σ). "
            f"The deviation from standard inflation IS the ToE signal."
        )
    elif dtoe_sigma >= 3.0:
        verdict = "TOE_REFUTED"
        explanation = (
            f"Q does not match ToE prediction: residual {dtoe_sigma:.1f}σ > 3σ. "
            f"ToE consistency relation c_s*/(1+2n̄_k) does not explain the data."
        )
    elif dobs_sigma < 2.0:
        verdict = "INCONCLUSIVE"
        explanation = (
            f"Q = {Q_mean:.3f} ± {Q_std:.3f}, consistent with 1 ({dobs_sigma:.1f}σ). "
            f"Cannot distinguish ToE from SI with current data. "
            f"Need σ(n_t) ~ 10⁻³ for meaningful test (LiteBIRD, CMB-S4)."
        )
    else:
        verdict = "MARGINAL"
        explanation = (
            f"Q deviates from 1 at {dobs_sigma:.1f}σ (2-3σ range). "
            f"ToE residual: {dtoe_sigma:.1f}σ. "
            f"Suggestive but not conclusive. More data needed."
        )
    
    return ErrorEvalResult(
        Q_mean=Q_mean, Q_std=Q_std, Q_median=Q_median,
        Q_ci_lo=Q_ci_lo, Q_ci_hi=Q_ci_hi,
        delta_obs_mean=dobs_mean, delta_obs_std=dobs_std, delta_obs_sigma=dobs_sigma,
        delta_toe_mean=dtoe_mean, delta_toe_std=dtoe_std, delta_toe_sigma=dtoe_sigma,
        Q_toe_pred_mean=Qpred_mean, Q_toe_pred_std=Qpred_std,
        nbar_k_mean=nbar_mean, nbar_k_std=nbar_std,
        r_mean=float(np.mean(r_v)), r_std=float(np.std(r_v)),
        nt_mean=float(np.mean(nt_v)), nt_std=float(np.std(nt_v)),
        verdict=verdict, explanation=explanation,
        n_total=n_total, n_valid=n_valid,
    )


# =============================================================================
# REPORT
# =============================================================================

def print_report(result: ErrorEvalResult):
    """Print the error evaluation report."""
    
    print()
    print("=" * 72)
    print("ToE ERROR EVALUATION — CONSISTENCY RELATION MEASUREMENT")
    print("=" * 72)
    print()
    
    # Section 1: What was measured
    print("1. MEASUREMENT")
    print("-" * 72)
    print()
    print("  Standard Inflation predicts: Q = r/(-8 n_t) = 1")
    print("  ToE predicts:                Q = c_s*/(1+2n̄_k)")
    print()
    print(f"  Samples: {result.n_valid} valid / {result.n_total} total")
    print()
    
    # Section 2: Input parameters
    print("2. INPUT PARAMETERS (from data)")
    print("-" * 72)
    print(f"  r     = {result.r_mean:.6f} ± {result.r_std:.6f}")
    print(f"  n_t   = {result.nt_mean:.6f} ± {result.nt_std:.6f}")
    print(f"  n̄_k   = {result.nbar_k_mean:.2e} ± {result.nbar_k_std:.2e}")
    print()
    
    # Section 3: The measurement
    print("3. CONSISTENCY RATIO Q = r/(-8 n_t)")
    print("-" * 72)
    print(f"  Q_obs    = {result.Q_mean:.4f} ± {result.Q_std:.4f}")
    print(f"  median   = {result.Q_median:.4f}")
    print(f"  95% CI   = [{result.Q_ci_lo:.4f}, {result.Q_ci_hi:.4f}]")
    print()
    
    # Section 4: Deviation from SI
    print("4. DEVIATION FROM STANDARD INFLATION")
    print("-" * 72)
    print(f"  δ_obs = Q - 1 = {result.delta_obs_mean:.4f} ± {result.delta_obs_std:.4f}")
    print(f"  Significance: {result.delta_obs_sigma:.1f}σ")
    print()
    if result.delta_obs_sigma >= 3.0:
        print("  → Q ≠ 1 at >3σ: Standard Inflation consistency VIOLATED")
    elif result.delta_obs_sigma >= 2.0:
        print("  → Q ≠ 1 at 2-3σ: Marginal deviation from SI")
    else:
        print("  → Q consistent with 1: Cannot distinguish from SI")
    print()
    
    # Section 5: Residual vs ToE
    print("5. RESIDUAL vs ToE PREDICTION")
    print("-" * 72)
    print(f"  Q_ToE_pred = c_s*/(1+2n̄_k) = {result.Q_toe_pred_mean:.4f} ± {result.Q_toe_pred_std:.4f}")
    print(f"  δ_ToE = Q - Q_pred = {result.delta_toe_mean:.4f} ± {result.delta_toe_std:.4f}")
    print(f"  Significance: {result.delta_toe_sigma:.1f}σ")
    print()
    if result.delta_toe_sigma < 2.0:
        print("  → ToE prediction MATCHES data (residual < 2σ)")
    elif result.delta_toe_sigma < 3.0:
        print("  → Marginal tension with ToE prediction")
    else:
        print("  → ToE prediction DOES NOT match data (residual > 3σ)")
    print()
    
    # Section 6: Verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print()
    
    symbol = {
        "TOE_CONFIRMED": "✓",
        "TOE_REFUTED": "✗",
        "INCONCLUSIVE": "≈",
        "MARGINAL": "?",
        "INSUFFICIENT_DATA": "⚠",
    }.get(result.verdict, "?")
    
    print(f"  {symbol} {result.verdict}")
    print()
    print(f"  {result.explanation}")
    print()
    
    # Decision table
    print("  Decision criteria:")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ CONFIRMED: δ_obs > 3σ AND δ_ToE < 2σ                   │")
    print("  │ REFUTED:   δ_ToE > 3σ                                   │")
    print("  │ INCONCLUSIVE: δ_obs < 2σ (can't distinguish ToE/SI)     │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ToE Error Evaluation — Analyze MCMC chains"
    )
    parser.add_argument(
        "--chains",
        default="chains_error_eval/toe_error_eval",
        help="Chain file prefix"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    
    print("Loading chains:", args.chains)
    
    try:
        samples, param_names = load_chains(args.chains)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run MCMC first: python toe_error_evaluation/run_mcmc.py")
        sys.exit(1)
    
    print(f"Loaded {len(samples)} samples, {len(param_names)} parameters")
    print()
    
    result = measure_consistency_error(samples, param_names)
    print_report(result)


if __name__ == "__main__":
    main()
