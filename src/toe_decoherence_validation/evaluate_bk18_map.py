#!/usr/bin/env python3
"""
ToE Sensitivity Map on BK18 Chains (NASA BICEP/Keck 2018 + Planck + BAO)

Scans ToE parameter space (k0, eps_H) and builds a sensitivity map
showing where Q(k) deviates from unity and by how much.

ALL PHYSICS imported from toe_decoherence_validation.toe_theory.ToETheoryErrorEval.
No formulas duplicated.

Usage:
    tar xzf modeling/chains_no_data_files.tar.gz -C /tmp/bk18_chains/
    python toe_error_evaluation/evaluate_bk18_map.py

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
# SENSITIVITY MAP: SCAN k0 × eps_H
# =============================================================================

# Scan grid (per companion recommendation)
K0_SCAN = np.array([0.0005, 0.001, 0.002, 0.005, 0.01])
EPSH_SCAN = np.array([0.001, 0.005, 0.01, 0.02, 0.05])

# k values to report Q at
K_REPORT = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])

# Fixed parameters during scan
FIXED_PARAMS = {
    "eta_H": 0.005,
    "s_cs": 0.0,
    "c_s_star": 1.0,
    "Gamma_over_H": 5.0,
}


def compute_sensitivity_map() -> dict:
    """
    Scan k0 × eps_H parameter space.
    For each (k0, eps_H), compute Q(k) on K_REPORT grid.
    
    Returns dict with arrays for plotting.
    All physics via ToETheoryErrorEval._compute_ms_on_sparse_grid.
    """
    theory = ToETheoryErrorEval.__new__(ToETheoryErrorEval)
    theory.k_pivot = K_PIVOT
    theory.c_s_star = FIXED_PARAMS["c_s_star"]
    theory.n_k_ms = 30

    n_k0 = len(K0_SCAN)
    n_eps = len(EPSH_SCAN)
    n_k = len(K_REPORT)

    # Output arrays
    Q_map = np.zeros((n_k0, n_eps, n_k))       # Q(k) for each (k0, eps_H)
    D_map = np.zeros((n_k0, n_eps, n_k))       # 1 - Q(k)
    nbar_map = np.zeros((n_k0, n_eps, n_k))    # nbar_k
    max_D = np.zeros((n_k0, n_eps))             # max deviation
    Q_at_k0 = np.zeros((n_k0, n_eps))           # Q at k = k0

    print(f"Scanning {n_k0} × {n_eps} = {n_k0 * n_eps} parameter combinations...")
    print(f"  k0 values: {K0_SCAN}")
    print(f"  eps_H values: {EPSH_SCAN}")
    print()

    for i, k0 in enumerate(K0_SCAN):
        for j, eps_H in enumerate(EPSH_SCAN):
            eta_0 = -1.0 / k0

            ms_results = theory._compute_ms_on_sparse_grid(
                k_sparse=K_REPORT,
                eta_0=eta_0,
                c_s=FIXED_PARAMS["c_s_star"],
                eps_H=eps_H,
                eta_H=FIXED_PARAMS["eta_H"],
                s=FIXED_PARAMS["s_cs"],
                Gamma_over_H=FIXED_PARAMS["Gamma_over_H"],
            )

            nbar_k = ms_results["nbar_k"]
            Q_k = FIXED_PARAMS["c_s_star"] / (1.0 + 2.0 * nbar_k)
            D_k = 1.0 - Q_k

            Q_map[i, j, :] = Q_k
            D_map[i, j, :] = D_k
            nbar_map[i, j, :] = nbar_k
            max_D[i, j] = np.max(D_k)

            # Q at k = k0 (nearest in K_REPORT)
            k0_idx = np.argmin(np.abs(K_REPORT - k0))
            Q_at_k0[i, j] = Q_k[k0_idx]

            print(f"  k0={k0:.4f}, eps_H={eps_H:.3f}: max(1-Q)={max_D[i,j]:.4f}, Q(k0)={Q_at_k0[i,j]:.4f}")

    return {
        "k0_scan": K0_SCAN,
        "eps_H_scan": EPSH_SCAN,
        "k_report": K_REPORT,
        "Q_map": Q_map,
        "D_map": D_map,
        "nbar_map": nbar_map,
        "max_D": max_D,
        "Q_at_k0": Q_at_k0,
    }


def plot_sensitivity_map(scan: dict, output_dir: str = "plots"):
    """Generate sensitivity map plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    k0_scan = scan["k0_scan"]
    eps_H_scan = scan["eps_H_scan"]
    max_D = scan["max_D"]
    Q_at_k0 = scan["Q_at_k0"]

    # --- Plot 1: max(1-Q) heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        max_D.T * 100,  # percent
        origin="lower",
        aspect="auto",
        extent=[
            np.log10(k0_scan[0]), np.log10(k0_scan[-1]),
            np.log10(eps_H_scan[0]), np.log10(eps_H_scan[-1]),
        ],
        cmap="inferno",
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("max(1 - Q) [%]", fontsize=12)
    ax.set_xlabel("log10(k0 [Mpc⁻¹])", fontsize=12)
    ax.set_ylabel("log10(eps_H)", fontsize=12)
    ax.set_title("ToE Sensitivity Map: Maximum Deviation from Q=1", fontsize=13)

    # Add text annotations
    for i, k0 in enumerate(k0_scan):
        for j, eps_H in enumerate(eps_H_scan):
            x = np.log10(k0)
            y = np.log10(eps_H)
            val = max_D[i, j] * 100
            color = "white" if val > 15 else "black"
            ax.text(x, y, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    path1 = os.path.join(output_dir, "sensitivity_max_deviation.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- Plot 2: Q(k) curves for each k0 (at manuscript eps_H=0.01) ---
    eps_idx = np.argmin(np.abs(eps_H_scan - 0.01))
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, k0 in enumerate(k0_scan):
        Q_k = scan["Q_map"][i, eps_idx, :]
        label = f"k0={k0:.4f}"
        ax.semilogx(scan["k_report"], Q_k, "o-", label=label, markersize=5)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="SI (Q=1)")
    ax.set_xlabel("k [Mpc⁻¹]", fontsize=12)
    ax.set_ylabel("Q_ToE(k) = c_s* / (1 + 2 n̄_k)", fontsize=12)
    ax.set_title(f"ToE Consistency Ratio Q(k) at eps_H={eps_H_scan[eps_idx]:.3f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path2 = os.path.join(output_dir, "Q_vs_k_by_k0.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Plot 3: Q(k) curves for each eps_H (at manuscript k0=0.002) ---
    k0_idx = np.argmin(np.abs(k0_scan - 0.002))
    fig, ax = plt.subplots(figsize=(9, 6))

    for j, eps_H in enumerate(eps_H_scan):
        Q_k = scan["Q_map"][k0_idx, j, :]
        label = f"eps_H={eps_H:.3f}"
        ax.semilogx(scan["k_report"], Q_k, "s-", label=label, markersize=5)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="SI (Q=1)")
    ax.set_xlabel("k [Mpc⁻¹]", fontsize=12)
    ax.set_ylabel("Q_ToE(k) = c_s* / (1 + 2 n̄_k)", fontsize=12)
    ax.set_title(f"ToE Consistency Ratio Q(k) at k0={k0_scan[k0_idx]:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path3 = os.path.join(output_dir, "Q_vs_k_by_epsH.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # --- Plot 4: n̄_k heatmap at k=0.002 (k0 scale) ---
    k_idx = np.argmin(np.abs(scan["k_report"] - 0.002))
    nbar_at_k0 = scan["nbar_map"][:, :, k_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        nbar_at_k0.T,
        origin="lower",
        aspect="auto",
        extent=[
            np.log10(k0_scan[0]), np.log10(k0_scan[-1]),
            np.log10(eps_H_scan[0]), np.log10(eps_H_scan[-1]),
        ],
        cmap="viridis",
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("n̄_k at k=0.002", fontsize=12)
    ax.set_xlabel("log10(k0 [Mpc⁻¹])", fontsize=12)
    ax.set_ylabel("log10(eps_H)", fontsize=12)
    ax.set_title("Occupancy n̄_k at k=0.002 Mpc⁻¹", fontsize=13)

    for i, k0 in enumerate(k0_scan):
        for j, eps_H in enumerate(eps_H_scan):
            x = np.log10(k0)
            y = np.log10(eps_H)
            val = nbar_at_k0[i, j]
            ax.text(x, y, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.05 else "black")

    fig.tight_layout()
    path4 = os.path.join(output_dir, "nbar_k_heatmap.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path4}")


def print_sensitivity_table(scan: dict):
    """Print sensitivity scan results as table."""
    print()
    print("=" * 80)
    print("SENSITIVITY MAP: k0 × eps_H → max(1-Q)")
    print("=" * 80)
    print()
    print(f"Fixed: eta_H={FIXED_PARAMS['eta_H']}, s={FIXED_PARAMS['s_cs']}, "
          f"c_s*={FIXED_PARAMS['c_s_star']}, Gamma/H={FIXED_PARAMS['Gamma_over_H']}")
    print()

    # Header
    header = f"{'k0 \\\\ eps_H':<12}"
    for eps_H in scan["eps_H_scan"]:
        header += f" {eps_H:<10.3f}"
    print(header)
    print("-" * (12 + 11 * len(scan["eps_H_scan"])))

    # Rows
    for i, k0 in enumerate(scan["k0_scan"]):
        row = f"{k0:<12.4f}"
        for j in range(len(scan["eps_H_scan"])):
            val = scan["max_D"][i, j] * 100
            row += f" {val:<10.1f}"
        print(row)

    print()
    print("Values: max(1-Q) in percent across all k")
    print()

    # Robustness classification
    print("ROBUSTNESS CLASSIFICATION:")
    print("-" * 60)
    strong = []
    moderate = []
    weak = []
    for i, k0 in enumerate(scan["k0_scan"]):
        for j, eps_H in enumerate(scan["eps_H_scan"]):
            d = scan["max_D"][i, j]
            label = f"k0={k0:.4f}, eps_H={eps_H:.3f}"
            if d > 0.05:
                strong.append((label, d))
            elif d > 0.01:
                moderate.append((label, d))
            else:
                weak.append((label, d))

    print(f"  Strong effect (>5%):    {len(strong)} / {len(scan['k0_scan'])*len(scan['eps_H_scan'])}")
    for label, d in strong[:5]:
        print(f"    {label}: {d*100:.1f}%")
    print(f"  Moderate effect (1-5%): {len(moderate)}")
    print(f"  Weak effect (<1%):      {len(weak)}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ToE Sensitivity Map on BK18 Chains")
    parser.add_argument(
        "--chains-dir",
        default="/tmp/bk18_chains/BK18_17_BK18lf_freebdust_incP2018_BAO",
        help="Path to extracted BK18 chain directory"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    # Load BK18 chains for r statistics
    print("Loading BK18 chains from:", args.chains_dir)
    samples, weights, param_names = load_bk18_chains(args.chains_dir)
    print(f"Loaded {len(samples)} samples, {len(param_names)} parameters")
    print()

    # Single-point evaluation at manuscript values
    print("=" * 80)
    print("SINGLE-POINT EVALUATION (manuscript parameters)")
    print("=" * 80)
    result = evaluate(samples, weights, param_names, TOE_PARAMS)
    print_report(result, TOE_PARAMS)

    # Sensitivity scan
    print()
    scan = compute_sensitivity_map()
    print_sensitivity_table(scan)

    # Plots
    if not args.no_plots:
        print("Generating plots...")
        plot_sensitivity_map(scan)
        print("Done.")


if __name__ == "__main__":
    main()
