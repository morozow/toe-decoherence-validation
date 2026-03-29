#!/usr/bin/env python3
"""
ToE Joint Analysis: Three Channels from One Mechanism

Shows that ONE set of parameters simultaneously explains:
  1. Q(k) < 1 in IR (consistency relation deviation)
  2. Ring-down amplitude A_ring(k) and phase phi_k(k)
  3. f_NL suppression via (1+2*nbar_k)

And Q = 1 at pivot (null test).

ALL PHYSICS from toe_decoherence_validation.toe_theory.ToETheoryErrorEval.

Usage:
    python toe_error_evaluation/joint_analysis.py

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np


from toe_decoherence_validation.toe_theory import ToETheoryErrorEval
from toe_decoherence_validation.evaluate_bk18_map import load_bk18_chains

K_PIVOT = 0.05
K_REPORT = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
K_LOW_MASK = K_REPORT <= 0.002  # IR region where ToE effect lives


# =============================================================================
# SCAN GRID
# =============================================================================

K0_SCAN = np.array([0.0005, 0.001, 0.002, 0.005, 0.01])
EPSH_SCAN = np.array([0.001, 0.005, 0.01, 0.02, 0.05])
GAMMA_SCAN = np.array([1.0, 3.0, 5.0, 10.0, 20.0])

FIXED = {"eta_H": 0.005, "s_cs": 0.0, "c_s_star": 1.0}


# =============================================================================
# JOINT RESULT PER PARAMETER POINT
# =============================================================================

@dataclass
class JointPoint:
    """Result of joint analysis at one parameter point."""
    k0: float
    eps_H: float
    Gamma_over_H: float

    # Channel 1: Q(k)
    Q_grid: np.ndarray        # Q at each k in K_REPORT
    D_low_k_mean: float       # mean(1-Q) for k <= 0.002
    D_low_k_max: float        # max(1-Q) for k <= 0.002
    Q_pivot: float             # Q at k=0.05

    # Channel 2: Ring-down
    A_ring_grid: np.ndarray   # A_ring at each k
    phi_k_grid: np.ndarray    # phi_k at each k
    A_ring_rms_low_k: float   # RMS of A_ring for k <= 0.002
    phi_k_smooth: bool        # is phi_k smooth (weighted metric)?
    ch2_status: str            # "pass", "fail", "undetermined"
    ch2_phase_score: float     # weighted smoothness [0,1]
    ch2_weighted_abs_delta: float
    ch2_max_abs_delta: float
    ch2_significant_points: int

    # Channel 3: f_NL suppression
    R_fNL_grid: np.ndarray    # 1/(1+2*nbar_k) at each k
    R_fNL_low_k_mean: float   # mean R_fNL for k <= 0.002

    # Joint assessment
    ch1_pass: bool  # Q < 1 in IR, Q = 1 at pivot
    ch2_pass: bool  # A_ring > 0, phi_k smooth (weighted)
    ch3_pass: bool  # R_fNL < 1 in IR
    all_pass: bool  # all three channels pass


# =============================================================================
# COMPUTE ONE POINT
# =============================================================================

def compute_joint_point(
    k0: float,
    eps_H: float,
    Gamma_over_H: float,
    theory: ToETheoryErrorEval,
) -> JointPoint:
    """
    Compute all three channels from ONE MS solver call.
    """
    eta_0 = -1.0 / k0
    c_s = FIXED["c_s_star"]

    # ONE call to MS solver — all physics from here
    ms = theory._compute_ms_on_sparse_grid(
        k_sparse=K_REPORT,
        eta_0=eta_0,
        c_s=c_s,
        eps_H=eps_H,
        eta_H=FIXED["eta_H"],
        s=FIXED["s_cs"],
        Gamma_over_H=Gamma_over_H,
    )

    nbar_k = ms["nbar_k"]
    phi_k = ms["phi_k"]
    A_ring = ms["A_ring"]

    # --- Channel 1: Q(k) ---
    Q_grid = c_s / (1.0 + 2.0 * nbar_k)
    D_grid = 1.0 - Q_grid

    D_low_k = D_grid[K_LOW_MASK]
    D_low_k_mean = float(np.mean(D_low_k)) if len(D_low_k) > 0 else 0.0
    D_low_k_max = float(np.max(D_low_k)) if len(D_low_k) > 0 else 0.0

    pivot_idx = np.argmin(np.abs(K_REPORT - K_PIVOT))
    Q_pivot = float(Q_grid[pivot_idx])

    ch1_pass = (D_low_k_mean >= 0.05) and (abs(1.0 - Q_pivot) < 1e-4)

    # --- Channel 2: Ring-down (complex-phase, weighted, with undetermined state) ---
    nbar_arr = np.asarray(nbar_k, dtype=float)
    phi_arr = np.asarray(phi_k, dtype=float)
    A_ring_arr = np.asarray(A_ring, dtype=float)
    k_arr = K_REPORT

    # Observational weight
    obs_weight = A_ring_arr * (1.0 + 2.0 * nbar_arr)
    w_max = float(np.max(obs_weight)) if obs_weight.size else 0.0

    # IR amplitude criterion
    ir_mask = k_arr <= 0.002
    if np.any(ir_mask):
        A_ring_rms = float(np.sqrt(np.mean(A_ring_arr[ir_mask] ** 2)))
    else:
        A_ring_rms = float(np.sqrt(np.mean(A_ring_arr ** 2))) if A_ring_arr.size else 0.0
    ch2_amp_pass = A_ring_rms > 1e-6

    # Significant points for phase diagnostics
    significance_frac = 0.01
    if w_max > 0.0:
        sig_mask = obs_weight > (significance_frac * w_max)
    else:
        sig_mask = np.zeros_like(obs_weight, dtype=bool)

    sig_idx = np.where(sig_mask)[0]

    if sig_idx.size < 2:
        ch2_status = "undetermined"
        ch2_pass = False
        phi_k_smooth = False
        ch2_phase_score = np.nan
        ch2_weighted_abs_delta = np.nan
        ch2_max_abs_delta = np.nan
    else:
        # Unit complex phase vector u_k = exp(i * phi_k)
        u = np.exp(1j * phi_arr)
        u_sig = u[sig_idx]

        # Local phase step: Delta_phi_k = arg(u_{k+1} * conj(u_k))
        delta_phi = np.angle(u_sig[1:] * np.conj(u_sig[:-1]))

        # Edge weights (phase-step relevance)
        w_sig = obs_weight[sig_idx]
        edge_w = 0.5 * (w_sig[1:] + w_sig[:-1])
        edge_w_sum = float(np.sum(edge_w))
        if edge_w_sum > 0.0:
            edge_w = edge_w / edge_w_sum
        else:
            edge_w = np.ones_like(edge_w) / max(len(edge_w), 1)

        abs_delta = np.abs(delta_phi)
        ch2_weighted_abs_delta = float(np.sum(edge_w * abs_delta))
        ch2_max_abs_delta = float(np.max(abs_delta))

        # Weighted smoothness metric in [0, 1]
        ch2_phase_score = float(np.clip(1.0 - ch2_weighted_abs_delta / np.pi, 0.0, 1.0))

        # Phase pass criterion
        phi_k_smooth = ch2_weighted_abs_delta < (np.pi / 2.0)

        ch2_pass = bool(ch2_amp_pass and phi_k_smooth)
        ch2_status = "pass" if ch2_pass else "fail"

    ch2_significant_points = int(sig_idx.size)

    # --- Channel 3: f_NL suppression ---
    # R_fNL = f_NL_ToE / f_NL_SI = 1/(1+2*nbar_k)
    R_fNL = 1.0 / (1.0 + 2.0 * nbar_k)
    R_fNL_low_k = R_fNL[K_LOW_MASK]
    R_fNL_low_k_mean = float(np.mean(R_fNL_low_k)) if len(R_fNL_low_k) > 0 else 1.0

    ch3_pass = (R_fNL_low_k_mean < 0.95)

    return JointPoint(
        k0=k0, eps_H=eps_H, Gamma_over_H=Gamma_over_H,
        Q_grid=Q_grid, D_low_k_mean=D_low_k_mean, D_low_k_max=D_low_k_max,
        Q_pivot=Q_pivot,
        A_ring_grid=A_ring, phi_k_grid=phi_k,
        A_ring_rms_low_k=A_ring_rms, phi_k_smooth=phi_k_smooth,
        ch2_status=ch2_status,
        ch2_phase_score=ch2_phase_score if not np.isnan(ch2_phase_score) else 0.0,
        ch2_weighted_abs_delta=ch2_weighted_abs_delta if not np.isnan(ch2_weighted_abs_delta) else 0.0,
        ch2_max_abs_delta=ch2_max_abs_delta if not np.isnan(ch2_max_abs_delta) else 0.0,
        ch2_significant_points=ch2_significant_points,
        R_fNL_grid=R_fNL, R_fNL_low_k_mean=R_fNL_low_k_mean,
        ch1_pass=ch1_pass, ch2_pass=ch2_pass, ch3_pass=ch3_pass,
        all_pass=(ch1_pass and ch2_pass and ch3_pass),
    )


# =============================================================================
# FULL SCAN
# =============================================================================

def run_joint_scan() -> List[JointPoint]:
    """Scan (k0, eps_H, Gamma/H) and compute joint analysis at each point."""
    theory = ToETheoryErrorEval.__new__(ToETheoryErrorEval)
    theory.k_pivot = K_PIVOT
    theory.c_s_star = FIXED["c_s_star"]
    theory.n_k_ms = 30

    total = len(K0_SCAN) * len(EPSH_SCAN) * len(GAMMA_SCAN)
    print(f"Joint scan: {len(K0_SCAN)} × {len(EPSH_SCAN)} × {len(GAMMA_SCAN)} = {total} points")
    print(f"  k0: {K0_SCAN}")
    print(f"  eps_H: {EPSH_SCAN}")
    print(f"  Gamma/H: {GAMMA_SCAN}")
    print()

    results = []
    for k0 in K0_SCAN:
        for eps_H in EPSH_SCAN:
            for gamma in GAMMA_SCAN:
                pt = compute_joint_point(k0, eps_H, gamma, theory)
                results.append(pt)

    return results


# =============================================================================
# REPORT
# =============================================================================

def print_joint_report(results: List[JointPoint], ns_mean: float = 0.965):
    """Print joint analysis report."""
    total = len(results)
    n_all = sum(1 for r in results if r.all_pass)
    n_ch1 = sum(1 for r in results if r.ch1_pass)
    n_ch2 = sum(1 for r in results if r.ch2_pass)
    n_ch3 = sum(1 for r in results if r.ch3_pass)

    print()
    print("=" * 80)
    print("JOINT ANALYSIS: THREE CHANNELS FROM ONE MECHANISM")
    print("=" * 80)
    print()
    print("Channels tested (all from single MS solver call per point):")
    print("  1. Q(k) < 1 in IR + Q = 1 at pivot")
    print("  2. Ring-down: A_ring > 0, phi_k smooth")
    print("  3. f_NL suppression: R_fNL < 0.95 in IR")
    print()

    print(f"Total parameter points: {total}")
    print(f"  Channel 1 pass (Q): {n_ch1} / {total} ({n_ch1/total*100:.0f}%)")
    print(f"  Channel 2 pass (ring-down): {n_ch2} / {total} ({n_ch2/total*100:.0f}%)")
    print(f"  Channel 3 pass (f_NL): {n_ch3} / {total} ({n_ch3/total*100:.0f}%)")
    print(f"  ALL THREE pass: {n_all} / {total} ({n_all/total*100:.0f}%)")
    print()

    # Show feasible region
    print("FEASIBLE REGION (all three channels pass):")
    print("-" * 80)
    print(f"{'k0':<10} {'eps_H':<10} {'Gamma/H':<10} {'D_IR_mean':<12} "
          f"{'Q_pivot':<10} {'A_ring_rms':<12} {'R_fNL_IR':<10}")
    print("-" * 80)

    for r in results:
        if r.all_pass:
            print(f"{r.k0:<10.4f} {r.eps_H:<10.3f} {r.Gamma_over_H:<10.1f} "
                  f"{r.D_low_k_mean:<12.4f} {r.Q_pivot:<10.8f} "
                  f"{r.A_ring_rms_low_k:<12.6f} {r.R_fNL_low_k_mean:<10.4f}")

    print()

    # Show manuscript point specifically
    print("MANUSCRIPT POINT (k0=0.002, eps_H=0.01, Gamma/H=5.0):")
    print("-" * 80)
    manuscript = [r for r in results
                  if abs(r.k0 - 0.002) < 1e-6
                  and abs(r.eps_H - 0.01) < 1e-6
                  and abs(r.Gamma_over_H - 5.0) < 0.1]

    if manuscript:
        r = manuscript[0]
        print()
        print("  Channel 1 — Consistency Relation Q(k):")
        for i, k in enumerate(K_REPORT):
            marker = ""
            if abs(k - K_PIVOT) < 0.001:
                marker = " ← PIVOT (null test)"
            elif abs(k - r.k0) < 0.0001:
                marker = " ← k0"
            print(f"    k={k:.4f}: Q={r.Q_grid[i]:.6f}, 1-Q={1-r.Q_grid[i]:.6e}{marker}")
        print(f"    → D_IR_mean = {r.D_low_k_mean:.4f} ({r.D_low_k_mean*100:.1f}%)")
        print(f"    → Q_pivot = {r.Q_pivot:.10f}")
        print(f"    → PASS: {r.ch1_pass}")
        print()

        print("  Channel 2 — Ring-down (complex-phase weighted metric):")
        for i, k in enumerate(K_REPORT):
            print(f"    k={k:.4f}: A_ring={r.A_ring_grid[i]:.6e}, "
                  f"phi_k={r.phi_k_grid[i]:.4f} rad")
        print(f"    → A_ring_rms(IR) = {r.A_ring_rms_low_k:.6e}")
        print(f"    → Significant points: {r.ch2_significant_points}")
        print(f"    → Weighted |Δφ|: {r.ch2_weighted_abs_delta:.4f} rad")
        print(f"    → Max |Δφ|: {r.ch2_max_abs_delta:.4f} rad")
        print(f"    → Phase score: {r.ch2_phase_score:.4f}")
        print(f"    → Status: {r.ch2_status}")
        print(f"    → PASS: {r.ch2_pass}")
        print()

        print("  Channel 3 — f_NL suppression:")
        # f_NL = (5/12)(1-ns)/(1+2*nbar_k)
        fNL_SI = (5.0 / 12.0) * (1.0 - ns_mean)
        for i, k in enumerate(K_REPORT):
            fNL_ToE = fNL_SI * r.R_fNL_grid[i]
            print(f"    k={k:.4f}: R_fNL={r.R_fNL_grid[i]:.6f}, "
                  f"f_NL_ToE={fNL_ToE:.6f} (SI: {fNL_SI:.6f})")
        print(f"    → R_fNL_IR_mean = {r.R_fNL_low_k_mean:.4f}")
        print(f"    → PASS: {r.ch3_pass}")
        print()

        print(f"  JOINT VERDICT: {'ALL THREE PASS' if r.all_pass else 'FAIL'}")
    else:
        print("  (manuscript point not in scan grid)")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    if n_all > 0:
        pct = n_all / total * 100
        print(f"  {n_all} / {total} ({pct:.0f}%) parameter points satisfy ALL THREE channels")
        print(f"  simultaneously from ONE MS solver call.")
        print()
        print(f"  This demonstrates that the ToE consistency relation deviation,")
        print(f"  ring-down oscillations, and f_NL suppression are NOT independent")
        print(f"  fits — they are THREE CONSEQUENCES of ONE mechanism (decoherence")
        print(f"  at eta_0), computed from ONE set of parameters.")
        print()
        print(f"  Q = 1 remains the limiting case when nbar_k → 0 (pivot scale).")
    else:
        print("  No parameter points satisfy all three channels simultaneously.")
    print()
    print("=" * 80)


# =============================================================================
# PLOTS
# =============================================================================

def plot_joint_results(results: List[JointPoint], output_dir: str = "plots"):
    """Generate joint analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Feasible region heatmap (k0 × eps_H, at Gamma/H=5) ---
    gamma_target = 5.0
    n_k0 = len(K0_SCAN)
    n_eps = len(EPSH_SCAN)
    joint_map = np.zeros((n_k0, n_eps))

    for r in results:
        if abs(r.Gamma_over_H - gamma_target) < 0.1:
            i = np.argmin(np.abs(K0_SCAN - r.k0))
            j = np.argmin(np.abs(EPSH_SCAN - r.eps_H))
            score = int(r.ch1_pass) + int(r.ch2_pass) + int(r.ch3_pass)
            joint_map[i, j] = score

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        joint_map.T,
        origin="lower", aspect="auto",
        extent=[np.log10(K0_SCAN[0]), np.log10(K0_SCAN[-1]),
                np.log10(EPSH_SCAN[0]), np.log10(EPSH_SCAN[-1])],
        cmap="RdYlGn", vmin=0, vmax=3,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cb.set_label("Channels passing (0-3)", fontsize=12)
    cb.set_ticklabels(["0", "1", "2", "3 (ALL)"])
    ax.set_xlabel("log10(k0 [Mpc⁻¹])", fontsize=12)
    ax.set_ylabel("log10(eps_H)", fontsize=12)
    ax.set_title(f"Joint Feasibility: 3 Channels at Gamma/H={gamma_target}", fontsize=13)

    for i, k0 in enumerate(K0_SCAN):
        for j, eps_H in enumerate(EPSH_SCAN):
            x, y = np.log10(k0), np.log10(eps_H)
            val = int(joint_map[i, j])
            color = "white" if val < 2 else "black"
            ax.text(x, y, str(val), ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    fig.tight_layout()
    p = os.path.join(output_dir, "joint_feasibility_map.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Plot 2: Three channels at manuscript point ---
    manuscript = [r for r in results
                  if abs(r.k0 - 0.002) < 1e-6
                  and abs(r.eps_H - 0.01) < 1e-6
                  and abs(r.Gamma_over_H - 5.0) < 0.1]

    if manuscript:
        r = manuscript[0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Channel 1: Q(k)
        ax = axes[0]
        ax.semilogx(K_REPORT, r.Q_grid, "o-", color="C0", markersize=6)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("k [Mpc⁻¹]")
        ax.set_ylabel("Q(k)")
        ax.set_title("Ch.1: Consistency Ratio")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)

        # Channel 2: Ring-down
        ax = axes[1]
        ax.semilogx(K_REPORT, r.A_ring_grid, "s-", color="C1", markersize=6)
        ax.set_xlabel("k [Mpc⁻¹]")
        ax.set_ylabel("A_ring(k)")
        ax.set_title("Ch.2: Ring-down Amplitude")
        ax.grid(True, alpha=0.3)

        # Channel 3: f_NL suppression
        ax = axes[2]
        ax.semilogx(K_REPORT, r.R_fNL_grid, "^-", color="C2", markersize=6)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("k [Mpc⁻¹]")
        ax.set_ylabel("R_fNL = f_NL(ToE)/f_NL(SI)")
        ax.set_title("Ch.3: f_NL Suppression")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Three Channels at Manuscript Point (k0=0.002, eps_H=0.01, Gamma/H=5)",
                     fontsize=13)
        fig.tight_layout()
        p = os.path.join(output_dir, "joint_three_channels.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p}")

    # --- Plot 3: Gamma/H effect on ring-down (at manuscript k0, eps_H) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for gamma in GAMMA_SCAN:
        pts = [r for r in results
               if abs(r.k0 - 0.002) < 1e-6
               and abs(r.eps_H - 0.01) < 1e-6
               and abs(r.Gamma_over_H - gamma) < 0.1]
        if pts:
            r = pts[0]
            ax.semilogx(K_REPORT, r.A_ring_grid, "o-",
                        label=f"Gamma/H={gamma:.0f}", markersize=5)

    ax.set_xlabel("k [Mpc⁻¹]", fontsize=12)
    ax.set_ylabel("A_ring(k)", fontsize=12)
    ax.set_title("Ring-down Amplitude vs Gamma/H (k0=0.002, eps_H=0.01)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, "ringdown_vs_gamma.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ToE Joint Analysis: Three Channels")
    parser.add_argument("--chains-dir",
                        default="/tmp/bk18_chains/BK18_17_BK18lf_freebdust_incP2018_BAO",
                        help="BK18 chains dir (for ns)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # Get ns from BK18 chains
    ns_mean = 0.965  # default
    try:
        samples, weights, param_names = load_bk18_chains(args.chains_dir)
        ns_idx = param_names.get("ns")
        if ns_idx is not None:
            ns_mean = float(np.average(samples[:, ns_idx], weights=weights))
        print(f"BK18 ns = {ns_mean:.6f}")
    except Exception as e:
        print(f"Could not load BK18 chains ({e}), using ns={ns_mean}")
    print()

    # Run joint scan
    results = run_joint_scan()

    # Report
    print_joint_report(results, ns_mean=ns_mean)

    # Plots
    if not args.no_plots:
        print("Generating plots...")
        plot_joint_results(results)
        print("Done.")


if __name__ == "__main__":
    main()
