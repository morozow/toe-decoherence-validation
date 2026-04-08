"""
Unified physics module for all ToE experiments.

ALL physics imported from the verified implementation:
  toe_error_evaluation/src/toe_decoherence_validation/

NOTHING from modeling/physics_model.py — that module is DEPRECATED.

This module re-exports everything experiments need:
  1. MS solver functions from mukhanov_sasaki.py
  2. ToETheoryErrorEval class and TOE_PARAMS from toe_theory.py / evaluate_bk18.py
  3. BK18 chain loading from evaluate_bk18.py
  4. Trivial manuscript formulas (w_ent, conservation_residual, z²) — ≤5 lines each

Reference: manuscript sec03, sec06, sec09, sec10, sec11, sec12, sec13
"""

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_MS_SRC = os.path.join(_PROJECT_ROOT, "toe_error_evaluation", "src")
_BK18_DEFAULT = os.path.join(
    _PROJECT_ROOT, "manuscript",
    "BK18_17_BK18lf_freebdust_incP2018_BAO",
    "BK18_17_BK18lf_freebdust_incP2018_BAO",
)

if _MS_SRC not in sys.path:
    sys.path.insert(0, _MS_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# 1. RE-EXPORTS from toe_decoherence_validation (read-only)
# ===================================================================

from toe_decoherence_validation.mukhanov_sasaki import (  # noqa: E402
    ModeResult,
    compute_mode_result,
    ringdown_from_bogoliubov,
    power_spectrum_with_ringdown,
    compute_spectrum_array,
    nu_from_slow_roll,
    bogoliubov_from_matching,
    solve_ms_to_eta0,
)

from toe_decoherence_validation.toe_theory import (  # noqa: E402
    ToETheoryErrorEval,
)

from toe_decoherence_validation.evaluate_bk18 import (  # noqa: E402
    TOE_PARAMS,
    K_PIVOT,
    BK18EvalResult,
    load_bk18_chains as _load_bk18_chains_canonical,
    evaluate as evaluate_bk18,
)


# ===================================================================
# 2. CANONICAL PARAMETERS from manuscript (sec13)
# ===================================================================

# Re-export TOE_PARAMS for convenience (same as evaluate_bk18.py)
TOE_K0 = TOE_PARAMS["k0"]              # 0.002 Mpc⁻¹
TOE_EPS_H = TOE_PARAMS["eps_H"]        # 0.01
TOE_ETA_H = TOE_PARAMS["eta_H"]        # 0.005
TOE_S_CS = TOE_PARAMS["s_cs"]          # 0.0
TOE_C_S_STAR = TOE_PARAMS["c_s_star"]  # 1.0
TOE_GAMMA_OVER_H = TOE_PARAMS["Gamma_over_H"]  # 5.0
TOE_ETA_0 = -1.0 / TOE_K0             # -500.0 Mpc

# SM content from manuscript (sec11, subsec:SM-kappa)
SM_CONTENT = {
    "N_s": 4,
    "N_w": 45,
    "N_v": 12,
    "a_SM": 1991.0 / 720.0,
    "c_SM": 209.0 / 60.0,
    "b1": 41.0 / 10.0,
    "b2": -19.0 / 6.0,
    "b3": -7.0,
}

# Planck 2018 posteriors (sec13, eq:posteriors)
PLANCK_POSTERIORS = {
    "n_s": {"value": 0.965, "error": 0.004},
    "A_s": {"value": 2.10e-9, "error": 0.03e-9},
    "r_upper_95CL": 0.036,
    "H0": {"value": 67.4, "error": 0.5},
    "alpha_2": {"value": -0.34, "error": 0.20},
    "alpha_3": {"value": 0.98, "error": 0.25},
}


# Default Cobaya-format parameters for run_toe_calculation()
# Combines BK18 MAP values + TOE_PARAMS
DEFAULT_COBAYA_PARAMS = {
    # Standard cosmological (from BK18+Planck+BAO MAP)
    "ombh2": 0.02236,
    "omch2": 0.1200,
    "theta_s_1e2": 1.0419,   # 100×θ_s (Cobaya convention)
    "tau": 0.054,
    "logA": 3.044,
    "ns": 0.965,
    # Tensor (from BK18)
    "r": 0.01,
    "nt": -0.00125,
    # IR feature (sec13)
    "k0": TOE_K0,            # 0.002 Mpc⁻¹
    "A_IR": 0.0,
    "sigma_IR": 1.0,
    # Higher-curvature (sec13, eq:posteriors)
    "alpha2": -0.30,          # ghost-free: α₂ + α₃/3 > 0
    "alpha3": 1.0,
    # ToE-specific (from TOE_PARAMS)
    "eps_H": TOE_EPS_H,      # 0.01
    "eta_H": TOE_ETA_H,      # 0.005
    "s_cs": TOE_S_CS,        # 0.0
    "Gamma_over_H": TOE_GAMMA_OVER_H,  # 5.0
}

# Parameter name mapping: BK18 chains ↔ Cobaya ↔ spec.yaml
PARAM_NAME_MAP = {
    # BK18 chain name → Cobaya name
    "omegabh2": "ombh2",
    "omegach2": "omch2",
    "theta": "theta_s_1e2",  # BK18 has 100×θ, Cobaya expects same
    "tau": "tau",
    "logA": "logA",
    "ns": "ns",
    "r": "r",
}


# ===================================================================
# 3. MS SOLVER ACCESS — same pattern as evaluate_bk18.py
# ===================================================================

_TOE_THEORY_MS = None


def get_toe_theory():
    """
    Get ToETheoryErrorEval instance for MS solver access.

    Same pattern as evaluate_bk18.py: __new__ without Cobaya init.
    """
    global _TOE_THEORY_MS
    if _TOE_THEORY_MS is not None:
        return _TOE_THEORY_MS
    theory = ToETheoryErrorEval.__new__(ToETheoryErrorEval)
    theory.k_pivot = K_PIVOT
    theory.c_s_star = TOE_C_S_STAR
    theory.n_k_ms = 30
    _TOE_THEORY_MS = theory
    return theory


def compute_ms_nbar(
    k_array: np.ndarray,
    toe_params: dict = None,
) -> Tuple[np.ndarray, dict]:
    """
    Compute n̄_k via MS solver — IDENTICAL to evaluate_bk18.py.

    Parameters
    ----------
    k_array : np.ndarray
        Wavenumbers in physical Mpc⁻¹.
    toe_params : dict, optional
        Default: TOE_PARAMS from evaluate_bk18.py.

    Returns
    -------
    nbar_k : np.ndarray
    ms_results : dict with nbar_k, phi_k, theta_k, A_ring, r_k
    """
    if toe_params is None:
        toe_params = TOE_PARAMS
    theory = get_toe_theory()
    eta_0 = -1.0 / toe_params["k0"]
    ms_results = theory._compute_ms_on_sparse_grid(
        k_sparse=k_array,
        eta_0=eta_0,
        c_s=toe_params["c_s_star"],
        eps_H=toe_params["eps_H"],
        eta_H=toe_params["eta_H"],
        s=toe_params["s_cs"],
        Gamma_over_H=toe_params["Gamma_over_H"],
    )
    return ms_results["nbar_k"], ms_results


# ===================================================================
# 4. FULL CAMB PIPELINE — for experiments needing C_ℓ
# ===================================================================

_TOE_THEORY_CAMB = None


def get_toe_theory_camb() -> ToETheoryErrorEval:
    """Get fully initialized ToETheoryErrorEval with CAMB."""
    global _TOE_THEORY_CAMB
    if _TOE_THEORY_CAMB is not None:
        return _TOE_THEORY_CAMB
    theory = ToETheoryErrorEval()
    theory.initialize()
    _TOE_THEORY_CAMB = theory
    return theory


def run_toe_calculation(
    params: Dict[str, float],
    want_derived: bool = True,
) -> Optional[Dict]:
    """
    Run full ToE pipeline (CAMB + MS solver).

    Parameters: Cobaya-format dict (ombh2, omch2, theta_s_1e2, tau,
    logA, ns, r, nt, k0, A_IR, sigma_IR, alpha2, alpha3, ...).

    Returns {"Cl": {...}, "derived": {...}} or None on ghost violation.
    """
    theory = get_toe_theory_camb()
    state = {}
    success = theory.calculate(state, want_derived=want_derived, **params)
    if not success:
        logger.warning("ToE calculation failed (ghost/positivity violation)")
        return None
    return state


# ===================================================================
# 5. BK18 CHAIN LOADING
# ===================================================================

def load_bk18_chains(
    chains_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load BK18+Planck+BAO chains.

    Uses canonical loader from evaluate_bk18.py.
    Adds max_samples support for testing.
    """
    if chains_dir is None:
        chains_dir = _BK18_DEFAULT
    if not os.path.isdir(chains_dir):
        raise FileNotFoundError(f"BK18 chains not found: {chains_dir}")

    if max_samples is not None:
        pn_files = list(Path(chains_dir).glob("*.paramnames"))
        if not pn_files:
            raise FileNotFoundError(f"No .paramnames in {chains_dir}")
        param_names = {}
        with open(pn_files[0]) as f:
            for i, line in enumerate(f):
                param_names[line.split()[0].strip("*")] = i
        prefix = pn_files[0].stem
        chain_files = sorted(Path(chains_dir).glob(f"{prefix}_*.txt"))
        data = np.vstack([np.loadtxt(str(cf), max_rows=max_samples) for cf in chain_files])
        return data[:, 2:], data[:, 0], param_names

    return _load_bk18_chains_canonical(chains_dir)


def get_bk18_posteriors(
    chains_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract weighted posterior statistics from BK18 chains."""
    samples, weights, param_names = load_bk18_chains(chains_dir, max_samples)
    posteriors = {}
    for name in ["omegabh2", "omegach2", "theta", "tau", "logA", "ns", "r",
                  "H0", "sigma8", "omegam"]:
        if name not in param_names:
            continue
        vals = samples[:, param_names[name]]
        mean = float(np.average(vals, weights=weights))
        std = float(np.sqrt(np.average((vals - mean) ** 2, weights=weights)))
        sort_idx = np.argsort(vals)
        cumw = np.cumsum(weights[sort_idx])
        cumw /= cumw[-1]
        median = float(vals[sort_idx[np.searchsorted(cumw, 0.5)]])
        posteriors[name] = {"mean": mean, "std": std, "median": median}
    return posteriors


# ===================================================================
# 6. TRIVIAL MANUSCRIPT FORMULAS — for background diagnostics
#    These are NOT from modeling/physics_model.py.
#    They are direct implementations of equations from the manuscript.
# ===================================================================

def w_ent(N: np.ndarray, epsilon: float, DeltaN: float, N0: float) -> np.ndarray:
    """w_ent(N) = -1 + ε·exp(-(N-N₀)/ΔN)  (sec03, eq:wEnt)"""
    return -1.0 + epsilon * np.exp(-(N - N0) / DeltaN)


def rho_ent_normalized(
    N: np.ndarray, N0: float, epsilon: float, DeltaN: float, Omega_ent0: float,
) -> np.ndarray:
    """ρ_ent/ρ_c0 — analytical solution of dρ/dN + 3(1+w)ρ = 0 (sec03)"""
    return Omega_ent0 * np.exp(
        -3.0 * epsilon * DeltaN * (1.0 - np.exp(-(N - N0) / DeltaN))
    )


def conservation_residual(
    N: np.ndarray, rho_ent: np.ndarray, w_ent_arr: np.ndarray,
) -> np.ndarray:
    """C(N) = d ln ρ/dN + 3(1+w) — must be zero (sec03, P3)"""
    dlnrho = np.gradient(np.log(np.maximum(rho_ent, 1e-300)), N, edge_order=2)
    return dlnrho + 3.0 * (1.0 + w_ent_arr)


def stability_z2(
    a: np.ndarray, epsH: np.ndarray, c_s: float,
) -> np.ndarray:
    """z² = 2a²ε_H/c_s² — must be positive (sec06)"""
    return 2.0 * a**2 * epsH / c_s**2


def finite_diff_log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d ln y / d ln x via finite differences"""
    return np.gradient(
        np.log(np.maximum(np.abs(y), 1e-300)),
        np.log(np.maximum(np.abs(x), 1e-300)),
        edge_order=2,
    )


def k_phys_to_code(k_phys_Mpc_inv: float, H0_km_s_Mpc: float = 67.4) -> float:
    """Convert k [Mpc⁻¹] to code units where H₀ = 1."""
    return k_phys_Mpc_inv / (H0_km_s_Mpc / 299792.458)


# ===================================================================
# 7. ADDITIONAL BACKGROUND FUNCTIONS — needed by experiments
#    All from manuscript, no modeling/physics_model.py dependency
# ===================================================================

def rho_ent_over_rhoc0(
    N: np.ndarray, N0: float, epsilon: float, DeltaN: float, Omega_ent0: float,
) -> np.ndarray:
    """Alias for rho_ent_normalized (legacy name compatibility)."""
    return rho_ent_normalized(N, N0, epsilon, DeltaN, Omega_ent0)


def eps_H(N: np.ndarray, HN: np.ndarray) -> np.ndarray:
    """First slow-roll parameter ε_H = -d ln H / dN (sec03)."""
    dlnH = np.gradient(np.log(np.maximum(HN, 1e-300)), N, edge_order=2)
    return -dlnH


def H_over_H0_simple(
    N: np.ndarray,
    Omega_r0: float, Omega_m0: float, Omega_L0: float,
    Omega_ent0: float, epsilon: float, DeltaN: float, N0: float,
) -> np.ndarray:
    """
    H(N)/H₀ from Friedmann equation (sec03, Algorithm C1).

    H²/H₀² = Ω_r e^{-4N} + Ω_m e^{-3N} + Ω_Λ + ρ_ent(N)/ρ_c0

    NOTE: This is a simplified background for diagnostic plots.
    For real cosmology, use run_toe_calculation() → CAMB.
    """
    a = np.exp(N)
    rho_r = Omega_r0 * a**(-4)
    rho_m = Omega_m0 * a**(-3)
    rho_L = Omega_L0
    rho_e = rho_ent_normalized(N, N0, epsilon, DeltaN, Omega_ent0)
    H2 = rho_r + rho_m + rho_L + rho_e
    return np.sqrt(np.maximum(H2, 0.0))


def conformal_time(N: np.ndarray, HN: np.ndarray) -> np.ndarray:
    """Conformal time η(N) = ∫ dN / (aH) (sec03)."""
    a = np.exp(N)
    integrand = 1.0 / (a * np.maximum(HN, 1e-300))
    from scipy.integrate import cumulative_trapezoid
    eta = np.zeros_like(N)
    eta[1:] = cumulative_trapezoid(integrand, N)
    return eta


def find_N_star_for_k(
    k: float, N: np.ndarray, a: np.ndarray, H: np.ndarray, c_s: float,
) -> Tuple[float, int]:
    """Find freeze-out N* where k = a(N*)·H(N*)·c_s (sec03)."""
    aH_cs = a * H * c_s
    crossings = np.where(np.diff(np.sign(aH_cs - k)))[0]
    if len(crossings) == 0:
        return float("nan"), -1
    idx = crossings[-1]
    x0, x1 = aH_cs[idx], aH_cs[idx + 1]
    if abs(x1 - x0) < 1e-30:
        return N[idx], idx
    frac = (k - x0) / (x1 - x0)
    return float(N[idx] + frac * (N[idx + 1] - N[idx])), idx


def inflation_window_mask(
    N: np.ndarray, epsH_arr: np.ndarray, N0: float, eps_max: float = 1.0,
) -> np.ndarray:
    """Boolean mask for inflation window: ε_H < eps_max and N ≥ N₀."""
    return (epsH_arr < eps_max) & (N >= N0)

