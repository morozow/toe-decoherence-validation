"""
Convenience layer for all ToE experiments.

ALL physics computations are delegated to the verified implementation:
  - toe_error_evaluation/src/toe_decoherence_validation/toe_theory.py
    (ToETheoryErrorEval — full CAMB pipeline with MS solver)
  - toe_error_evaluation/src/toe_decoherence_validation/mukhanov_sasaki.py
    (MS ODE, Bogoliubov coefficients, ring-down, compute_spectrum_array)
  - toe_error_evaluation/src/toe_decoherence_validation/evaluate_bk18.py
    (BK18+Planck+BAO chain loading and evaluation)
  - modeling/physics_model.py
    (background cosmology: w_ent, H_over_H0, conservation_residual, stability_z2)

NO formulas are duplicated here — this module is purely a convenience layer
that handles import paths, parameter extraction, chain loading, and error handling.

NOTHING in toe_error_evaluation/ is modified — read-only imports only.

Reference: sec03, eq:consistency; sec07; sec13
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (two levels up from experiments/common/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ensure the main project is importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.toe_physics import (
    H_over_H0_simple,
    conformal_time,
    eps_H as compute_eps_H,
    find_N_star_for_k,
    k_phys_to_code,
    compute_ms_nbar,
    get_toe_theory,
    TOE_PARAMS,
)


# ===================================================================
# MS-solver import
# ===================================================================

def get_ms_solver():
    """
    Import the Mukhanov-Sasaki solver module by adding
    ``toe_error_evaluation/src/`` to *sys.path*.

    Returns
    -------
    module
        The ``toe_decoherence_validation.mukhanov_sasaki`` module,
        which exposes ``compute_mode_result``, ``ringdown_from_bogoliubov``,
        and the ``ModeResult`` dataclass.

    Raises
    ------
    ImportError
        If the module cannot be found — with a clear diagnostic message.
    """
    ms_src = os.path.join(_PROJECT_ROOT, "toe_error_evaluation", "src")
    if ms_src not in sys.path:
        sys.path.insert(0, ms_src)

    try:
        from toe_decoherence_validation import mukhanov_sasaki
        return mukhanov_sasaki
    except ImportError as exc:
        raise ImportError(
            "MS-солвер недоступен. Убедитесь что toe_error_evaluation/src/ существует. "
            "Необходимый модуль: toe_decoherence_validation.mukhanov_sasaki"
        ) from exc


# ===================================================================
# Background helpers (for extracting slow-roll at freeze-out)
# ===================================================================

def _build_background(spec_params: dict,
                      N_min: float = -20.0,
                      N_max: float = 8.0,
                      nN: int = 4001) -> dict:
    """
    Build background arrays (N, a, H, eta, eps_H) from *spec_params*.

    Returns dict with keys: N, a, H, eta, epsH, cosmology.
    """
    cosmo: CosmologyParams = spec_params["cosmology"]
    cosmo.finalize()

    N = np.linspace(N_min, N_max, nN)
    a = np.exp(N)
    H = H_over_H0(N, cosmo) * cosmo.H0
    epsH = compute_eps_H(N, H)
    eta = conformal_time(N, H)

    return {
        "N": N,
        "a": a,
        "H": H,
        "eta": eta,
        "epsH": epsH,
        "cosmology": cosmo,
    }


def _slow_roll_at_Nstar(N_star: float, bg: dict) -> Tuple[float, float, float]:
    """
    Interpolate slow-roll parameters at freeze-out *N_star*.

    Returns (eps_H, eta_H, s) where:
      eps_H  — first slow-roll parameter
      eta_H  — d ln eps / dN  (second slow-roll)
      s      — d ln c_s / dN  (= 0 for constant c_s)
    """
    N, epsH = bg["N"], bg["epsH"]
    eps_star = float(np.interp(N_star, N, epsH))

    # eta_H = d ln eps / dN — numerical gradient
    deps_dN = np.gradient(epsH, N, edge_order=2)
    eta_H_arr = deps_dN / np.clip(epsH, 1e-30, None)
    eta_H_star = float(np.interp(N_star, N, eta_H_arr))

    # s = d ln c_s / dN — zero for constant sound speed
    s_star = 0.0

    return eps_star, eta_H_star, s_star


# ===================================================================
# compute_nbar_from_ms
# ===================================================================

def compute_nbar_from_ms(
    k_array: np.ndarray,
    spec_params: dict,
) -> Tuple[np.ndarray, List[Optional[object]]]:
    """
    Compute n̄_k = |β_k|² via the Mukhanov-Sasaki solver for an array of k.

    Delegates to ``ms.compute_mode_result()`` for each k with per-k
    slow-roll extraction at freeze-out.

    The *k_array* must be in **physical Mpc⁻¹** units.

    Parameters
    ----------
    k_array : np.ndarray
        Comoving wavenumbers in physical Mpc⁻¹.
    spec_params : dict
        Output of ``load_spec_params()``.

    Returns
    -------
    nbar_array : np.ndarray
        Occupancy n̄_k for each k.  NaN where the ODE solver failed.
    mode_results : list[ModeResult | None]
        Per-k ModeResult objects (None for skipped points).

    Reference: sec03, eq:consistency; Requirement 1.4
    """
    ms = get_ms_solver()
    bg = _build_background(spec_params)
    cosmo: CosmologyParams = bg["cosmology"]

    ms_params = spec_params["ms_solver_params"]
    eta_0 = ms_params["eta_0"]
    Gamma_over_H = ms_params["Gamma_over_H"]

    N, a, H = bg["N"], bg["a"], bg["H"]

    nbar_array = np.full(len(k_array), np.nan)
    mode_results: List[Optional[object]] = [None] * len(k_array)
    n_skipped = 0

    for i, k_phys in enumerate(k_array):
        try:
            k_phys_val = float(k_phys)
            k_code = k_phys_to_code(k_phys_val)

            N_star, _ = find_N_star_for_k(
                k_code, N, a, H, cosmo.c_s_scalar,
            )
            if not np.isfinite(N_star):
                logger.warning("No freeze-out for k=%.6e Mpc⁻¹ — skip", k_phys_val)
                n_skipped += 1
                continue

            eps_star, eta_H_star, s_star = _slow_roll_at_Nstar(N_star, bg)
            if eps_star <= 0:
                logger.warning("eps_H <= 0 at N_star for k=%.6e — skip", k_phys_val)
                n_skipped += 1
                continue

            # Delegate to MS solver (physical units, H_star=1)
            # sec03: η_star = -1/(c_s k), Δη = η_star - η₀, Γ_k = (Γ/H)·H_star
            c_s = cosmo.c_s_scalar
            eta_star = -1.0 / (c_s * k_phys_val)
            delta_eta = max(eta_star - eta_0, 0.0)
            Gamma_k = Gamma_over_H * 1.0  # H_star = 1

            mode = ms.compute_mode_result(
                k=k_phys_val, eta_0=eta_0, c_s=c_s,
                eps_H=eps_star, eta_H=eta_H_star, s=s_star,
                Gamma_k=Gamma_k, delta_eta=delta_eta,
            )

            nbar_array[i] = mode.nbar_k
            mode_results[i] = mode

        except Exception as exc:
            logger.warning("ODE failed for k=%.6e: %s — skip", float(k_phys), exc)
            n_skipped += 1

    if n_skipped > 0:
        logger.info("compute_nbar_from_ms: %d/%d skipped", n_skipped, len(k_array))

    return nbar_array, mode_results


# ===================================================================
# compute_ringdown_from_ms
# ===================================================================

def compute_ringdown_from_ms(
    k_array: np.ndarray,
    spec_params: dict,
    mode_results: Optional[List[Optional[object]]] = None,
) -> Tuple[List[Optional[Dict[str, float]]], int]:
    """
    Compute ring-down parameters via ``ms.ringdown_from_bogoliubov()``.

    If *mode_results* from a prior ``compute_nbar_from_ms()`` call are
    provided, reuses them to avoid redundant ODE solves.

    Parameters
    ----------
    k_array : np.ndarray
        Comoving wavenumbers in physical Mpc⁻¹.
    spec_params : dict
        Output of ``load_spec_params()``.
    mode_results : list, optional
        Pre-computed ModeResult list from ``compute_nbar_from_ms()``.

    Returns
    -------
    ringdown_list : list[dict | None]
        Per-k dict with keys r_k, theta_k, nbar_k, phi_k, A_ring, damping.
    n_skipped : int

    Reference: sec03; Requirement 5.3
    """
    ms = get_ms_solver()

    # If no pre-computed modes, compute them now
    if mode_results is None:
        _, mode_results = compute_nbar_from_ms(k_array, spec_params)

    bg = _build_background(spec_params)
    cosmo: CosmologyParams = bg["cosmology"]
    ms_params = spec_params["ms_solver_params"]
    eta_0 = ms_params["eta_0"]
    Gamma_over_H = ms_params["Gamma_over_H"]
    N, a, H = bg["N"], bg["a"], bg["H"]

    ringdown_list: List[Optional[Dict[str, float]]] = [None] * len(k_array)
    n_skipped = 0

    for i, k_phys in enumerate(k_array):
        mode = mode_results[i] if i < len(mode_results) else None
        if mode is None:
            n_skipped += 1
            continue

        try:
            k_phys_val = float(k_phys)
            k_code = k_phys_to_code(k_phys_val)

            N_star, _ = find_N_star_for_k(
                k_code, N, a, H, cosmo.c_s_scalar,
            )
            eps_star, eta_H_star, s_star = _slow_roll_at_Nstar(N_star, bg)

            c_s = cosmo.c_s_scalar
            eta_star = -1.0 / (c_s * k_phys_val)
            delta_eta = max(eta_star - eta_0, 0.0)
            Gamma_k = Gamma_over_H * 1.0

            # Delegate ring-down extraction to MS solver
            rd = ms.ringdown_from_bogoliubov(
                alpha_k=mode.alpha_k, beta_k=mode.beta_k,
                eps_H=eps_star, eta_H=eta_H_star, s=s_star,
                Gamma_k=Gamma_k, delta_eta=delta_eta,
            )
            ringdown_list[i] = rd

        except Exception as exc:
            logger.warning("Ring-down failed for k=%.6e: %s", float(k_phys), exc)
            n_skipped += 1

    if n_skipped > 0:
        logger.info("compute_ringdown_from_ms: %d/%d skipped", n_skipped, len(k_array))

    return ringdown_list, n_skipped


# ===================================================================
# compute_full_spectra_from_ms — uses ms.compute_spectrum_array()
# ===================================================================

def compute_full_spectra_from_ms(
    k_array: np.ndarray,
    P0_array: np.ndarray,
    spec_params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectra with physical ring-down for an array of k.

    Delegates entirely to ``ms.compute_spectrum_array()`` — the full
    pipeline from the MS solver (ODE → Bogoliubov → ring-down → P_ζ).

    This uses a single set of slow-roll parameters (from spec_params)
    for all k, matching the toe_theory.py convention.

    Parameters
    ----------
    k_array : np.ndarray
        Comoving wavenumbers in physical Mpc⁻¹.
    P0_array : np.ndarray
        Base power spectrum P_ζ^(0)(k) for each k.
    spec_params : dict
        Output of ``load_spec_params()``.

    Returns
    -------
    P_zeta : np.ndarray
        Power spectrum with ring-down.
    nbar_k : np.ndarray
        Occupancy n̄_k for each k.
    phi_k : np.ndarray
        Ring-down phase φ_k for each k.
    A_ring : np.ndarray
        Ring-down amplitude A(k) for each k.

    Reference: sec03, eq:consistency; sec13
    """
    ms = get_ms_solver()
    ms_params = spec_params["ms_solver_params"]
    cosmo: CosmologyParams = spec_params["cosmology"]

    # Delegate to ms.compute_spectrum_array() — NO formula duplication
    return ms.compute_spectrum_array(
        k_array=k_array,
        P0_array=P0_array,
        eta_0=ms_params["eta_0"],
        c_s=cosmo.c_s_scalar,
        eps_H=cosmo.epsilon,       # slow-roll from spec
        eta_H=0.0,                 # η_H ≈ 0 for slow-roll
        s=0.0,                     # s = 0 for constant c_s
        Gamma_over_H=ms_params["Gamma_over_H"],
        H_star=1.0,
    )


# ===================================================================
# BK18 chain loading (NASA LAMBDA data)
# ===================================================================

# Default chain paths
_BK18_CHAINS_DEFAULT = os.path.join(
    _PROJECT_ROOT,
    "manuscript",
    "BK18_17_BK18lf_freebdust_incP2018_BAO",
    "BK18_17_BK18lf_freebdust_incP2018_BAO",
)


def load_bk18_chains(
    chains_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load BK18+Planck+BAO MCMC chains from NASA LAMBDA.

    Returns weighted posterior samples with parameter name → column mapping.

    Parameters
    ----------
    chains_dir : str, optional
        Path to the extracted chain directory.
        Default: ``manuscript/BK18_17_BK18lf_freebdust_incP2018_BAO/...``
    max_samples : int, optional
        If set, load at most this many rows per chain file (for testing).

    Returns
    -------
    samples : np.ndarray, shape (N, n_cols)
        All chain samples concatenated. Column 0 = weight, column 1 = -logpost.
    weights : np.ndarray, shape (N,)
        Sample weights (column 0).
    param_names : dict[str, int]
        Mapping from parameter name to column index (0-based).

    Reference: NASA LAMBDA BK18 data products; sec13
    """
    import glob

    if chains_dir is None:
        chains_dir = _BK18_CHAINS_DEFAULT

    if not os.path.isdir(chains_dir):
        raise FileNotFoundError(
            f"BK18 chains directory not found: {chains_dir}\n"
            f"Download from NASA LAMBDA and extract to this path."
        )

    # Find paramnames file
    pn_files = glob.glob(os.path.join(chains_dir, "*.paramnames"))
    if not pn_files:
        raise FileNotFoundError(f"No .paramnames file in {chains_dir}")

    # Parse parameter names
    param_names: Dict[str, int] = {}
    param_names["weight"] = 0
    param_names["minuslogpost"] = 1
    with open(pn_files[0], "r") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if parts:
                name = parts[0].rstrip("*")
                param_names[name] = i + 2  # +2 for weight and -logpost

    # Load chain txt files
    txt_files = sorted(glob.glob(os.path.join(chains_dir, "*_*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No chain .txt files in {chains_dir}")

    all_data = []
    for f in txt_files:
        if max_samples is not None:
            data = np.loadtxt(f, max_rows=max_samples)
        else:
            data = np.loadtxt(f)
        all_data.append(data)

    samples = np.vstack(all_data)
    weights = samples[:, 0]

    logger.info(
        "Loaded BK18 chains: %d samples, %d parameters from %d files",
        len(samples), len(param_names), len(txt_files),
    )

    return samples, weights, param_names


def get_bk18_posteriors(
    chains_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Extract weighted posterior statistics from BK18 chains.

    Returns dict of {param_name: {mean, std, median, lower_68, upper_68}}.

    Reference: sec13, eq:posteriors
    """
    samples, weights, param_names = load_bk18_chains(chains_dir)

    # Key cosmological parameters to extract
    key_params = [
        "omegabh2", "omegach2", "theta", "tau", "logA", "ns", "r",
        "H0", "sigma8", "omegam",
    ]

    posteriors = {}
    for name in key_params:
        if name not in param_names:
            continue
        col = param_names[name]
        vals = samples[:, col]
        w = weights

        # Weighted statistics
        w_norm = w / w.sum()
        mean = float(np.average(vals, weights=w))
        var = float(np.average((vals - mean) ** 2, weights=w))
        std = float(np.sqrt(var))

        # Weighted percentiles (approximate via sorted cumulative weights)
        sort_idx = np.argsort(vals)
        cumw = np.cumsum(w[sort_idx])
        cumw /= cumw[-1]
        median = float(vals[sort_idx[np.searchsorted(cumw, 0.5)]])
        lower_68 = float(vals[sort_idx[np.searchsorted(cumw, 0.16)]])
        upper_68 = float(vals[sort_idx[np.searchsorted(cumw, 0.84)]])

        posteriors[name] = {
            "mean": mean,
            "std": std,
            "median": median,
            "lower_68": lower_68,
            "upper_68": upper_68,
        }

    return posteriors


# ===================================================================
# ToE Theory adapter (full CAMB pipeline)
# ===================================================================

_TOE_THEORY_INSTANCE = None


def get_toe_theory():
    """
    Create and initialize a ``ToETheoryErrorEval`` instance with CAMB.

    The instance is cached (singleton) to avoid repeated CAMB initialization.

    Returns
    -------
    ToETheoryErrorEval
        Initialized theory instance ready for ``calculate()``.

    Raises
    ------
    ImportError
        If Cobaya or CAMB are not available.
    """
    global _TOE_THEORY_INSTANCE
    if _TOE_THEORY_INSTANCE is not None:
        return _TOE_THEORY_INSTANCE

    ms_src = os.path.join(_PROJECT_ROOT, "toe_error_evaluation", "src")
    if ms_src not in sys.path:
        sys.path.insert(0, ms_src)

    try:
        from toe_decoherence_validation.toe_theory import ToETheoryErrorEval
    except ImportError as exc:
        raise ImportError(
            "ToETheoryErrorEval unavailable. "
            "Ensure cobaya and camb are installed."
        ) from exc

    theory = ToETheoryErrorEval()
    theory.initialize()
    _TOE_THEORY_INSTANCE = theory
    return theory


def run_toe_calculation(
    params: Dict[str, float],
    want_derived: bool = True,
) -> Optional[Dict]:
    """
    Run the full ToE calculation pipeline (CAMB + MS solver).

    Delegates to ``ToETheoryErrorEval.calculate()`` — the IDENTICAL
    physics used in the Cobaya MCMC inference.

    Parameters
    ----------
    params : dict
        Parameter dict with keys matching ``get_can_support_params()``:
        ombh2, omch2, theta_s_1e2, tau, logA, ns, r, nt,
        k0, A_IR, sigma_IR, alpha2, alpha3,
        [eps_H, eta_H, s_cs, Gamma_over_H] (optional, have defaults).
    want_derived : bool
        If True, compute derived parameters (H0, sigma8, Q_obs, etc.).

    Returns
    -------
    dict or None
        On success: {"Cl": {...}, "derived": {...}} with C_ℓ spectra
        and derived parameters. On failure (ghost/positivity violation): None.

    Reference: toe_theory.py calculate(); sec03, sec13
    """
    theory = get_toe_theory()
    state = {}

    success = theory.calculate(state, want_derived=want_derived, **params)

    if not success:
        logger.warning("ToE calculation failed (ghost/positivity violation)")
        return None

    return state
