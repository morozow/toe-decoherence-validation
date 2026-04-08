"""
Parameter loading from spec.yaml and TOE_PARAMS for all ToE experiments.

Returns structured dicts — NO CosmologyParams dataclass.
All physics parameters come from:
  - TOE_PARAMS in evaluate_bk18.py (canonical manuscript values)
  - spec.yaml (posteriors, implementation details)
  - BK18 chains (observational posteriors)

Reference: sec13, eq:posteriors
"""

import os
import sys
from typing import Optional

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.toe_physics import (
    TOE_PARAMS, SM_CONTENT, PLANCK_POSTERIORS, k_phys_to_code,
)


def _get_nested(d: dict, dotpath: str, default=None):
    """Retrieve value from nested dict using dot-separated path."""
    keys = dotpath.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_spec_params(spec_path: Optional[str] = None) -> dict:
    """
    Load spec.yaml and return structured dict.

    Returns
    -------
    dict with keys:
        'toe_params'      – dict from TOE_PARAMS (k0, eps_H, eta_H, ...)
        'posteriors'      – dict of posterior values (α₂, α₃, n_s, r, ...)
        'sm_content'      – SM field content (N_s, N_w, N_v, a_SM, c_SM, b1-b3)
        'ms_solver_params' – MS solver params (eta_0, Gamma_over_H)
        'spec_values'     – additional values extracted from spec.yaml
        'raw'             – full YAML dict
    """
    if spec_path is None:
        spec_path = os.path.join(_PROJECT_ROOT, "spec.yaml")

    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"spec.yaml not found at '{spec_path}'.")

    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    if not isinstance(spec, dict):
        raise ValueError(f"spec.yaml did not parse to a dict (got {type(spec).__name__})")

    # --- Posteriors from spec.yaml ---
    posteriors = {}
    hc = _get_nested(spec, "parameters.higher_curvature") or {}
    for key in ("alpha_2", "alpha_3"):
        node = hc.get(key, {})
        if isinstance(node, dict):
            posteriors[key] = {
                "value": node.get("posterior_value"),
                "error": node.get("posterior_error"),
                "citation": node.get("citation", ""),
            }

    sc = _get_nested(spec, "parameters.standard_cosmological") or {}
    for key in ("n_s", "A_s", "omega_b", "omega_c", "theta_s", "tau"):
        node = sc.get(key, {})
        if isinstance(node, dict):
            posteriors[key] = {
                "value": node.get("posterior_value"),
                "error": node.get("posterior_error"),
                "citation": node.get("citation", ""),
            }

    tensor = _get_nested(spec, "parameters.tensor") or {}
    r_node = tensor.get("r", {})
    if isinstance(r_node, dict):
        posteriors["r_upper"] = r_node.get("upper_limit")
    nt_node = tensor.get("n_t", {})
    if isinstance(nt_node, dict):
        posteriors["n_t"] = {
            "value": nt_node.get("posterior_value"),
            "error": nt_node.get("posterior_error"),
        }

    # --- Spec-specific values (implementation details) ---
    occ_base = "implementation_spec.primordial_spectra.scalar_power.occupancy_profile.parameters"
    rd_base = "implementation_spec.primordial_spectra.scalar_power.ring_down.parameters"

    spec_values = {
        "n0": _get_nested(spec, f"{occ_base}.n0.value"),
        "sigma_ln_k": _get_nested(spec, f"{occ_base}.sigma.value"),
        "Gamma_over_H": _get_nested(spec, f"{rd_base}.Gamma_over_H.value"),
        "A_ring": _get_nested(spec, f"{rd_base}.A.value"),
        "k0_phys": _get_nested(spec, "parameters.ir_feature.k_0.posterior_value"),
        "c_s_star": None,
    }

    # Sound speed from spec.yaml
    cs_node = _get_nested(spec, "parameters.perturbations.c_s_star")
    if isinstance(cs_node, dict):
        spec_values["c_s_star"] = cs_node.get("posterior_value", cs_node.get("value"))
        if spec_values["c_s_star"] is None and "allowed_range" in cs_node:
            spec_values["c_s_star"] = float(cs_node["allowed_range"][1])
    elif cs_node is not None:
        spec_values["c_s_star"] = float(cs_node)

    # --- MS solver params ---
    k0_phys = spec_values.get("k0_phys")
    eta_0 = -1.0 / float(k0_phys) if k0_phys else TOE_PARAMS["k0"]
    gamma_val = spec_values.get("Gamma_over_H")
    ms_solver_params = {
        "eta_0": eta_0 if k0_phys else -1.0 / TOE_PARAMS["k0"],
        "Gamma_over_H": float(gamma_val) if gamma_val is not None else TOE_PARAMS["Gamma_over_H"],
    }

    # --- Backward-compatible 'cosmology' namespace ---
    # Experiments use spec["cosmology"].epsilon, .DeltaN, .N0, etc.
    # This provides a simple namespace with the same attributes,
    # sourced from TOE_PARAMS + spec.yaml values.
    class _CosmoCompat:
        """Backward-compatible namespace replacing old CosmologyParams."""
        def __init__(self, tp, sv, ms):
            # From TOE_PARAMS (canonical manuscript values)
            self.epsilon = tp["eps_H"]       # 0.01
            self.DeltaN = 4.0                # sec03, eq:wEnt
            self.N0 = -5.0                   # sec03
            self.Omega_ent0 = 1.0e-3         # sec03
            self.c_s_scalar = tp["c_s_star"] # 1.0
            self.Gamma_over_H = tp["Gamma_over_H"]  # 5.0
            self.k0 = k_phys_to_code(tp["k0"]) if tp["k0"] else 0.05
            # From spec.yaml implementation_spec
            self.n0 = float(sv["n0"]) if sv.get("n0") is not None else 0.5
            self.sigma_ln_k = float(sv["sigma_ln_k"]) if sv.get("sigma_ln_k") is not None else 0.4
            self.A_ring = float(sv["A_ring"]) if sv.get("A_ring") is not None else 0.02
            self.phi_ring = 0.0
            # Standard cosmological (well-known values, not from chains)
            self.Omega_r0 = 9.2e-5
            self.Omega_m0 = 0.315
            self.Omega_L0 = 1.0 - self.Omega_m0 - self.Omega_r0 - self.Omega_ent0
            self.Omega_k0 = 0.0
            self.H0 = 1.0  # code units
            self.c_s_ent = 1.0
        def finalize(self):
            self.Omega_k0 = 1.0 - self.Omega_r0 - self.Omega_m0 - self.Omega_L0 - self.Omega_ent0

    cosmology = _CosmoCompat(TOE_PARAMS, spec_values, ms_solver_params)

    return {
        "cosmology": cosmology,  # backward-compatible namespace
        "toe_params": dict(TOE_PARAMS),
        "posteriors": posteriors,
        "sm_content": dict(SM_CONTENT),
        "ms_solver_params": ms_solver_params,
        "spec_values": spec_values,
        "raw": spec,
    }
