#!/usr/bin/env python3
"""
ToE Error Evaluation — MCMC Runner

MEASURES the consistency relation deviation from unity:

  Q_obs = r / (-8 n_t)

  Standard Inflation: Q = 1
  ToE prediction:     Q = c_s* / (1 + 2n̄_k)

KEY: r and n_t are FREE parameters — data determines their values.
The "error" (Q - 1) IS the signal. If ToE is correct:
  - Q ≠ 1 (deviation from SI)
  - Q ≈ c_s*/(1+2n̄_k) (matches ToE prediction)
  - delta_toe ≈ 0 (residual is zero)

Derived parameters in chains:
  Q_obs      = r/(-8 n_t)
  delta_obs  = Q_obs - 1              (deviation from SI)
  delta_toe  = Q_obs - c_s*/(1+2n̄_k) (residual vs ToE)
  Q_toe_pred = c_s*/(1+2n̄_k)         (ToE prediction)

CONFIRMATION of ToE:
  posterior(delta_obs) excludes 0 at >3σ  AND  posterior(delta_toe) includes 0

REFUTATION of ToE:
  posterior(delta_toe) excludes 0 at >3σ

Usage:
    python toe_error_evaluation/run_mcmc.py [--test] [--resume]

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

import os
import sys
import argparse
from pathlib import Path


def get_info_dict(test_mode: bool = False) -> dict:
    """
    Build Cobaya info dictionary for ToE error evaluation.
    
    IDENTICAL parameters to toe_mcmc_physical/run_mcmc.py.
    Uses ToETheoryErrorEval which adds Q_obs, delta_obs, delta_toe.
    """
    
    info = {
        "params": {
            # Standard cosmological parameters
            "ombh2": {
                "prior": {"min": 0.019, "max": 0.025},
                "ref": 0.02236, "proposal": 0.0001,
                "latex": r"\omega_b",
            },
            "omch2": {
                "prior": {"min": 0.10, "max": 0.14},
                "ref": 0.1200, "proposal": 0.001,
                "latex": r"\omega_c",
            },
            "theta_s_1e2": {
                "prior": {"min": 1.03, "max": 1.05},
                "ref": 1.0419, "proposal": 0.0003,
                "latex": r"100\theta_s",
            },
            "tau": {
                "prior": {"min": 0.01, "max": 0.10},
                "ref": 0.054, "proposal": 0.005,
                "latex": r"\tau",
            },
            "logA": {
                "prior": {"min": 2.5, "max": 3.5},
                "ref": 3.044, "proposal": 0.01,
                "latex": r"\ln(10^{10}A_s)",
            },
            "ns": {
                "prior": {"min": 0.9, "max": 1.0},
                "ref": 0.965, "proposal": 0.004,
                "latex": r"n_s",
            },
            
            # =========================================================
            # TENSOR PARAMETERS — FREE (data determines values)
            # This is the KEY difference from physical approach
            # =========================================================
            "r": {
                "prior": {"min": 0.0, "max": 0.1},
                "ref": 0.01, "proposal": 0.005,
                "latex": r"r",
            },
            "nt": {
                "prior": {"min": -0.1, "max": 0.1},
                "ref": -0.0013, "proposal": 0.002,
                "latex": r"n_t",
            },
            
            # ToE parameters
            "k0": {
                "prior": {"min": 0.0005, "max": 0.005},
                "ref": 0.002, "proposal": 0.0005,
                "latex": r"k_0",
            },
            "A_IR": {
                "prior": {"min": -0.1, "max": 0.1},
                "ref": 0.0, "proposal": 0.01,
                "latex": r"A_{\mathrm{IR}}",
            },
            "sigma_IR": {
                "prior": {"min": 0.3, "max": 2.0},
                "ref": 1.0, "proposal": 0.2,
                "latex": r"\sigma_{\mathrm{IR}}",
            },
            "alpha2": {
                "prior": {"dist": "norm", "loc": -0.32, "scale": 0.20},
                "ref": -0.32, "proposal": 0.1,
                "latex": r"\alpha_2",
            },
            "alpha3": {
                "prior": {"min": 0.0, "max": 2.0},
                "ref": 0.98, "proposal": 0.15,
                "latex": r"\alpha_3",
            },
            
            # Slow-roll for MS solver
            "eps_H": {
                "prior": {"min": 1e-4, "max": 0.05},
                "ref": 0.01, "proposal": 0.005,
                "latex": r"\epsilon_H",
            },
            "eta_H": {
                "prior": {"min": -0.1, "max": 0.1},
                "ref": 0.005, "proposal": 0.01,
                "latex": r"\eta_H",
            },
            "s_cs": {
                "prior": {"min": -0.1, "max": 0.1},
                "ref": 0.0, "proposal": 0.01,
                "latex": r"s",
            },
            "Gamma_over_H": {
                "prior": {"min": 1.0, "max": 20.0},
                "ref": 5.0, "proposal": 2.0,
                "latex": r"\Gamma/H",
            },
            
            # =========================================================
            # Nuisance: Planck calibration
            # =========================================================
            "A_planck": {
                "prior": {"dist": "norm", "loc": 1.0, "scale": 0.0025},
                "ref": 1.0, "proposal": 0.001,
                "latex": r"y_{\rm cal}",
            },
            
            # =========================================================
            # Nuisance: BICEP/Keck 2018 foregrounds
            # =========================================================
            "BBdust": {
                "prior": {"min": 0.0, "max": 15.0},
                "ref": 3.0, "proposal": 0.5,
                "latex": r"A_{\rm dust}^{BB}",
            },
            "BBsync": {
                "prior": {"min": 0.0, "max": 50.0},
                "ref": 1.0, "proposal": 1.0,
                "latex": r"A_{\rm sync}^{BB}",
            },
            "BBalphadust": {
                "prior": {"dist": "norm", "loc": -0.42, "scale": 0.01},
                "ref": -0.42, "proposal": 0.01,
                "latex": r"\alpha_{\rm dust}",
            },
            "BBbetadust": {
                "prior": {"dist": "norm", "loc": 1.59, "scale": 0.11},
                "ref": 1.59, "proposal": 0.02,
                "latex": r"\beta_{\rm dust}",
            },
            "BBalphasync": {
                "prior": {"dist": "norm", "loc": -0.6, "scale": 0.1},
                "ref": -0.6, "proposal": 0.05,
                "latex": r"\alpha_{\rm sync}",
            },
            "BBbetasync": {
                "prior": {"dist": "norm", "loc": -3.1, "scale": 0.3},
                "ref": -3.1, "proposal": 0.1,
                "latex": r"\beta_{\rm sync}",
            },
            "BBdustsynccorr": {
                "prior": {"min": -1.0, "max": 1.0},
                "ref": 0.0, "proposal": 0.1,
                "latex": r"\rho_{\rm dust,sync}",
            },
            
            # =========================================================
            # DERIVED parameters (computed by theory, not sampled)
            # =========================================================
            "Q_obs": {"derived": True, "latex": r"Q_{\rm obs}"},
            "delta_obs": {"derived": True, "latex": r"\delta_{\rm obs}"},
            "delta_toe": {"derived": True, "latex": r"\delta_{\rm ToE}"},
            "Q_toe_pred": {"derived": True, "latex": r"Q_{\rm ToE}"},
            "nbar_k_physical": {"derived": True, "latex": r"\bar{n}_k"},
            "occupancy_enhancement": {"derived": True, "latex": r"1+2\bar{n}_k"},
            "consistency_ratio": {"derived": True, "latex": r"r/(-8n_t)"},
            "H0": {"derived": True, "latex": r"H_0"},
            "sigma8": {"derived": True, "latex": r"\sigma_8"},
            "omegam": {"derived": True, "latex": r"\Omega_m"},
        },
        
        # Theory — error evaluation variant
        "theory": {
            "toe_decoherence_validation.toe_theory.ToETheoryErrorEval": {
                "k_pivot": 0.05,
                "c_s_star": 1.0,
                "lmax": 2508,
                "n_k_ms": 30,
            },
        },
        
        # Likelihoods
        "likelihood": {
            "bicep_keck_2018.bicep_keck_2018": None,
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
        },
        
        # Sampler
        "sampler": {
            "mcmc": {
                "burn_in": 10 if test_mode else 100,
                "max_samples": 100 if test_mode else 100000,
                "Rminus1_stop": 0.1 if test_mode else 0.01,
                "Rminus1_cl_stop": 0.2 if test_mode else 0.1,
                "drag": True,
                "oversample_power": 0.4,
                "covmat": "auto",
            },
        },
        
        # Output
        "output": "chains_error_eval/toe_error_eval",
    }
    
    return info


def run_mcmc(test_mode: bool = False, resume: bool = False):
    """Run ToE error evaluation MCMC."""
    
    try:
        from cobaya.run import run
    except ImportError:
        print("ERROR: Cobaya not installed. Run: pip install cobaya")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    
    print("=" * 70)
    print("ToE ERROR EVALUATION — Consistency Relation Measurement")
    print("=" * 70)
    print()
    print("WHAT IS MEASURED:")
    print("  Q_obs = r / (-8 n_t)  — from data (r, n_t are FREE)")
    print()
    print("WHAT IS TESTED:")
    print("  SI predicts:  Q = 1")
    print("  ToE predicts: Q = c_s* / (1 + 2n̄_k)")
    print()
    print("  delta_obs = Q - 1              (deviation from SI)")
    print("  delta_toe = Q - c_s*/(1+2n̄_k) (residual vs ToE)")
    print()
    print("CONFIRMATION: delta_obs ≠ 0 (>3σ) AND delta_toe ≈ 0")
    print("REFUTATION:   delta_toe ≠ 0 (>3σ)")
    print()
    print(f"Mode: {'TEST' if test_mode else 'FULL'}")
    print(f"Resume: {resume}")
    print()
    
    info = get_info_dict(test_mode=test_mode)
    
    if resume:
        info["resume"] = True
    
    Path("chains_error_eval").mkdir(exist_ok=True)
    
    print("Starting MCMC...")
    print("-" * 70)
    
    updated_info, sampler = run(info)
    
    print("-" * 70)
    print("MCMC completed!")
    print()
    print("Chains: chains_error_eval/toe_error_eval.*")
    print()
    print("KEY DERIVED PARAMETERS in chains:")
    print("  Q_obs      — r/(-8 n_t)")
    print("  delta_obs  — Q - 1 (deviation from SI)")
    print("  delta_toe  — Q - c_s*/(1+2n̄_k) (residual vs ToE)")
    print("  Q_toe_pred — c_s*/(1+2n̄_k) (ToE prediction)")
    print()
    print("Next: python toe_error_evaluation/analyze.py")
    
    return updated_info, sampler


def main():
    parser = argparse.ArgumentParser(
        description="ToE Error Evaluation — Consistency Relation Measurement"
    )
    parser.add_argument("--test", action="store_true",
                        help="Test mode (fewer samples)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    
    args = parser.parse_args()
    run_mcmc(test_mode=args.test, resume=args.resume)


if __name__ == "__main__":
    main()
