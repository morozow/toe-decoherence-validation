# Theory of Everything Decoherence Validation: Empirical Validation of the ToE Consistency Relation

[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey?style=for-the-badge&logo=creativecommons&logoColor=white)](LICENSE-docs)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19313505-blue?style=for-the-badge&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19313505)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

Empirical validation framework for a Theory of Everything (ToE) based on internal decoherence, entanglement-sourced stress-energy, geometry as an equation of state of entanglement, and emergent gauge symmetries from branch algebra.

**Author:** Raman Marozau · [ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135) · Independent Researcher \
**Manuscript:** *A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress-Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra* \
**Repository scope:** BK18/Planck/BAO-driven inference, three-channel consistency validation, and per-claim experimental verification.

---

## What's New in v3.1.0

Version 3.1.0 adds a **Claim Validation Experiment Suite** (`src/experiments/`) — a growing collection of self-contained experiments, each testing a specific falsifiable claim of the theory against real data. The core package (`toe_decoherence_validation`) remains the foundation; experiments build on top of it via a shared physics layer (`experiments/common/`). New experiments will continue to be added in future releases.

---

## Abstract

This repository provides the analysis code and reproducibility workflow for empirical ToE validation using cosmological data products from BK18 + Planck 2018 + BAO (NASA LAMBDA). The pipeline integrates:

1. A theory implementation with Mukhanov-Sasaki (MS) dynamics and Bogoliubov matching at the decoherence time $\eta_0$;
2. MCMC inference with free $n_t$ (tensor spectral index);
3. Single-point and map-based BK18 evaluation;
4. Joint feasibility tests across three independent channels:
   - Consistency relation channel: $Q(k) = c_s^\ast / (1 + 2\bar{n}_k) < 1$
   - Ring-down channel: oscillations in $P_\zeta(k)$ with physical phase $\phi_k$
   - Non-Gaussianity channel: $f_{NL}$ suppression by the same occupancy factor

The repository is intended for transparent verification, independent reruns, and archival reproducibility (GitHub + Zenodo DOI).

---

## Scientific Overview

Standard single-field slow-roll inflation predicts the tensor consistency relation:

$Q \equiv \frac{r}{-8\,n_t} = 1$

The ToE predicts a scale-dependent modification due to decoherence-induced occupancy $\bar{n}_k$:

$Q(k) = \frac{c_s^\ast}{1 + 2\bar{n}_k(k)}$

where $\bar{n}_k = |\beta_k|^2$ is the Bogoliubov particle number from matching at the decoherence time $\eta_0 = -1/k_0$.

At the pivot scale ($k = 0.05$ Mpc$^{-1}$), $\bar{n}_k \approx 0$ and $Q = 1$ (standard inflation recovered). At low $k$ ($k \lesssim k_0$), $\bar{n}_k > 0$ and $Q < 1$ — this is the ToE signal.

---

## Installation

### Requirements

- Python 3.10+
- `numpy`, `scipy`
- `cobaya`, `camb` (for MCMC; optional for BK18 evaluation)
- `matplotlib` (for plots)

### Setup

```bash
git clone https://github.com/morozow/toe-decoherence-validation.git
cd toe-decoherence-validation

# Install package in editable mode
pip install -e .

# With MCMC support (cobaya + camb + getdist)
pip install -e ".[mcmc]"
```

---

## Quick Start

### Step 0: Download BK18 chains (required, ~2.5 GB)

The BK18 MCMC chains are public data from NASA LAMBDA. They are not included in this repository due to size.

```bash
# Download BICEP/Keck 2018 chains (without raw data files)
curl -L -o chains_no_data_files.tar.gz \
  https://lambda.gsfc.nasa.gov/data/suborbital/BICEPK_2021/chains_no_data_files.tar.gz

# Extract to temporary directory
mkdir -p /tmp/bk18_chains
tar xzf chains_no_data_files.tar.gz -C /tmp/bk18_chains/
```

This archive contains public MCMC posterior chains from the joint analysis of BICEP/Keck 2018 B-mode + Planck 2018 TT/TE/EE + BAO data, produced by the BICEP/Keck collaboration using CosmoMC. The chain set used in this analysis is `BK18_17_BK18lf_freebdust_incP2018_BAO` (1,948,224 samples).

Source: [NASA LAMBDA — BICEP/Keck 2018 Data Products](https://lambda.gsfc.nasa.gov/product/bicepkeck/)

### Step 1–3: Run core analysis

```bash
# Via entry points (after pip install -e .)
toe-eval-bk18
toe-eval-map
toe-joint

# Or via python -m
python -m toe_decoherence_validation.evaluate_bk18
python -m toe_decoherence_validation.evaluate_bk18_map
python -m toe_decoherence_validation.joint_analysis
```

### Step 4: Run claim validation experiments

```bash
# Via entry points
toe-exp03    # Consistency relation Q(k)
toe-exp04    # Conservation law ∇_μ T^ent = 0
toe-exp05    # Ghost-freedom / stability
toe-exp06    # Power spectra & ring-down
toe-exp11    # Emergent gravity from entanglement

# Or directly
python src/experiments/exp03_consistency_relation/run_experiment.py
python src/experiments/exp04_conservation_law/run_experiment.py
python src/experiments/exp05_stability/run_experiment.py
python src/experiments/exp06_power_spectra/run_experiment.py
python src/experiments/exp11_quantum_gravity/run_experiment.py
```

Each experiment produces `RESULTS.txt`, data CSVs, and plots in its own directory.

---

## Repository Structure

### Core package (primary)

```
src/toe_decoherence_validation/
    ├── __init__.py
    ├── toe_theory.py            # Theory class (MS solver + Bogoliubov matching)
    ├── mukhanov_sasaki.py       # Mukhanov-Sasaki ODE solver
    ├── run_mcmc.py              # MCMC with free n_t (Cobaya)
    ├── analyze.py               # Chain post-processing and verdict
    ├── evaluate_bk18.py         # Single-point BK18 evaluation
    ├── evaluate_bk18_map.py     # Sensitivity map (k0 × eps_H scan)
    └── joint_analysis.py        # Three-channel joint feasibility
```

### Claim validation experiments (growing suite)

```
src/experiments/
    ├── common/                  # Shared physics layer (toe_physics, reporting, params)
    ├── exp03_consistency_relation/   # Claim: Q(k) = c_s*/(1+2n̄_k)
    ├── exp04_conservation_law/       # Claim: ∇_μ T^ent_μν = 0
    ├── exp05_stability/              # Claim: ghost-freedom & z² > 0
    ├── exp06_power_spectra/          # Claim: ring-down oscillations in P_ζ(k)
    └── exp11_quantum_gravity/        # Claim: 1/(4G_eff) = σ_EE = κ_EE · ℓ_c⁻²
```

Each experiment is self-contained with `run_experiment.py`, `data/`, `plots/`, and `RESULTS.txt`. New experiments will be added as additional claims are tested.

### Documentation

```
docs/
    ├── ONE_CLAIM_PAPER.md / .pdf         # One-claim paper
    ├── CLAIM_exp03_consistency_relation   # Per-experiment claim documents
    ├── CLAIM_exp04_conservation
    ├── CLAIM_exp05_ghost_freedom
    ├── CLAIM_exp06_ring_down
    └── CLAIM_exp11_emergent_gravity
```

### Plots (core analysis)

```
plots/
    ├── sensitivity_max_deviation.png
    ├── Q_vs_k_by_k0.png
    ├── Q_vs_k_by_epsH.png
    ├── nbar_k_heatmap.png
    ├── joint_feasibility_map.png
    ├── joint_three_channels.png
    └── ringdown_vs_gamma.png
```

---

## Claim Validation Experiments

The `src/experiments/` suite tests individual falsifiable claims of the theory. Each experiment:

- Uses the shared physics layer (`experiments/common/toe_physics`) — zero local formulas
- Loads real BK18+Planck+BAO data or runs the MS solver with canonical parameters
- Produces a PASS/FAIL verdict with quantitative thresholds
- Generates reproducible data files and plots

| Experiment | Claim |
|---|---|
| `exp03` | Generalized consistency relation $Q(k) < 1$ |
| `exp04` | Entanglement stress-energy conservation $\nabla_\mu T^{\text{ent}}_{\mu\nu} = 0$ |
| `exp05` | Ghost-freedom and stability $z^2 > 0$ |
| `exp06` | Ring-down oscillations in scalar power spectrum |
| `exp11` | Emergent gravity $1/(4G_{\text{eff}}) = \sigma_{EE}$ |

Companion claim documents are in `docs/CLAIM_*.md` and `docs/CLAIM_*.pdf`.

---

## Key Results

**Data:** BK18 + Planck 2018 + BAO (NASA LAMBDA), 1,948,224 samples

| Quantity | Value |
|----------|-------|
| $r$ | $0.016268 \pm 0.010134$ |
| $n_s$ | $0.966912$ |
| $Q(k)$ range | 0.76 – 1.00 (scale-dependent) |
| Manuscript point | ALL THREE CHANNELS PASS |
| Joint feasible | 75 / 125 (60%) |
| Phase score | 0.98 (gauge-robust complex metric) |
| $Q(k_0)$ quasi-invariant | $\approx 0.94$ |

### Scale-dependent deviation at manuscript point ($k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5$)

| $k$ [Mpc$^{-1}$] | $\bar{n}_k$ | $Q(k)$ | Deviation |
|---|---|---|---|
| 0.0005 | 0.156 | 0.762 | 23.8% |
| 0.001 | 0.038 | 0.930 | 7.0% |
| 0.002 | 0.033 | 0.939 | 6.1% |
| 0.05 | $10^{-9}$ | 1.000 | ~0 (pivot null test) |

### Sensitivity

- $\varepsilon_H$: weak influence (< 1 pp variation across factor-50 range)
- $\Gamma/H$: no effect on $Q$ or $f_{NL}$ (affects only ring-down damping)
- $k_0$: dominant parameter (determines scale and amplitude of ToE effect)

---

## Reproducibility

### Data provenance

| Source | URL |
|--------|-----|
| BK18 chains | `https://lambda.gsfc.nasa.gov/data/suborbital/BICEPK_2021/chains_no_data_files.tar.gz` |
| Chain set | `BK18_17_BK18lf_freebdust_incP2018_BAO` |
| Origin | NASA LAMBDA |

### Reproduction steps

1. Clone repository and install dependencies
2. Extract BK18 chains to `/tmp/bk18_chains/`
3. Run core analysis: `toe-eval-bk18`, `toe-eval-map`, `toe-joint`
4. Run experiments: `toe-exp03` through `toe-exp11`
5. Compare output with `RESULTS.txt` in each experiment directory and `plots/`

### Determinism notes

- MS solver is deterministic (no random seeds)
- BK18 chain loading is deterministic (weighted statistics)
- Plots generated via matplotlib with `Agg` backend
- All experiments produce identical output on re-run

---

## Citation

If you use this code or derived results, please cite both the software and the manuscript.

### CITATION.cff

```yaml
cff-version: 1.2.0
title: "toe-decoherence-validation"
message: "If you use this software, please cite it as below."
type: software
authors:
  - family-names: "Marozau"
    given-names: "Raman"
    orcid: "https://orcid.org/0009-0000-0241-1135"
    affiliation: "Independent Researcher"
repository-code: "https://github.com/morozow/toe-decoherence-validation"
license: "Apache-2.0"
doi: "10.5281/zenodo.19313505"
version: "v3.1.0"
date-released: "2026-04-08"
keywords:
  - cosmology
  - theory of everything
  - decoherence
  - entanglement
  - consistency relation
  - BK18
  - Planck 2018
  - Mukhanov-Sasaki
```

### BibTeX (software)

```bibtex
@software{marozau_toe_decoherence_validation_2026,
  author       = {Marozau, Raman},
  title        = {toe-decoherence-validation},
  year         = {2026},
  version      = {v3.1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19313505},
  url          = {https://github.com/morozow/toe-decoherence-validation}
}
```

### BibTeX (manuscript)

```bibtex
@article{marozau_toe_2026,
  author  = {Marozau, Raman},
  title   = {A Theory of Everything from Internal Decoherence, Entanglement-Sourced
             Stress-Energy, Geometry as an Equation of State of Entanglement,
             and Emergent Gauge Symmetries from Branch Algebra},
  year    = {2026},
  note    = {Submitted to JCAP}
}
```

---

## License

- **Code** (`*.py`, scripts, pipeline): [Apache License 2.0](LICENSE)
- **Documents** (`*.md`, manuscript text, figures): [CC BY 4.0](LICENSE-docs)
- Third-party data and dependencies retain their original licenses.

---

## References

1. R. Marozau, "A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress-Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra" (2026).
2. BICEP/Keck Collaboration, "Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season," Phys. Rev. Lett. 127, 151301 (2021).
3. Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," A&A 641, A6 (2020).
4. J. Torrado and A. Lewis, "Cobaya: Code for Bayesian Analysis of hierarchical physical models," JCAP 05, 057 (2021).
5. A. Lewis, A. Challinor, and A. Lasenby, "Efficient Computation of CMB anisotropies in closed FRW models," Astrophys. J. 538, 473 (2000).

---

## Contact

**Raman Marozau** — author and maintainer.
[ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135)

For issues: use GitHub Issues with commit hash, environment details, and full traceback.
