# toe-decoherence-validation

[![Code License: Apache-2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19313505.svg)](https://doi.org/10.5281/zenodo.19313505)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

Empirical validation framework for a Theory of Everything (ToE) based on internal decoherence, entanglement-sourced stress-energy, geometry as an equation of state of entanglement, and emergent gauge symmetries from branch algebra.

**Author:** Raman Marozau · [ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135) · Independent Researcher \
**Manuscript:** *A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress-Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra*
**Repository scope:** BK18/Planck/BAO-driven inference and three-channel consistency validation.

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

$$Q \equiv \frac{r}{-8\,n_t} = 1$$

The ToE predicts a scale-dependent modification due to decoherence-induced occupancy $\bar{n}_k$:

$$Q(k) = \frac{c_s^\ast}{1 + 2\bar{n}_k(k)}$$

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

### Step 1–3: Run analysis

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

---

## Repository Structure

```
src/toe_decoherence_validation
    ├── __init__.py
    ├── toe_theory.py            # Theory class (imports MS solver from toe_mcmc_physical)
    ├── run_mcmc.py              # MCMC with free n_t (Cobaya)
    ├── analyze.py               # Chain post-processing and verdict
    ├── evaluate_bk18.py         # Single-point BK18 evaluation
    ├── evaluate_bk18_map.py     # Sensitivity map (k0 × eps_H scan)
    ├── joint_analysis.py        # Three-channel joint feasibility
    └── README.md                # This file
plots/
    ├── sensitivity_max_deviation.png
    ├── Q_vs_k_by_k0.png
    ├── Q_vs_k_by_epsH.png
    ├── nbar_k_heatmap.png
    ├── joint_feasibility_map.png
    ├── joint_three_channels.png
    └── ringdown_vs_gamma.png
docs/
    ├── ONE_CLAIM_PAPER.md       # One-claim paper, Markdown
    └── ONE_CLAIM_PAPER.pdf      # One-claim paper, PDF
```

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
3. Run `evaluate_bk18.py` → single-point result
4. Run `evaluate_bk18_map.py` → sensitivity map + plots
5. Run `joint_analysis.py` → three-channel feasibility
6. Compare output with `EVALUATION_REPORT.md` and `plots/`

### Determinism notes

- MS solver is deterministic (no random seeds)
- BK18 chain loading is deterministic (weighted statistics)
- Plots generated via matplotlib with `Agg` backend

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
version: "v1.0.0"
date-released: "2026-03-28"
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
  version      = {v1.0.0},
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
