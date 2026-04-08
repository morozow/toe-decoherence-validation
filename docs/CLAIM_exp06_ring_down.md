# Ring-Down Oscillations in the Primordial Power Spectrum from Decoherence-Induced Bogoliubov Occupancy

**Author:** Raman Marozau · [ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135) · Independent Researcher

**Date:** 2026-04-05

---

## Abstract

We report a data-conditioned inference of ring-down oscillations in the primordial scalar power spectrum $P_\zeta(k)$, arising from the Bogoliubov occupancy $\bar{n}_k$ computed via the Mukhanov–Sasaki solver within the ToE decoherence framework. Using BK18+Planck+BAO chains (2,842,467 samples) and the full CAMB pipeline, we find: (i) the occupancy-enhanced spectrum $P_\zeta(k) = (1+2\bar{n}_k) \cdot A_s \cdot (k/k_\ast)^{n_s-1}$ is compatible with Planck 2018 at the pivot ($n_s = 0.9669$, $0.5\sigma$; $A_s = 2.098 \times 10^{-9}$, $0.1\sigma$); (ii) ring-down modulation with effective amplitude $A_\text{eff} \sim \mathcal{O}(\varepsilon_H) \approx 1.25\%$ is predicted at $k \lesssim k_0$; (iii) the predicted naive per-mode signal-to-noise ratio for CMB-S4 at $0.1\%$ fractional precision is $\text{SNR} = 12.5$, making this a detection candidate pending full Fisher analysis. The ring-down amplitude is set by the first slow-roll parameter $\varepsilon_H$ and the Bogoliubov matching at $\eta_0 = -1/k_0$, with no additional free parameters. At the pivot scale, the occupancy enhancement is $1 + 2\bar{n}_k = 1.000000004$, recovering standard inflation as a null test.

---

## 1. The Claim

The ToE decoherence mechanism produces a ring-down modulation of the primordial scalar power spectrum with:

(i) Effective amplitude $A_\text{eff}(k) = A_\text{ring}(k) \cdot e^{-\Gamma_k \Delta\eta}$, peaking at $A_\text{eff} \approx 1.25\%$ near $k \sim k_0$;

(ii) Oscillatory structure $\delta P_\zeta / P_\zeta = A_\text{eff} \cos(2 c_s k \eta_0 + \phi_k)$ with phase $\phi_k$ from Bogoliubov matching;

(iii) Predicted naive per-mode SNR = 12.5 for CMB-S4 ($\sigma(P_\zeta)/P_\zeta = 0.1\%$), SNR = 0.63 for Planck ($\sigma = 2\%$). A full Fisher forecast is needed to confirm;

(iv) Full compatibility with Planck 2018 at the pivot scale ($n_s$ within $0.5\sigma$, $A_s$ within $0.1\sigma$).

The ring-down amplitude is $\mathcal{O}(\varepsilon_H)$ — set by slow-roll, not by fitting. This is a falsifiable prediction testable with CMB-S4.

---

## 2. What Is New

- **Ring-down from first principles.** The oscillation amplitude $A_\text{ring}(k)$ and phase $\phi_k$ emerge from a single call to the Mukhanov–Sasaki solver with Bogoliubov matching at $\eta_0 = -1/k_0$. No phenomenological template is assumed.

- **Quantitative forecast.** Naive per-mode SNR = 12.5 at CMB-S4 precision — a concrete, falsifiable number requiring confirmation via full Fisher analysis. If CMB-S4 achieves $0.1\%$ fractional precision on $P_\zeta(k)$ and no ring-down is detected, the ToE decoherence mechanism is ruled out.

- **Occupancy enhancement profile.** The full $\bar{n}_k$ profile from 60 MS solver modes shows enhancement from $\bar{n}_k \approx 0.16$ at $k = 5 \times 10^{-4}$ Mpc$^{-1}$ to $\bar{n}_k \sim 10^{-9}$ at the pivot — a smooth transition from ToE-dominated to standard inflation.

- **CAMB pipeline verification.** $H_0 = 67.66$ km/s/Mpc from the full CAMB calculation confirms background cosmology consistency.

---

## 3. Physical Framework

The ring-down oscillations arise from the Bogoliubov particle production at the decoherence matching surface $\eta_0 = -1/k_0$.

When the first internal decoherence act occurs at conformal time $\eta_0$, the Mukhanov–Sasaki mode function $v_k(\eta)$ must be matched across the transition. Before $\eta_0$, the mode is in the Bunch-Davies vacuum; after $\eta_0$, it is in a mixed state described by Bogoliubov coefficients $(\alpha_k, \beta_k)$. The coefficient $\beta_k$ encodes particle production: $\bar{n}_k = |\beta_k|^2$ is the mean occupancy number.

The scalar power spectrum in the presence of occupancy is [1]:

$$P_\zeta(k) = \frac{(1 + 2\bar{n}_k) H_*^2}{8\pi^2 M_\text{Pl}^2 \varepsilon_{H*} c_s^*}$$

The factor $(1 + 2\bar{n}_k)$ enhances the spectrum at scales affected by decoherence. The interference between the $\alpha_k$ and $\beta_k$ components produces an oscillatory modulation — the ring-down:

$$\frac{\delta P_\zeta}{P_\zeta} = A_\text{eff}(k) \cos(2 c_s k \eta_0 + \phi_k)$$

where $A_\text{ring}(k)$ is the amplitude from $|\beta_k|$ (set by horizon geometry at $\eta_0$), $\phi_k = \arg(\alpha_k \beta_k^*)$ is the phase from Bogoliubov matching, and the effective amplitude includes decoherence damping: $A_\text{eff}(k) = A_\text{ring}(k) \cdot e^{-(\Gamma/H) \Delta\eta}$, where $\Delta\eta = \eta_* - \eta_0$ is the conformal time between the decoherence act and horizon crossing.

The ring-down amplitude scales as $A_\text{eff} \sim \mathcal{O}(\varepsilon_H)$ because the Bogoliubov coefficient $|\beta_k|$ is determined by the slow-roll parameter through the mode equation. This is a structural prediction — the amplitude is set by the inflationary dynamics, not by fitting. For $\varepsilon_H = 0.01$, the peak effective amplitude is $A_\text{eff,max} \approx 1.25\%$.

This differs fundamentally from phenomenological feature models (e.g., step potentials, particle production templates) in that the oscillation frequency $2 c_s k \eta_0$, amplitude $A_\text{ring}(k)$, and phase $\phi_k$ are all computed from a single call to the Mukhanov–Sasaki solver with no additional free parameters beyond the ToE parameter set $(k_0, \varepsilon_H, \eta_H, c_s^*, \Gamma/H)$.

---

## 4. Data

The observational data are drawn from the joint BK18+Planck+BAO analysis (NASA LAMBDA). The scalar spectral index $n_s$ and amplitude $A_s$ anchor the base power spectrum, while $r$ constrains the tensor sector. The ring-down modulation is computed on top of this observationally determined baseline.

| Property | Value |
|----------|-------|
| Source | BICEP/Keck 2018 + Planck 2018 + BAO joint analysis |
| Chain set | `BK18_17_BK18lf_freebdust_incP2018_BAO` |
| Raw samples | 2,842,467 |
| Effective samples | 6,700,148 (weighted) |
| $n_s$ (chains) | $0.9669 \pm 0.0037$ |
| $r$ (chains) | $0.0163 \pm 0.0102$ |
| $A_s$ (Planck) | $(2.10 \pm 0.03) \times 10^{-9}$ |

---

## 5. Method

### 5.1 ToE Parameters

The five ToE parameters define the decoherence mechanism. The IR feature scale $k_0$ determines where the ring-down peaks in $k$-space (and correspondingly at $\ell \sim 40$–$100$ in the CMB). The slow-roll parameters $\varepsilon_H$ and $\eta_H$ set the inflationary background, $c_s^*$ is the sound speed, and $\Gamma/H$ controls how rapidly the ring-down oscillations are damped.

| Parameter | Value | Role |
|-----------|-------|------|
| $k_0$ | 0.002 Mpc$^{-1}$ | IR feature scale, $\eta_0 = -1/k_0 = -500$ Mpc |
| $\varepsilon_H$ | 0.01 | First slow-roll parameter |
| $\eta_H$ | 0.005 | Second slow-roll parameter |
| $c_s^\ast$ | 1.0 | Sound speed at horizon crossing |
| $\Gamma/H$ | 5.0 | Decoherence rate (controls damping) |

### 5.2 Computation Pipeline

The computation combines three independent data streams: the CAMB Boltzmann solver for background cosmology, the Mukhanov–Sasaki solver for the occupancy profile $\bar{n}_k$ and ring-down quantities, and the BK18 chain posteriors for the observational baseline. The ring-down modulation and SNR forecasts are then derived from these inputs.

1. `run_toe_calculation(DEFAULT_COBAYA_PARAMS)` → CAMB pipeline: $H_0$, $\sigma_8$, $C_\ell$, $\bar{n}_k$ at pivot, $Q_\text{toe}$.
2. `compute_ms_nbar(K_FINE, TOE_PARAMS)` → MS solver on 60-mode $k$-grid: $\bar{n}_k$, $A_\text{ring}$, $\phi_k$ at each $k$.
3. `load_bk18_chains()` → BK18 posteriors: $n_s$, $r$, $A_s$.
4. Compute $P_\zeta(k) = (1+2\bar{n}_k) \cdot A_s \cdot (k/k_\ast)^{n_s-1}$.
5. Compute ring-down: $\delta P_\zeta / P_\zeta = A_\text{eff}(k) \cos(2 c_s k \eta_0 + \phi_k)$.
6. Compute SNR forecasts at Planck, CMB-S4, and ideal precision.

### 5.3 Ring-Down Physics

The ring-down arises from the Bogoliubov coefficient $\beta_k$ at the matching surface $\eta_0$:

- $A_\text{ring}(k)$: amplitude from $|\beta_k|$, set by horizon geometry at $\eta_0$
- $\phi_k = \arg(\alpha_k \beta_k^\ast)$: phase from Bogoliubov matching
- Damping: $e^{-(\Gamma/H) \cdot \Delta\eta}$ where $\Delta\eta = \eta_\ast - \eta_0$
- Effective amplitude: $A_\text{eff}(k) = A_\text{ring}(k) \cdot e^{-(\Gamma/H) \Delta\eta}$

No free parameters beyond the ToE parameter set.

---

## 6. Results

### 6.1 Occupancy Profile

The Bogoliubov occupancy $\bar{n}_k$ is computed from the MS solver on a 60-mode $k$-grid. The occupancy enhancement factor $(1 + 2\bar{n}_k)$ directly multiplies the scalar power spectrum — values significantly above 1 indicate strong ToE modification of the primordial spectrum at that scale.

| $k$ [Mpc$^{-1}$] | $\bar{n}_k$ | $1+2\bar{n}_k$ | Note |
|---|---|---|---|
| $5 \times 10^{-4}$ | $1.56 \times 10^{-1}$ | $1.31$ | |
| $1 \times 10^{-3}$ | $3.86 \times 10^{-2}$ | $1.077$ | |
| $2 \times 10^{-3}$ | $3.07 \times 10^{-2}$ | $1.061$ | $k_0$ |
| $2.85 \times 10^{-3}$ | $2.30$ | $5.60$ | **Peak occupancy** |
| $5 \times 10^{-3}$ | $1.46 \times 10^{-3}$ | $1.003$ | |
| $1 \times 10^{-2}$ | $1.86 \times 10^{-5}$ | $1.00004$ | |
| $5 \times 10^{-2}$ | $1.91 \times 10^{-9}$ | $1.000000004$ | **Pivot (null test)** |

Note: $\bar{n}_k$ is non-monotonic. The peak at $k \approx 2.85 \times 10^{-3}$ Mpc$^{-1}$ (near $k_0$) arises from resonant Bogoliubov particle production at the matching surface $\eta_0$. The profile falls off both toward lower $k$ (fewer modes affected) and higher $k$ (modes deep sub-horizon at $\eta_0$).

![Fig. 1: Primordial scalar power spectrum P_ζ(k): standard inflation (dashed) vs ToE-enhanced with occupancy factor (1+2n̄_k) (solid).](../exp06_power_spectra/plots/power_spectrum.png)

### 6.2 Ring-Down Forecast

The ring-down detection forecast compares the peak effective amplitude $A_\text{eff,max} = 1.25\%$ against the fractional precision $\sigma(P_\zeta)/P_\zeta$ achievable by current and future CMB experiments. The naive per-mode SNR is the ratio of signal to noise at the peak.

| Instrument | $\sigma(P_\zeta)/P_\zeta$ | Max SNR (per-mode) | Detection? |
|-----------|--------------------------|---------|-----------|
| Planck 2018 | 2% | 0.63 | No |
| **CMB-S4** | **0.1%** | **12.5** | **Candidate** |
| Ideal future | 0.01% | 125 | Yes |

Ring-down amplitude: $A_\text{eff,max} = 1.25\%$ at $k \sim k_0$.

Note: The SNR quoted here is a naive per-mode estimate: $\text{SNR} = A_\text{eff,max} / \sigma(P_\zeta/P_\zeta)$. This is an upper bound. A realistic detection forecast requires a full Fisher matrix analysis with $C_\ell$ covariance, foreground marginalization, and look-elsewhere correction, which may reduce the effective SNR.

![Fig. 2: Ring-down modulation δP_ζ/P_ζ with Planck and CMB-S4 detection thresholds. The oscillatory structure arises from Bogoliubov matching at η₀.](../exp06_power_spectra/plots/ring_down_modulation.png)

![Fig. 3: Effective ring-down amplitude A_eff(k) vs detection thresholds for Planck, CMB-S4, and ideal future experiments.](../exp06_power_spectra/plots/ring_down_forecast.png)

### 6.3 Planck Compatibility

The ToE framework must be compatible with existing Planck 2018 constraints at the pivot scale. All four key observables ($n_s$, $A_s$, $r$, $H_0$) are within $0.5\sigma$ of the Planck values, confirming that the ring-down modulation at low $k$ does not spoil the fit at the pivot.

| Observable | ToE value | Planck 2018 | Tension |
|-----------|-----------|-------------|---------|
| $n_s$ | 0.9669 | $0.965 \pm 0.004$ | $0.5\sigma$ |
| $A_s$ | $2.098 \times 10^{-9}$ | $(2.10 \pm 0.03) \times 10^{-9}$ | $0.1\sigma$ |
| $r$ | 0.0163 | $< 0.036$ (95% CL) | PASS |
| $H_0$ | 67.66 km/s/Mpc | $67.4 \pm 0.5$ | $0.5\sigma$ |

---

## 7. Null Test: Pivot Scale

At $k_\ast = 0.05$ Mpc$^{-1}$:
- $\bar{n}_k = 1.91 \times 10^{-9}$
- Occupancy enhancement = $1.000000004$
- $A_\text{eff} = 0$ (ring-down fully damped)
- $P_\zeta(k_\ast) = 2.098 \times 10^{-9}$ (standard inflation recovered)

The ToE reduces to standard inflation at the pivot scale. This null test passes.

---

## 8. Robustness

### 8.1 Ring-Down Amplitude Scaling

$A_\text{eff} \sim \mathcal{O}(\varepsilon_H)$. For $\varepsilon_H = 0.01$: $A_\text{eff,max} = 1.25\%$. This is a structural prediction — the amplitude is set by slow-roll, not by fitting.

### 8.2 Sensitivity to $\Gamma/H$

$\Gamma/H$ controls damping rate. Larger $\Gamma/H$ suppresses ring-down faster. At $\Gamma/H = 5$: ring-down visible at $k \lesssim 0.01$ Mpc$^{-1}$. At $\Gamma/H = 20$: ring-down confined to $k \lesssim 0.003$ Mpc$^{-1}$.

### 8.3 Sensitivity to $k_0$

$k_0$ sets the scale of the feature. Ring-down peaks near $k \sim k_0$. For $k_0 = 0.002$: feature at $\ell \sim 40$–$100$ in CMB.

---

## 9. Falsification Criteria

### 9.1 Confirmation
CMB-S4 detects oscillatory modulation in $P_\zeta(k)$ at $k \lesssim 0.01$ Mpc$^{-1}$ with amplitude $\sim 1\%$ and frequency matching $2 c_s k \eta_0$.

### 9.2 Refutation
CMB-S4 achieves $0.1\%$ precision on $P_\zeta(k)$ and finds no ring-down modulation at $> 3\sigma$ — rules out $A_\text{eff} > 0.03\%$.

### 9.3 Inconclusive
CMB-S4 precision insufficient to reach $0.1\%$ at relevant $k$-scales.

---

## 10. Limitations

Several limitations affect the scope of this inference. The most important is that the SNR estimate is a naive per-mode upper bound; a realistic forecast with foreground marginalization, $C_\ell$ covariance, and look-elsewhere correction may significantly reduce the effective detectability.

| Limitation | Impact | Path forward |
|-----------|--------|-------------|
| $n_s$, $A_s$ from BK18 chains, not predicted | Occupancy enhancement tested, not base spectrum | Full MCMC with ToE theory class |
| Ring-down forecast assumes white noise | Real foregrounds reduce SNR | Foreground marginalization study |
| Instantaneous matching at $\eta_0$ | Leading-order approximation | Finite-width transition |
| Single $k_0$ value tested | Feature scale depends on $k_0$ | $k_0$ scan (done: see exp03) |
| SNR estimate is per-mode, not integrated | Integrated SNR with foregrounds may be significantly lower | Fisher forecast with full $C_\ell$ covariance and foreground marginalization |

---

## 11. Reproducibility

All results presented in this work are computed from a publicly available open-source pipeline implementing the Mukhanov–Sasaki solver with Bogoliubov matching for ring-down computation, the CAMB Boltzmann solver for background cosmology, and BK18+Planck+BAO public chains (NASA LAMBDA) for the observational baseline. The pipeline requires Python 3.8+, NumPy, SciPy, Cobaya, and CAMB. No manual parameter tuning is involved.

Code and data: [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18923066-blue?style=for-the-badge&logo=DOI&logoColor=white&color=blue)](https://doi.org/10.5281/zenodo.19313505) 

---

## References

1. R. Marozau, "A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress–Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra" (manuscript, 2026).

2. P. A. R. Ade, Z. Ahmed, M. Amiri, D. Barkats, R. Basu Thakur, C. A. Bischoff, D. Beck, J. J. Bock, H. Boenish, E. Bullock *et al.* (BICEP/Keck Collaboration), "BICEP/Keck XIII: Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season," *Phys. Rev. Lett.* **127**, 151301 (2021). [doi:10.1103/PhysRevLett.127.151301](https://doi.org/10.1103/PhysRevLett.127.151301). arXiv: [2110.00483](https://arxiv.org/abs/2110.00483).

3. N. Aghanim, Y. Akrami, M. Ashdown, J. Aumont, C. Baccigalupi, M. Ballardini, A. J. Banday, R. B. Barreiro, N. Bartolo, S. Basak *et al.* (Planck Collaboration), "Planck 2018 results. VI. Cosmological parameters," *Astron. Astrophys.* **641**, A6 (2020). [doi:10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910). arXiv: [1807.06209](https://arxiv.org/abs/1807.06209).

4. K. N. Abazajian, P. Adshead, Z. Ahmed, S. W. Allen, D. Alonso, K. S. Arnold *et al.* (CMB-S4 Collaboration), "CMB-S4 Science Book, First Edition," arXiv: [1610.02743](https://arxiv.org/abs/1610.02743) (2016).

5. K. Abazajian, G. Addison, P. Adshead, Z. Ahmed, S. W. Allen *et al.* (CMB-S4 Collaboration), "CMB-S4 Science Case, Reference Design, and Project Plan," arXiv: [1907.04473](https://arxiv.org/abs/1907.04473) (2019).

6. E. Allys, K. Arnold, J. Aumont, R. Aurlien, S. Azzoni, C. Baccigalupi, A. J. Banday, R. Banerji, R. B. Barreiro, N. Bartolo *et al.* (LiteBIRD Collaboration), "Probing Cosmic Inflation with the LiteBIRD Cosmic Microwave Background Polarization Survey," *Prog. Theor. Exp. Phys.* **2023**(4), 042F01 (2023). [doi:10.1093/ptep/ptac150](https://doi.org/10.1093/ptep/ptac150). arXiv: [2202.02773](https://arxiv.org/abs/2202.02773).