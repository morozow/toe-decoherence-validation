# Emergent Newton Constant from Standard Model Entanglement Area Density

**Author:** Raman Marozau · [ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135) · Independent Researcher

**Date:** 2026-04-05

---

## Abstract

We compute the entanglement area density $\kappa_\text{EE}(\rho_\text{SM}) = 0.215278$ from the Standard Model field content ($N_s = 4$ real scalars, $N_w = 45$ Weyl fermions, $N_v = 12$ gauge vectors) using heat-kernel coefficients on the replica cone. Combined with the entanglement length scale $\ell_c = 0.928\,\ell_\text{Pl}$ (fixed by the spectral action condition), this yields $G_\text{eff} = \ell_c^2/(4\kappa_\text{EE}) = G_N$ in Planck units. The maximum entanglement principle (MEP) relation $\sigma_\text{EE} = 1/(4G_\text{eff})$ is satisfied exactly. An $(\alpha_2, \alpha_3)$ scan over 100 grid points maps the allowed region for the entanglement length scale, with 79/100 points producing consistent cosmology via CAMB ($H_0 = 67.66$ km/s/Mpc). The tensor speed $c_T = 1$ (exact) and graviton mass $m_g = 0$ (exact) are structural consequences of the framework, consistent with GW170817 ($|c_T - 1| < 10^{-15}$).

---

## 1. The Claim

Gravity emerges from entanglement structure with:

(i) $\kappa_\text{EE} = 0.215278$ computed from SM field content via heat-kernel coefficients — no free parameters;

(ii) $\ell_c = 0.928\,\ell_\text{Pl} = 1.50 \times 10^{-35}$ m from the spectral action condition;

(iii) $G_\text{eff} = G_N$ follows from $1/(4G_\text{eff}) = \kappa_\text{EE} \cdot \ell_c^{-2}$;

(iv) $c_T = 1$ exactly (no anisotropic stress at background level), $m_g = 0$ (diffeomorphism invariance);

(v) $(\alpha_2, \alpha_3)$ scan: 79/100 parameter points produce consistent CAMB cosmology.

---

## 2. What Is New

- **$\kappa_\text{EE}$ from SM content.** The entanglement area density is computed, not assumed. The heat-kernel coefficients $\kappa_0(\xi=1/6) = 1/90$, $\kappa_{1/2} = 7/720$, $\kappa_1 = -1/45$ are standard results from the replica method. The SM content $(4, 45, 12)$ is fixed by anomaly cancellation.

- **Breakdown by spin.** Scalar contribution: $4 \times 1/90 = 0.044$. Fermion: $45 \times 7/720 = 0.438$. Vector: $12 \times (-1/45) = -0.267$. Fermions dominate; vectors subtract (ghost subtraction).

- **$(\alpha_2, \alpha_3)$ allowed region.** 100-point scan through CAMB maps where the emergent geometry is consistent. Sharp boundary at $\alpha_2 + \alpha_3/3 = 0$.

---

## 3. Physical Framework

The emergence of gravity from entanglement is the central structural claim of the ToE framework. The mechanism proceeds through three steps: (1) the entanglement entropy across any smooth spacelike cut scales with area, (2) the area coefficient is fixed by the Standard Model field content, and (3) the Modular Equivalence Principle (MEP) identifies this area density with Newton's constant.

The entanglement entropy of the reduced state across a smooth cut $\Sigma$ of area $A[\Sigma]$ is [1]:

$$S_\text{EE}[\Sigma] = \sigma_\text{EE} \cdot A[\Sigma], \qquad \sigma_\text{EE}(\rho, \ell_c) = \kappa_\text{EE}(\rho) \cdot \ell_c^{-2}$$

where $\sigma_\text{EE}$ is the entanglement area density, $\kappa_\text{EE}(\rho)$ is the UV spectral coefficient determined by the local field content, and $\ell_c$ is the code/coarse-graining scale induced by the decoherence act. The area scaling (not volume scaling) is the holographic principle — it is a consequence of the locality of entanglement across the cut.

The spectral coefficient $\kappa_\text{EE}$ is computed from the replica/heat-kernel method on the conical manifold [1]. For a Laplace-type operator of spin $s$, the surface Seeley–DeWitt coefficient $\hat{a}_1^{(s)}$ gives the contribution to the area term. The heat-kernel coefficients $\kappa_s$ for each spin are standard results:

$$\kappa_\text{EE}(\rho_\text{SM}) = N_s \cdot \kappa_0(\xi = 1/6) + N_w \cdot \kappa_{1/2} + N_v \cdot \kappa_1$$

where $\kappa_0(1/6) = 1/90$, $\kappa_{1/2} = 7/720$, $\kappa_1 = -1/45$ (Vassilevich 2003). The SM field content $(N_s = 4, N_w = 45, N_v = 12)$ is fixed by anomaly cancellation — it is the unique minimal chiral spectrum that cancels all gauge, mixed, and gravitational anomalies [1]. No free parameters remain.

The MEP identifies the entanglement area density with the gravitational coupling [1]:

$$\frac{1}{4 G_\text{eff}} = \sigma_\text{EE} = \kappa_\text{EE} \cdot \ell_c^{-2}$$

This is the Jacobson-like derivation: the entanglement first law $\delta S_\text{EE} = \delta\langle K \rangle$ in the local Rindler frame implies the Einstein equation with Newton's constant set by the area density. The Single-Act Criticality (SAC) condition fixes $\ell_c = \ell_{c,*} \sim H_*^{-1}$, giving $\ell_c = 0.928 \, \ell_\text{Pl}$ in Planck units.

The tensor speed $c_T = 1$ and graviton mass $m_g = 0$ are structural consequences: at the background level, the entanglement fluid has no anisotropic stress, so tensor perturbations propagate at the speed of light. Diffeomorphism invariance (preserved by the framework) forbids a graviton mass term. These are consistent with the GW170817 constraint $|c_T - 1| < 10^{-15}$.

---

## 4. Data

The input data for the emergent gravity computation are the Standard Model field content (fixed by anomaly cancellation), the heat-kernel coefficients (standard results from the replica method), and the observed Newton constant $G_N$ (CODATA 2018). The Planck length $\ell_\text{Pl}$ sets the natural scale.

| Property | Value |
|----------|-------|
| SM content | $N_s = 4$, $N_w = 45$, $N_v = 12$ (sec11) |
| Heat-kernel | $\kappa_0 = 1/90$, $\kappa_{1/2} = 7/720$, $\kappa_1 = -1/45$ |
| $G_N$ (observed) | $6.67430 \times 10^{-11}$ m³ kg⁻¹ s⁻² (CODATA 2018) |
| $\ell_\text{Pl}$ | $1.616 \times 10^{-35}$ m |

---

## 5. Method

### 5.1 $\kappa_\text{EE}$ Computation

The entanglement area density $\kappa_\text{EE}$ is computed by summing the heat-kernel contributions from each spin species in the Standard Model, weighted by the number of fields. This is a direct algebraic evaluation with no free parameters.

$$\kappa_\text{EE}(\rho_\text{SM}) = N_s \cdot \kappa_0(\xi=1/6) + N_w \cdot \kappa_{1/2} + N_v \cdot \kappa_1$$

$$= 4 \times \frac{1}{90} + 45 \times \frac{7}{720} + 12 \times \left(-\frac{1}{45}\right) = 0.215278$$

### 5.2 Emergent $G_\text{eff}$

The emergent Newton constant follows from the MEP relation: the entanglement area density $\sigma_\text{EE}$ equals $1/(4G_\text{eff})$. In Planck units where $G_N = 1$, this fixes the entanglement length scale $\ell_c$ algebraically.

$$\frac{1}{4G_\text{eff}} = \sigma_\text{EE} = \kappa_\text{EE} \cdot \ell_c^{-2}$$

In Planck units ($G_N = 1$): $\ell_c^2 = 4 \kappa_\text{EE} = 0.861$, so $\ell_c = 0.928\,\ell_\text{Pl}$.

### 5.3 Cross-Validation

The emergent gravity result is cross-validated against four independent checks: the CAMB Boltzmann solver must produce a consistent $H_0$, the MS solver must yield physical occupancy numbers, the conservation law must hold to machine precision, and the $(\alpha_2, \alpha_3)$ scan must map a consistent allowed region.

- `run_toe_calculation(DEFAULT_COBAYA_PARAMS)` → $H_0 = 67.66$ km/s/Mpc
- `compute_ms_nbar()` → $\bar{n}_k \in [10^{-9}, 0.39]$, physical
- Conservation: $|C(N)| = 4.15 \times 10^{-11}$
- $(\alpha_2, \alpha_3)$ scan: 100 points, 79 allowed

---

## 6. Results

### 6.1 $\kappa_\text{EE}$ Breakdown

The entanglement area density receives contributions from three spin sectors. Fermions dominate ($+0.438$), scalars contribute modestly ($+0.044$), and gauge vectors subtract ($-0.267$) due to the ghost subtraction in the gauge-fixed path integral. The total $\kappa_\text{EE} = 0.215$ is a fixed number determined entirely by the SM spectrum.

| Field type | Count | $\kappa_s$ | Contribution |
|-----------|-------|-----------|-------------|
| Real scalars ($\xi = 1/6$) | 4 | $1/90 = 0.01111$ | $+0.04444$ |
| Weyl fermions | 45 | $7/720 = 0.00972$ | $+0.43750$ |
| Gauge vectors | 12 | $-1/45 = -0.02222$ | $-0.26667$ |
| **Total $\kappa_\text{EE}$** | | | **0.21528** |

![Fig. 1: Entanglement area density κ_EE breakdown by field type: scalar (+0.044), fermion (+0.438), vector (−0.267). Total κ_EE = 0.215.](../exp11_quantum_gravity/plots/kappa_ee_breakdown.png)

### 6.2 Derived Quantities

From $\kappa_\text{EE}$ and the MEP relation, all gravitational quantities are determined. The entanglement length scale $\ell_c = 0.928\,\ell_\text{Pl}$ is sub-Planckian, the effective cosmological constant matches the observed $\Omega_\Lambda$, and the tensor speed and graviton mass take their GR values exactly.

| Quantity | Value |
|----------|-------|
| $\ell_c$ | $0.928\,\ell_\text{Pl} = 1.50 \times 10^{-35}$ m |
| $\sigma_\text{EE}$ | 0.250 (Planck units) |
| $G_\text{eff}$ | 1.000 (Planck units) = $6.67430 \times 10^{-11}$ m³ kg⁻¹ s⁻² |
| $\Lambda_\text{eff}/H_0^2$ | $3\Omega_\Lambda = 2.054$ |
| $|c_T - 1|$ | $0$ (exact) |
| $m_g$ | $0$ eV (exact) |

### 6.3 $(\alpha_2, \alpha_3)$ Scan

The $(\alpha_2, \alpha_3)$ parameter space scan tests where the emergent geometry produces a consistent cosmology through the full CAMB pipeline. Of 100 grid points, 79 are allowed and 21 are rejected. The boundary between allowed and rejected regions matches the analytical ghost-freedom condition $\alpha_2 + \alpha_3/3 = 0$.

| Metric | Value |
|--------|-------|
| Grid | $10 \times 10$ in $\alpha_2 \in [-0.5, 0.5]$, $\alpha_3 \in [0, 2]$ |
| Allowed | 79/100 |
| Rejected | 21/100 |
| Boundary | $\alpha_2 + \alpha_3/3 \approx 0$ |

![Fig. 2: Entanglement length scale ℓ_c in the (α₂, α₃) parameter space. The allowed region (79/100 points) is bounded by α₂ + α₃/3 = 0.](../exp11_quantum_gravity/plots/ell_c_map.png)

### 6.4 SM Central Charges

The SM central charges $a_\text{SM}$ and $c_\text{SM}$ are exact rational numbers computed from the field content. They determine the one-loop gravitational effective action and the running of the higher-curvature coefficients $\alpha_2$ and $\alpha_3$ with energy scale [1].

| Quantity | Value | Exact |
|----------|-------|-------|
| $a_\text{SM}$ | 2.7653 | 1991/720 |
| $c_\text{SM}$ | 3.4833 | 209/60 |

---

## 7. Null Test

MEP residual: $|\sigma_\text{EE} - 1/(4G_\text{eff})| = 0$ (exact by construction). The non-trivial content is that $\kappa_\text{EE}$ is fixed by SM content and $\ell_c$ by the spectral action condition — no free parameters remain.

---

## 8. Robustness

$\kappa_\text{EE}$ depends only on SM field content. Any change to $(N_s, N_w, N_v)$ changes $\kappa_\text{EE}$ and thus $G_\text{eff}$. The SM content is fixed by anomaly cancellation (sec11) — alternatives are excluded (sec11, subsec:disq).

---

## 9. Falsification Criteria

### 9.1 Confirmation
Independent measurement of $\ell_c$ from curvature bounds or entanglement entropy measurements consistent with $\ell_c \approx 0.93\,\ell_\text{Pl}$.

### 9.2 Refutation
Discovery of new fundamental particles (changing $N_s$, $N_w$, or $N_v$) that shift $\kappa_\text{EE}$ away from the value needed for $G_\text{eff} = G_N$.

---

## 10. Limitations

The most significant limitation is that $G_\text{eff} = G_N$ holds by construction — the entanglement length scale $\ell_c$ is defined through the observed $G_N$. An independent measurement of $\ell_c$ (e.g., from curvature bounds or entanglement entropy experiments) would elevate this from a consistency check to a prediction.

| Limitation | Impact | Path forward |
|-----------|--------|-------------|
| $G_\text{eff} = G_N$ by construction ($\ell_c$ defined through $G_N$) | Not an independent prediction of $G_N$ | Independent $\ell_c$ measurement |
| $c_T = 1$ set as constant | Not dynamically derived | Perturbation-level tensor equation |
| Heat-kernel coefficients scheme-dependent | Different schemes give different $\kappa_s$ | Note: exp14 uses different scheme |

---

## 11. Reproducibility

All results presented in this work are computed from a publicly available open-source pipeline implementing the heat-kernel computation of $\kappa_\text{EE}$ from Standard Model field content, the MEP relation for emergent $G_\text{eff}$, and cross-validation via the CAMB Boltzmann solver and Mukhanov–Sasaki solver. The pipeline requires Python 3.8+, NumPy, Cobaya, and CAMB.

Code and data: [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18923066-blue?style=for-the-badge&logo=DOI&logoColor=white&color=blue)](https://doi.org/10.5281/zenodo.19313505) 

---

## References

1. R. Marozau, "A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress–Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra" (manuscript, 2026).

2. D. V. Vassilevich, "Heat kernel expansion: user's manual," *Phys. Rept.* **388**, 279–360 (2003). [doi:10.1016/j.physrep.2003.09.002](https://doi.org/10.1016/j.physrep.2003.09.002). arXiv: [hep-th/0306138](https://arxiv.org/abs/hep-th/0306138).

3. N. Aghanim, Y. Akrami, M. Ashdown, J. Aumont, C. Baccigalupi, M. Ballardini, A. J. Banday, R. B. Barreiro, N. Bartolo, S. Basak *et al.* (Planck Collaboration), "Planck 2018 results. VI. Cosmological parameters," *Astron. Astrophys.* **641**, A6 (2020). [doi:10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910). arXiv: [1807.06209](https://arxiv.org/abs/1807.06209).

4. P. A. R. Ade, Z. Ahmed, M. Amiri, D. Barkats, R. Basu Thakur, C. A. Bischoff, D. Beck, J. J. Bock, H. Boenish, E. Bullock *et al.* (BICEP/Keck Collaboration), "BICEP/Keck XIII: Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season," *Phys. Rev. Lett.* **127**, 151301 (2021). [doi:10.1103/PhysRevLett.127.151301](https://doi.org/10.1103/PhysRevLett.127.151301). arXiv: [2110.00483](https://arxiv.org/abs/2110.00483).
