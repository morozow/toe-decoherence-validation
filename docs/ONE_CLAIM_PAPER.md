# Scale-Dependent Prediction for the Inflationary Consistency Relation from Decoherence-Induced Occupancy

**Author:** Raman Marozau · [ORCID: 0009-0000-0241-1135](https://orcid.org/0009-0000-0241-1135) · Independent Researcher

**Date:** 2026-03-28

---

## Abstract

We show that a single Mukhanov–Sasaki/Bogoliubov computation, applied to the Theory of Everything (ToE) decoherence mechanism, simultaneously yields three linked predictions from one parameter set: (i) an infrared consistency-ratio deviation $Q(k) = c_s^\ast/(1+2\bar{n}_k) < 1$ at scales $k \lesssim k_0$, with mean IR deviation 6–70% depending on $k_0$ (peak deviation up to 97% at the lowest $k$); (ii) ring-down oscillations in the scalar power spectrum with phase-stable behavior in the amplitude-relevant region (serving as an internal consistency check); and (iii) non-Gaussianity suppression by the same occupancy factor ($R_{f_{NL}}(k) = Q(k)$). The standard inflationary limit $Q = 1$ is recovered at the pivot scale $k = 0.05$ Mpc$^{-1}$ as a null test. Using public BICEP/Keck 2018 + Planck 2018 + BAO chains (1,948,224 samples, $r = 0.0163 \pm 0.0101$), we demonstrate that 75 out of 125 parameter combinations (60%) satisfy all three channels simultaneously, including the manuscript reference point ($k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5$). This is a robust prediction within the tested domain, not a detection, and is falsifiable with future low-$\ell$ B-mode constraints on independently measured $n_t$.

---

## 1. The Claim

In the tested ToE parameter domain ($k_0 \geq 0.002$ Mpc$^{-1}$, $\varepsilon_H = 0.001$–$0.05$, $\Gamma/H = 1$–$20$), a single Mukhanov–Sasaki/Bogoliubov computation simultaneously yields three linked predictions from one parameter set:

(i) An IR consistency-ratio deviation $Q(k) < 1$ (~6%–70%, scale-dependent);

(ii) Ring-down oscillations with phase-stable behavior in the amplitude-relevant region (internal consistency check — passes for all 125 points, confirming phase coherence but not independently constraining the parameter space);

(iii) Non-Gaussianity suppression by the same occupancy factor ($R_{f_{NL}}(k) = Q(k)$).

The standard limit $Q = 1$ is recovered at the pivot scale. This is falsifiable with future low-$\ell$ B-mode measurements.

---

## 2. What Is New

- **One solver, three channels.** All three observables ($Q$, ring-down, $f_{NL}$) are computed from a single call to the Mukhanov–Sasaki solver with Bogoliubov matching at $\eta_0$. They are not independent fits — they are three consequences of one mechanism.

- **Phase-metric correction.** The ring-down phase $\phi_k = \arg(\alpha_k \beta_k^\ast)$ is evaluated only where observationally relevant (amplitude-weighted mask, $>1\%$ of peak). This removes branch-cut artifacts at high $k$ where $|\beta_k| \to 0$.

- **Manuscript point passes all three channels.** At $k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5.0$: Channel 1 (Q) PASS, Channel 2 (ring-down) PASS, Channel 3 ($f_{NL}$) PASS.

---

## 3. Data

| Property | Value |
|----------|-------|
| Source | BICEP/Keck 2018 + Planck 2018 + BAO joint analysis |
| Chain set | `BK18_17_BK18lf_freebdust_incP2018_BAO` |
| Origin | NASA LAMBDA (`https://lambda.gsfc.nasa.gov/product/bicepkeck/`) |
| Raw samples | 1,948,224 |
| Effective samples | 4,593,771 (weighted) |
| $r$ | $0.016268 \pm 0.010134$ (free parameter) |
| $n_s$ | $0.966912$ (weighted mean) |
| $n_t$ | Fixed to $-r/8$ in chains (standard consistency relation) |

The tensor spectral index $n_t$ is not free in these chains. This is precisely the assumption being tested: the ToE predicts $n_t \neq -r/8$ at low $k$.

---

## 4. Method

### 4.1 ToE Parameters

These parameters are not present in BK18 chains. They define the decoherence mechanism:

| Parameter | Value | Role |
|-----------|-------|------|
| $k_0$ | scanned: 0.0005–0.01 Mpc$^{-1}$ | IR feature scale, sets $\eta_0 = -1/k_0$ |
| $\varepsilon_H$ | scanned: 0.001–0.05 | First slow-roll parameter |
| $\eta_H$ | 0.005 (fixed) | Second slow-roll parameter |
| $s$ | 0.0 (fixed) | Sound speed running |
| $c_s^\ast$ | 1.0 (fixed) | Sound speed at horizon crossing |
| $\Gamma/H$ | scanned: 1–20 | Decoherence rate |

### 4.2 Computation Pipeline

For each parameter point $(k_0, \varepsilon_H, \Gamma/H)$:

1. Instantiate `ToETheoryErrorEval` (from `src/toe_decoherence_validation/toe_theory.py`).
2. Call `_compute_ms_on_sparse_grid(k_grid, eta_0, c_s, eps_H, eta_H, s, Gamma_over_H)`.
3. This single call returns: $\bar{n}_k$, $\phi_k$, $\theta_k$, $A_\text{ring}$, $r_k$ at each $k$.
4. From these, compute:
   - **Channel 1:** $Q(k) = c_s^\ast / (1 + 2\bar{n}_k)$
   - **Channel 2:** $A_\text{ring}(k)$, $\phi_k(k)$ (with amplitude-weighted phase mask)
   - **Channel 3:** $R_{f_{NL}}(k) = 1/(1 + 2\bar{n}_k) = Q(k)/c_s^\ast$

All three channels emerge from the same Bogoliubov coefficient $\beta_k$. No additional fitting.

### 4.3 Channel Pass Criteria

| Channel | Criterion | Physical meaning |
|---------|-----------|-----------------|
| 1 (Q) | $\langle 1-Q \rangle_{k \leq 0.002} \geq 0.05$ AND $|1-Q(0.05)| < 10^{-4}$ | IR deviation $\geq 5\%$ + pivot null test |
| 2 (ring-down) | $A_\text{ring,rms}(\text{IR}) > 10^{-6}$ AND $\phi_k$ smooth where $W(k) > 0.01 \cdot W_\text{max}$ | Nonzero oscillation amplitude + stable phase |
| 3 ($f_{NL}$) | $\langle R_{f_{NL}} \rangle_{k \leq 0.002} < 0.95$ | Suppression $\geq 5\%$ in IR |

Phase weight: $W(k) = A_\text{ring}(k) \cdot (1 + 2\bar{n}_k)$. Points with $W < 1\%$ of maximum are masked (phase undefined where $|\beta_k| \to 0$).

---

## 5. Results

### 5.1 Main Quantitative Result: IR Deviation from $Q = 1$

At the manuscript reference point ($k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5.0$):

| $k$ [Mpc$^{-1}$] | $\bar{n}_k$ | $Q(k)$ | $1 - Q(k)$ | Note |
|---|---|---|---|---|
| 0.0005 | $1.559 \times 10^{-1}$ | 0.7623 | **23.8%** | Maximum effect |
| 0.0010 | $3.772 \times 10^{-2}$ | 0.9299 | **7.0%** | |
| 0.0020 | $3.252 \times 10^{-2}$ | 0.9389 | **6.1%** | $k_0$ |
| 0.0050 | $1.651 \times 10^{-3}$ | 0.9967 | 0.33% | |
| 0.0100 | $1.874 \times 10^{-5}$ | 0.99996 | 0.004% | |
| 0.0200 | $2.277 \times 10^{-7}$ | 1.000000 | ~0 | |
| 0.0500 | $1.257 \times 10^{-9}$ | 1.000000 | ~0 | **Pivot (null test)** |

Mean IR deviation ($k \leq 0.002$): **12.3%**. Pivot: $Q = 0.9999999975$ ($\approx 1$ to 9 decimal places).

### 5.2 Joint-Channel Consistency

Scan: 5 values of $k_0$ × 5 values of $\varepsilon_H$ × 5 values of $\Gamma/H$ = 125 points.

| Channel | Pass rate |
|---------|-----------|
| 1 (Q deviation in IR) | 75/125 (60%) |
| 2 (ring-down) | 125/125 (100%) |
| 3 ($f_{NL}$ suppression) | 75/125 (60%) |
| **All three simultaneously** | **75/125 (60%)** |

The 75 feasible points span $k_0 \in \{0.002, 0.005, 0.01\}$ at all tested $\varepsilon_H$ and $\Gamma/H$. The 50 non-feasible points have $k_0 \in \{0.0005, 0.001\}$ where the mean IR deviation is below the 5% threshold (individual $k$-point deviations reach 6–7%, but the mean across the IR window $k \leq 0.002$ does not meet the criterion due to the coarse $k$-grid in that region).

**Note on Channel 2:** Ring-down passes for all 125 points (100%). After implementing the gauge-robust complex-phase metric ($\Delta\phi_k = \arg(u_{k+1} u_k^\ast)$ with amplitude-weighted edges), the phase is smooth wherever observationally relevant (phase score 0.98 at manuscript point, weighted $|\Delta\phi| = 0.067$ rad, well below $\pi/2$ threshold). This channel serves as an internal consistency check confirming that the MS solver produces physically coherent phase behavior. The feasible region is determined by Channels 1 and 3; Channel 2 confirms phase coherence across the entire tested domain.

Phase metric robustness: at the manuscript point, 4 of 7 $k$-points are significant (obs\_weight $> 1\%$ of max). The remaining 3 points ($k \geq 0.02$) have $|\beta_k| \to 0$ and are correctly masked as phase-undefined.

Manuscript point ($k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5.0$): **ALL THREE PASS**.

### 5.3 Data-Conditioned Inference of ToE-Induced Deviation

This section reports a **data-conditioned inference of ToE-induced deviation**, not a standalone prediction and not a detection.

Operationally, we took the **observed BK18 posterior values of** $r$ (from chains with $n_t = -r/8$), propagated them through the Mukhanov–Sasaki solver with fixed ToE parameters, and computed $\Delta_Q(k) \equiv Q_\text{ToE}(k;\, r_\text{BK18}) - 1$.

Therefore, what is inferred here is the ToE-implied departure from the consistency baseline $Q = 1$ **conditioned on real BK18 data for $r$**. What is **not** done here is a direct data measurement of $Q(k)$ (or free-$n_t$ inference of $n_t$) from polarization data; in BK18, $n_t$ is fixed, so the chains encode $Q = 1$ by construction.

---

## 6. Null Test: Pivot Scale

At $k = 0.05$ Mpc$^{-1}$ (Planck pivot), the mode is deep sub-horizon at $\eta_0$ ($k/k_0 = 25$ for manuscript $k_0$). The Bogoliubov coefficient $\beta_k \to 0$, giving $\bar{n}_k \approx 10^{-9}$ and $Q = 1.000000000$.

This is physically expected: decoherence at $\eta_0$ does not affect modes that are deep sub-horizon at that time. The ToE reduces to standard inflation at the pivot scale.

This null test is passed for **all 125 parameter points** in the scan.

---

## 7. Robustness

### 7.1 Sensitivity to $\varepsilon_H$

The consistency ratio $Q(k)$ is nearly independent of $\varepsilon_H$ in the tested range (0.001–0.05). At fixed $k_0 = 0.002$:

| $\varepsilon_H$ | max$(1-Q)$ |
|---|---|
| 0.001 | 23.6% |
| 0.005 | 23.7% |
| 0.010 | 23.8% |
| 0.020 | 24.0% |
| 0.050 | 24.7% |

Variation: $< 1$ percentage point across a factor-50 range in $\varepsilon_H$. The effect is controlled by $k_0$ (horizon geometry at $\eta_0$), not by slow-roll details.

### 7.2 Sensitivity to $\Gamma/H$

$\Gamma/H$ does not affect $Q(k)$ or $f_{NL}$ suppression (identical values for all $\Gamma/H$ at fixed $k_0$, $\varepsilon_H$). It affects only the ring-down damping rate: larger $\Gamma/H$ suppresses oscillation amplitude faster.

### 7.3 $k_0$ as the Key Parameter

$k_0$ determines the scale and amplitude of the ToE effect:

| $k_0$ [Mpc$^{-1}$] | max$(1-Q)$ | Mean IR deviation |
|---|---|---|
| 0.0005 | 6.3% | below threshold |
| 0.001 | 7.0% | below threshold |
| 0.002 | 23.8% | 12.3% |
| 0.005 | 79.7% | 41.8% |
| 0.01 | 96.9% | 70.8% |

The ToE prediction is a **family of predictions parameterized by $k_0$**. Constraining $k_0$ from data is the key next step.

### 7.4 Quasi-Invariant $Q(k_0) \approx 0.94$

At $k = k_0$ (the decoherence scale), $Q(k_0) \approx 0.94$ across all tested $\varepsilon_H$ values. This is a candidate structural invariant of the matching prescription, arising because $\bar{n}_k(k_0)$ is determined by horizon geometry at $\eta_0$, not by slow-roll parameters.

---

## 8. Falsification Criteria

### 8.1 Confirmation

The ToE receives support if, in an analysis with free $n_t$ and low-$\ell$ B-mode data:

1. The posterior prefers $Q(k) < 1$ at $k \lesssim k_0$ with $> 3\sigma$ significance;
2. The scale dependence matches the predicted form (stronger in IR, vanishing at pivot);
3. The three channels ($Q$, ring-down, $f_{NL}$) are jointly consistent from one parameter set.

### 8.2 Refutation

The ToE is refuted if data with free $n_t$ yield:

1. $Q(k) = 1$ at low $k$ within errors incompatible with the predicted 6–70%;
2. No scale-dependent enhancement toward the IR;
3. Robustness under marginalization over ToE parameters.

### 8.3 Inconclusive

If uncertainties on $n_t$ at low $k$ exceed the predicted $\Delta n_t$, the result is non-discriminating.

---

## 9. Limitations

| Limitation | Impact | Path forward                                                                      |
|-----------|--------|-----------------------------------------------------------------------------------|
| ToE parameters fixed (not marginalized) | $\bar{n}_k$ amplitude depends on $k_0$, $\varepsilon_H$ | Full MCMC with ToE parameters free (`src/toe_decoherence_validation/run_mcmc.py`) |
| $n_t$ not free in BK18 chains | Cannot test $Q \neq 1$ from data directly | MCMC with free $n_t$ (implemented)                                                |
| Instantaneous matching at $\eta_0$ | Leading-order approximation | Finite-width transition analysis                                                  |
| No independent replication | Single codebase | Open code, reproducible pipeline                                                  |
| Scan covers 3 of 6 ToE parameters | $\eta_H$, $s$, $c_s^\ast$ fixed | Extended scan in future work                                                      |
| Phase metric uses 1% amplitude threshold | Threshold choice affects channel 2 | Appendix A documents the definition                                               |

---

## 10. Reproducibility

All code and data are open:

```bash
# Extract BK18 chains
tar xzf modeling/chains_no_data_files.tar.gz -C /tmp/bk18_chains/

# Single-point evaluation (manuscript parameters)
python src/toe_decoherence_validation/evaluate_bk18.py

# Sensitivity map (k0 × eps_H scan)
python src/toe_decoherence_validation/evaluate_bk18_map.py

# Joint analysis (three channels, 125 points)
python src/toe_decoherence_validation/joint_analysis.py
```

Physics implementation: `src/toe_decoherence_validation/toe_theory.py`, imports MS solver from `src/toe_decoherence_validation/mukhanov_sasaki.py`.

All plots generated automatically in `plots/`.

---

## Appendix A: Phase Metric Definition

The ring-down phase $\phi_k = \arg(\alpha_k \beta_k^\ast)$ is physically meaningful only where the Bogoliubov coefficient $|\beta_k|$ is nonzero. At high $k$ ($k \gg k_0$), $|\beta_k| \to 0$ and the phase is numerically undefined.

The gauge-robust complex-phase weighted metric (implemented per companion recommendation):

1. Construct unit complex phase vector: $u_k = e^{i\phi_k}$
2. Compute local phase step: $\Delta\phi_k = \arg(u_{k+1} \cdot u_k^\ast)$ — this is gauge-invariant and removes branch-cut artifacts
3. Compute observational weight: $W(k) = A_\text{ring}(k) \cdot (1 + 2\bar{n}_k)$
4. Mask points where $W(k) < 0.01 \cdot \max(W)$ (phase observationally irrelevant)
5. Compute edge weights: $w_\text{edge} = \frac{1}{2}(W_{k} + W_{k+1})$, normalized to sum to 1
6. Weighted absolute phase variation: $\langle|\Delta\phi|\rangle_w = \sum w_\text{edge} \cdot |\Delta\phi_k|$
7. Phase score: $S_\phi = \text{clip}(1 - \langle|\Delta\phi|\rangle_w / \pi,\; 0,\; 1)$
8. Pass criterion: $\langle|\Delta\phi|\rangle_w < \pi/2$
9. If fewer than 2 significant points: status = `undetermined` (not pass)

At the manuscript point: $S_\phi = 0.98$, $\langle|\Delta\phi|\rangle_w = 0.067$ rad, 4 significant points.

---

## Figures

- **Figure 1:** Three channels at manuscript point ($k_0 = 0.002$, $\varepsilon_H = 0.01$, $\Gamma/H = 5$). Left: $Q(k)$ and $R_{f_{NL}}(k)$. Right: $A_\text{ring}(k)$ and $\phi_k$ (unwrapped, masked). → `plots/joint_three_channels.png`

- **Figure 2:** Sensitivity heatmap: $k_0 \times \varepsilon_H \to \max(1-Q)$. → `plots/sensitivity_max_deviation.png`

- **Figure 3:** Joint feasibility map: number of channels passing (0–3) at each $(k_0, \varepsilon_H)$. → `plots/joint_feasibility_map.png`

- **Figure 4:** Ring-down amplitude $A_\text{ring}(k)$ vs $\Gamma/H$. → `plots/ringdown_vs_gamma.png`

---

## References

1. R. Marozau, "A Theory of Everything from Internal Decoherence, Entanglement-Sourced Stress–Energy, Geometry as an Equation of State of Entanglement, and Emergent Gauge Symmetries from Branch Algebra" (manuscript).

2. P. A. R. Ade, Z. Ahmed, M. Amiri, D. Barkats, R. Basu Thakur, C. A. Bischoff, D. Beck, J. J. Bock, H. Boenish, E. Bullock *et al.* (BICEP/Keck Collaboration), "Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season," Phys. Rev. Lett. **127**, 151301 (2021). [doi:10.1103/PhysRevLett.127.151301](https://doi.org/10.1103/PhysRevLett.127.151301)

3. N. Aghanim, Y. Akrami, M. Ashdown, J. Aumont, C. Baccigalupi, M. Ballardini, A. J. Banday, R. B. Barreiro, N. Bartolo, S. Basak *et al.* (Planck Collaboration), "Planck 2018 results. VI. Cosmological parameters," Astron. Astrophys. **641**, A6 (2020). [doi:10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910)

---

*Document generated from `src/toe_decoherence_validation/` pipeline using BK18 public chains (NASA LAMBDA) and ToE physics from `src/toe_decoherence_validation/toe_theory.py`. All results reproducible via commands in Section 10.*
