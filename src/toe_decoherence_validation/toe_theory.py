"""
ToE Theory Class for Cobaya - ERROR EVALUATION

IDENTICAL physics to toe_mcmc_physical/toe_theory.py (NO formula changes).
Added derived parameters for consistency relation error measurement:

  Q_obs = r / (-8 n_t)                    — observed consistency ratio
  delta_obs = Q_obs - 1                   — deviation from standard inflation
  delta_toe = Q_obs - c_s*/(1+2n̄_k)      — residual vs ToE prediction
  nbar_k_pivot                            — occupancy at pivot

MEASUREMENT LOGIC:
  Standard Inflation predicts: Q = 1
  ToE predicts:                Q = c_s*/(1+2n̄_k)
  
  If data gives Q ≠ 1, the "error" IS the signal.
  If delta_toe ≈ 0, ToE explains the deviation.
  If delta_toe ≠ 0, ToE prediction doesn't match data.

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

import numpy as np
from typing import Dict, Any, Optional
from cobaya.theory import Theory
from scipy.interpolate import interp1d

from toe_decoherence_validation.mukhanov_sasaki import (
    compute_mode_result,
    nu_from_slow_roll,
)


class ToETheoryErrorEval(Theory):
    """
    Theory of Everything - Error Evaluation variant.
    
    SAME physics as ToETheoryPhysical (toe_mcmc_physical/toe_theory.py).
    r and n_t are FREE parameters (data determines their values).
    
    Additional derived parameters for error measurement:
    - Q_obs:     r/(-8 n_t)
    - delta_obs: Q_obs - 1
    - delta_toe: Q_obs - c_s*/(1+2n̄_k)
    """
    
    k_pivot: float = 0.05
    c_s_star: float = 1.0
    lmax: int = 2508
    n_k_ms: int = 30
    
    def initialize(self):
        import camb
        self._camb = camb
    
    def get_requirements(self):
        return {}
    
    def get_can_provide(self):
        return ["Cl"]
    
    def get_can_provide_params(self):
        return [
            # === ORIGINAL derived (from toe_theory.py) ===
            "nbar_k_physical",
            "phi_k_physical",
            "theta_k_physical",
            "occupancy_enhancement",
            "consistency_ratio",
            "H0",
            "sigma8",
            "omegam",
            # === ERROR EVALUATION derived ===
            "Q_obs",          # r/(-8 n_t) — the measured ratio
            "delta_obs",      # Q_obs - 1 — deviation from SI
            "delta_toe",      # Q_obs - c_s*/(1+2n̄_k) — residual vs ToE
            "Q_toe_pred",     # c_s*/(1+2n̄_k) — ToE prediction for Q
        ]
    
    def get_can_support_params(self):
        """IDENTICAL to toe_theory.py — r and nt are FREE."""
        return [
            "ombh2", "omch2", "theta_s_1e2", "tau",
            "logA", "ns",
            "r", "nt",           # FREE — data determines these
            "k0", "A_IR", "sigma_IR",
            "alpha2", "alpha3",
            "eps_H", "eta_H", "s_cs",
            "Gamma_over_H",
        ]

    def _compute_ms_on_sparse_grid(
        self,
        k_sparse: np.ndarray,
        eta_0: float,
        c_s: float,
        eps_H: float,
        eta_H: float,
        s: float,
        Gamma_over_H: float,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Mukhanov-Sasaki solution on sparse k grid.
        IDENTICAL to toe_mcmc_physical/toe_theory.py.
        """
        n = len(k_sparse)
        nbar_k = np.zeros(n)
        phi_k = np.zeros(n)
        theta_k = np.zeros(n)
        A_ring = np.zeros(n)
        r_k = np.zeros(n)
        
        H_star = 1.0
        
        for i, k in enumerate(k_sparse):
            eta_star = -1.0 / (c_s * k)
            delta_eta = max(eta_star - eta_0, 0.0)
            Gamma_k = Gamma_over_H * H_star
            
            try:
                result = compute_mode_result(
                    k=k,
                    eta_0=eta_0,
                    c_s=c_s,
                    eps_H=eps_H,
                    eta_H=eta_H,
                    s=s,
                    Gamma_k=Gamma_k,
                    delta_eta=delta_eta,
                )
                nbar_k[i] = result.nbar_k
                phi_k[i] = result.phi_k
                theta_k[i] = result.theta_k
                A_ring[i] = result.A_ring
                r_k[i] = result.r_k
            except Exception:
                nbar_k[i] = 0.0
                phi_k[i] = 0.0
                theta_k[i] = 0.0
                A_ring[i] = 0.0
                r_k[i] = 0.0
        
        return {
            "nbar_k": nbar_k,
            "phi_k": phi_k,
            "theta_k": theta_k,
            "A_ring": A_ring,
            "r_k": r_k,
        }
    
    def calculate(self, state, want_derived=True, **params):
        """
        Main calculation — IDENTICAL physics to toe_theory.py.
        Added: Q_obs, delta_obs, delta_toe derived parameters.
        """
        # Extract parameters (IDENTICAL to toe_theory.py)
        ombh2 = params["ombh2"]
        omch2 = params["omch2"]
        theta_s = params["theta_s_1e2"] / 100.0
        tau = params["tau"]
        logA = params["logA"]
        ns = params["ns"]
        r = params["r"]
        nt = params["nt"]
        k0 = params["k0"]
        A_IR = params["A_IR"]
        sigma_IR = params["sigma_IR"]
        alpha2 = params["alpha2"]
        alpha3 = params["alpha3"]
        eps_H = params.get("eps_H", 0.01)
        eta_H = params.get("eta_H", 0.005)
        s_cs = params.get("s_cs", 0.0)
        Gamma_over_H = params.get("Gamma_over_H", 5.0)
        
        As = 1e-10 * np.exp(logA)
        
        # Ghost-freedom check (IDENTICAL)
        if alpha3 < 0 or (alpha2 + alpha3 / 3.0) < 0:
            return False
        if r < 0:
            return False
        
        try:
            # Setup CAMB (IDENTICAL)
            pars = self._camb.CAMBparams()
            pars.set_cosmology(
                ombh2=ombh2, omch2=omch2,
                cosmomc_theta=theta_s, tau=tau,
                mnu=0.06, omk=0
            )
            pars.set_for_lmax(self.lmax, lens_potential_accuracy=1)
            pars.WantScalars = True
            pars.WantTensors = True
            pars.WantCls = True
            pars.DoLensing = True
            
            # k grids (IDENTICAL)
            k_min = 1e-5
            k_max = 1.0
            n_k = 500
            k_full = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
            k_sparse = np.logspace(np.log10(k_min), np.log10(k_max), self.n_k_ms)
            
            eta_0 = -1.0 / k0
            
            # MS solver (IDENTICAL)
            ms_results = self._compute_ms_on_sparse_grid(
                k_sparse=k_sparse, eta_0=eta_0,
                c_s=self.c_s_star, eps_H=eps_H,
                eta_H=eta_H, s=s_cs,
                Gamma_over_H=Gamma_over_H,
            )
            
            # Interpolate (IDENTICAL)
            phi_k_interp = interp1d(
                np.log(k_sparse), ms_results["phi_k"],
                kind="linear", fill_value="extrapolate"
            )
            A_ring_interp = interp1d(
                np.log(k_sparse), ms_results["A_ring"],
                kind="linear", fill_value="extrapolate"
            )
            nbar_k_interp = interp1d(
                np.log(k_sparse), ms_results["nbar_k"],
                kind="linear", fill_value="extrapolate"
            )
            
            phi_k_full = phi_k_interp(np.log(k_full))
            A_ring_full = A_ring_interp(np.log(k_full))
            nbar_k_full = nbar_k_interp(np.log(k_full))
            
            # Power spectra (IDENTICAL to toe_theory.py)
            ln_k_ratio = np.log(k_full / self.k_pivot)
            ln_k = np.log(k_full)
            ln_k0 = np.log(k0)
            
            ir_feature = A_IR * np.exp(-(ln_k - ln_k0)**2 / (2.0 * sigma_IR**2))
            
            ln_Pzeta_base = np.log(As) + (ns - 1.0) * ln_k_ratio + ir_feature
            P_zeta_base = np.exp(ln_Pzeta_base)
            
            H_star = 1.0
            eta_star = -1.0 / (self.c_s_star * k_full)
            delta_eta = np.maximum(eta_star - eta_0, 0.0)
            Gamma_k = Gamma_over_H * H_star
            
            damp = np.exp(-Gamma_k * delta_eta)
            osc = np.cos(2.0 * self.c_s_star * k_full * eta_0 + phi_k_full)
            ring_factor = 1.0 + A_ring_full * osc * damp
            
            # Occupancy enhancement (IDENTICAL — eq:Pzeta)
            occupancy_enhancement = 1.0 + 2.0 * nbar_k_full
            P_zeta = P_zeta_base * occupancy_enhancement * ring_factor
            
            # Tensor spectrum (IDENTICAL)
            P_zeta_pivot = As
            P_t = r * P_zeta_pivot * (k_full / self.k_pivot)**nt
            
            # CAMB (IDENTICAL)
            pars.set_initial_power_table(k_full, pk=P_zeta, pk_tensor=P_t)
            pars.InitPower.effective_ns_for_nonlinear = ns
            
            results = self._camb.get_results(pars)
            cls = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
            
            state["Cl"] = {
                "tt": cls["total"][:self.lmax+1, 0],
                "ee": cls["total"][:self.lmax+1, 1],
                "bb": cls["total"][:self.lmax+1, 2],
                "te": cls["total"][:self.lmax+1, 3],
            }
            if "lens_potential" in cls:
                state["Cl"]["pp"] = cls["lens_potential"][:self.lmax+1, 0]
            
            # =============================================================
            # DERIVED PARAMETERS
            # =============================================================
            if want_derived:
                state["derived"] = {}
                
                # CAMB derived (IDENTICAL)
                state["derived"]["H0"] = pars.H0
                h = pars.H0 / 100.0
                state["derived"]["omegam"] = (ombh2 + omch2) / h**2
                try:
                    state["derived"]["sigma8"] = results.get_sigma8_0()
                except:
                    state["derived"]["sigma8"] = np.nan
                
                # Physical quantities at pivot (IDENTICAL)
                k_piv_idx = np.argmin(np.abs(k_full - self.k_pivot))
                nbar_pivot = float(nbar_k_full[k_piv_idx])
                
                state["derived"]["nbar_k_physical"] = nbar_pivot
                state["derived"]["phi_k_physical"] = float(phi_k_full[k_piv_idx])
                state["derived"]["occupancy_enhancement"] = float(1.0 + 2.0 * nbar_pivot)
                
                k_sparse_piv_idx = np.argmin(np.abs(k_sparse - self.k_pivot))
                state["derived"]["theta_k_physical"] = float(ms_results["theta_k"][k_sparse_piv_idx])
                
                # Consistency ratio (IDENTICAL)
                if nt != 0 and r > 0:
                    Q_obs = r / (-8.0 * nt)
                    state["derived"]["consistency_ratio"] = Q_obs
                else:
                    Q_obs = np.nan
                    state["derived"]["consistency_ratio"] = np.nan
                
                # =========================================================
                # ERROR EVALUATION — NEW derived parameters
                # =========================================================
                # Q_obs = r/(-8 n_t) — what data says
                state["derived"]["Q_obs"] = float(Q_obs)
                
                # delta_obs = Q_obs - 1 — deviation from SI prediction
                state["derived"]["delta_obs"] = float(Q_obs - 1.0) if np.isfinite(Q_obs) else np.nan
                
                # Q_toe_pred = c_s*/(1+2n̄_k) — what ToE predicts
                Q_toe_pred = self.c_s_star / (1.0 + 2.0 * nbar_pivot)
                state["derived"]["Q_toe_pred"] = float(Q_toe_pred)
                
                # delta_toe = Q_obs - Q_toe_pred — residual vs ToE
                if np.isfinite(Q_obs):
                    state["derived"]["delta_toe"] = float(Q_obs - Q_toe_pred)
                else:
                    state["derived"]["delta_toe"] = np.nan
            
            return True
            
        except Exception as e:
            import traceback
            if hasattr(self, "log"):
                self.log.debug(f"ToE calculation failed: {e}")
            else:
                print(f"ToE calculation failed: {e}")
                traceback.print_exc()
            return False
    
    def get_Cl(self, ell_factor=False, units="muK2"):
        """Return C_ℓ for likelihood. IDENTICAL to toe_theory.py."""
        cls = self.current_state["Cl"]
        if not ell_factor:
            return cls
        result = {}
        for key, cl in cls.items():
            ell = np.arange(len(cl))
            factor = np.where(ell > 0, ell * (ell + 1) / (2 * np.pi), 0.0)
            result[key] = cl * factor
        return result
