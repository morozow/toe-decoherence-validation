"""
Mukhanov-Sasaki Solver with Physical Ring-Down

Implements the full physics from manuscript:
1. MS equation (eq:MS): v_k'' + (c_s² k² - z''/z) v_k = 0
2. z''/z (eq:zpp-over-z): (ν² - 1/4)/η², ν ≈ 3/2 + ε_H + η_H/2 + s/2
3. Hankel solution (eq:Hankel-solution): v_k(η) = √(-πη)/2 · H_ν^(1)(c_s k η)
4. Bogoliubov coefficients (eq:squeezing-params):
   α_k = cosh(r_k), β_k = e^{iθ_k} sinh(r_k), n_k = sinh²(r_k)
5. Ring-down (eq:ringdown):
   P_ζ(k) = P_ζ^(0)(k) · [1 + A(k)·cos(2c_s k η_0 + φ_k)·e^{-Γ_k Δη}]
   φ_k = arg(α_k β_k*) = -θ_k (for real α_k)

Reference: Raman Marozau, "A Theory of Everything from Internal Decoherence..."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ModeResult:
    """Result of Mukhanov-Sasaki mode evolution and matching."""
    alpha_k: complex      # Bogoliubov coefficient α_k
    beta_k: complex       # Bogoliubov coefficient β_k
    r_k: float            # Squeezing parameter r_k = arcsinh(√n̄_k)
    theta_k: float        # Squeezing angle θ_k = arg(β_k)
    nbar_k: float         # Occupancy n̄_k = |β_k|² = sinh²(r_k)
    phi_k: float          # Ring-down phase φ_k = arg(α_k β_k*) = -θ_k
    A_ring: float         # Ring-down amplitude A(k) = O(ε_H, η_H, s)
    damping: float        # Decoherence damping exp(-Γ_k Δη)


def nu_from_slow_roll(eps_H: float, eta_H: float, s: float) -> float:
    """
    Compute ν from slow-roll parameters (eq:zpp-over-z).
    
    ν ≈ 3/2 + ε_H + η_H/2 + s/2
    
    where:
    - ε_H = -Ḣ/H² (first slow-roll parameter)
    - η_H = d ln ε_H / dN (second slow-roll parameter)
    - s = d ln c_s / dN (sound speed running)
    """
    return 1.5 + eps_H + 0.5 * eta_H + 0.5 * s


def zpp_over_z(eta: float, nu: float) -> float:
    """
    Compute z''/z from eq:zpp-over-z.
    
    z''/z = (ν² - 1/4) / η²
    """
    return (nu**2 - 0.25) / (eta**2)


def _ms_rhs(eta: float, y: np.ndarray, k: float, c_s: float, nu: float) -> np.ndarray:
    """
    Right-hand side of Mukhanov-Sasaki equation as first-order system.
    
    v_k'' + (c_s² k² - z''/z) v_k = 0
    
    Written as:
    y = [Re(v), Im(v), Re(v'), Im(v')]
    """
    v = y[0] + 1j * y[1]
    vp = y[2] + 1j * y[3]

    omega2 = c_s**2 * k**2 - zpp_over_z(eta, nu)
    vpp = -omega2 * v

    return np.array([vp.real, vp.imag, vpp.real, vpp.imag], dtype=float)


def _bd_initial_conditions(eta_i: float, k: float, c_s: float) -> Tuple[complex, complex]:
    """
    Bunch-Davies initial conditions deep inside horizon.
    
    For c_s k |η_i| >> 1:
    v_k → e^{-i c_s k η} / √(2 c_s k)
    """
    omega = c_s * k
    v_i = np.exp(-1j * omega * eta_i) / np.sqrt(2.0 * omega)
    vp_i = -1j * omega * v_i
    return v_i, vp_i


def solve_ms_to_eta0(
    k: float,
    eta_0: float,
    c_s: float,
    eps_H: float,
    eta_H: float,
    s: float,
    eta_i: Optional[float] = None,
    subhorizon_factor: float = 100.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> Tuple[complex, complex]:
    """
    Solve Mukhanov-Sasaki equation from deep sub-horizon to η_0.
    
    Parameters:
    -----------
    k : float
        Comoving wavenumber
    eta_0 : float
        Conformal time of the act (negative)
    c_s : float
        Sound speed
    eps_H, eta_H, s : float
        Slow-roll parameters
    eta_i : float, optional
        Initial conformal time (if None, computed from subhorizon_factor)
    subhorizon_factor : float
        Factor for sub-horizon initial time: η_i = -factor/(c_s k)
        
    Returns:
    --------
    v0, vp0 : complex
        Mode function and its derivative at η_0
    """
    if eta_0 >= 0.0:
        raise ValueError("eta_0 must be negative conformal time.")

    nu = nu_from_slow_roll(eps_H, eta_H, s)

    if eta_i is None:
        # Choose eta_i so mode starts deep sub-horizon
        eta_i = -subhorizon_factor / max(c_s * k, 1e-14)

    if eta_i >= eta_0:
        raise ValueError("eta_i must be earlier than eta_0 (more negative).")

    v_i, vp_i = _bd_initial_conditions(eta_i, k, c_s)
    y0 = np.array([v_i.real, v_i.imag, vp_i.real, vp_i.imag], dtype=float)

    sol = solve_ivp(
        fun=lambda eta, y: _ms_rhs(eta, y, k, c_s, nu),
        t_span=(eta_i, eta_0),
        y0=y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"MS ODE solve failed: {sol.message}")

    y_end = sol.y[:, -1]
    v0 = y_end[0] + 1j * y_end[1]
    vp0 = y_end[2] + 1j * y_end[3]
    return v0, vp0


def _instantaneous_adiabatic_mode(
    eta_0: float,
    k: float,
    c_s: float,
    nu: float,
) -> Tuple[complex, complex]:
    """
    Instantaneous adiabatic basis mode at η_0.
    
    Used for Bogoliubov matching.
    """
    omega2 = c_s**2 * k**2 - zpp_over_z(eta_0, nu)
    omega = np.sqrt(omega2 + 0j)  # complex sqrt for super-horizon regime

    f = np.exp(-1j * omega * eta_0) / np.sqrt(2.0 * omega)
    fp = -1j * omega * f
    return f, fp


def bogoliubov_from_matching(
    v0: complex,
    vp0: complex,
    eta_0: float,
    k: float,
    c_s: float,
    eps_H: float,
    eta_H: float,
    s: float,
) -> Tuple[complex, complex]:
    """
    Extract Bogoliubov coefficients from matching at η_0.
    
    Uses Klein-Gordon inner products:
    α_k = i (f* v' - f'* v)
    β_k = -i (f v' - f' v)
    
    where f is the instantaneous adiabatic mode.
    
    Returns:
    --------
    alpha_k, beta_k : complex
        Bogoliubov coefficients satisfying |α|² - |β|² = 1
    """
    nu = nu_from_slow_roll(eps_H, eta_H, s)
    f, fp = _instantaneous_adiabatic_mode(eta_0, k, c_s, nu)

    # Klein-Gordon products
    alpha = 1j * (np.conj(f) * vp0 - np.conj(fp) * v0)
    beta = -1j * (f * vp0 - fp * v0)
    return alpha, beta


def ringdown_from_bogoliubov(
    alpha_k: complex,
    beta_k: complex,
    eps_H: float,
    eta_H: float,
    s: float,
    Gamma_k: float,
    delta_eta: float,
) -> Dict[str, float]:
    """
    Compute ring-down parameters from Bogoliubov coefficients.
    
    From eq:squeezing-params:
    - n̄_k = |β_k|² = sinh²(r_k)
    - r_k = arcsinh(√n̄_k)
    - θ_k = arg(β_k)
    
    From eq:ringdown:
    - φ_k = arg(α_k β_k*) = -θ_k (for real α_k > 0)
    - A(k) = O(ε_H, η_H, s) × visibility × damping
    
    Returns:
    --------
    dict with r_k, theta_k, nbar_k, phi_k, A_ring, damping
    """
    nbar = float(np.abs(beta_k) ** 2)
    r_k = float(np.arcsinh(np.sqrt(max(nbar, 0.0))))
    theta_k = float(np.angle(beta_k))
    phi_k = float(np.angle(alpha_k * np.conj(beta_k)))  # = -theta_k if alpha real positive

    # Decoherence damping from eq:IC-deco and eq:ringdown
    # NOTE: damping is applied SEPARATELY in the spectrum formula, not inside A_ring
    damping = float(np.exp(-Gamma_k * delta_eta))

    # A(k) = O(ε_H, η_H, s) from eq:ringdown
    # This is an evanescent ansatz since manuscript doesn't give explicit formula
    # A(k) = C_A × (|ε_H| + ½|η_H| + ½|s|) × visibility
    # where visibility = 2|α_k β_k*| / (|α_k|² + |β_k|²) is interference visibility
    
    # Small slow-roll prefactor for A(k) = O(ε_H, η_H, s)
    slowroll_pref = float(np.clip(abs(eps_H) + 0.5 * abs(eta_H) + 0.5 * abs(s), 0.0, 1.0))

    # Interference visibility from squeezing geometry
    # visibility = 2|α β*| / (|α|² + |β|²)
    denom = float(np.abs(alpha_k) ** 2 + np.abs(beta_k) ** 2 + 1e-30)
    visibility = float(2.0 * np.abs(alpha_k * np.conj(beta_k)) / denom)

    # A_ring WITHOUT damping (damping applied separately in spectrum formula)
    A_ring = slowroll_pref * visibility

    return {
        "r_k": r_k,
        "theta_k": theta_k,
        "nbar_k": nbar,
        "phi_k": phi_k,
        "A_ring": A_ring,
        "damping": damping,
    }


def compute_mode_result(
    k: float,
    eta_0: float,
    c_s: float,
    eps_H: float,
    eta_H: float,
    s: float,
    Gamma_k: float,
    delta_eta: float,
    eta_i: Optional[float] = None,
) -> ModeResult:
    """
    Full pipeline for single mode k:
    
    1. Solve MS equation with Bunch-Davies initial conditions up to η_0
    2. Match to instantaneous adiabatic basis at η_0
    3. Extract α_k, β_k (Bogoliubov coefficients)
    4. Compute r_k, θ_k, n̄_k, φ_k and ring-down amplitude A_ring
    
    Parameters:
    -----------
    k : float
        Comoving wavenumber
    eta_0 : float
        Conformal time of the act (negative)
    c_s : float
        Sound speed at horizon crossing
    eps_H : float
        First slow-roll parameter ε_H = -Ḣ/H²
    eta_H : float
        Second slow-roll parameter η_H = d ln ε_H / dN
    s : float
        Sound speed running s = d ln c_s / dN
    Gamma_k : float
        Decoherence rate Γ_k
    delta_eta : float
        Conformal time interval Δη = η_*(k) - η_0
        
    Returns:
    --------
    ModeResult with all physical quantities
    """
    v0, vp0 = solve_ms_to_eta0(
        k=k,
        eta_0=eta_0,
        c_s=c_s,
        eps_H=eps_H,
        eta_H=eta_H,
        s=s,
        eta_i=eta_i,
    )

    alpha_k, beta_k = bogoliubov_from_matching(
        v0=v0,
        vp0=vp0,
        eta_0=eta_0,
        k=k,
        c_s=c_s,
        eps_H=eps_H,
        eta_H=eta_H,
        s=s,
    )

    rd = ringdown_from_bogoliubov(
        alpha_k=alpha_k,
        beta_k=beta_k,
        eps_H=eps_H,
        eta_H=eta_H,
        s=s,
        Gamma_k=Gamma_k,
        delta_eta=delta_eta,
    )

    return ModeResult(
        alpha_k=alpha_k,
        beta_k=beta_k,
        r_k=rd["r_k"],
        theta_k=rd["theta_k"],
        nbar_k=rd["nbar_k"],
        phi_k=rd["phi_k"],
        A_ring=rd["A_ring"],
        damping=rd["damping"],
    )


def power_spectrum_with_ringdown(
    P0_k: float,
    k: float,
    eta_0: float,
    c_s: float,
    A_ring: float,
    phi_k: float,
    Gamma_k: float,
    delta_eta: float,
) -> float:
    """
    Compute power spectrum with ring-down (eq:ringdown).
    
    P_ζ(k) = P_ζ^(0)(k) · [1 + A(k)·cos(2c_s k η_0 + φ_k)·e^{-Γ_k Δη}]
    
    Parameters:
    -----------
    P0_k : float
        Base power spectrum P_ζ^(0)(k)
    k : float
        Comoving wavenumber
    eta_0 : float
        Conformal time of the act
    c_s : float
        Sound speed
    A_ring : float
        Ring-down amplitude A(k)
    phi_k : float
        Ring-down phase φ_k = arg(α_k β_k*)
    Gamma_k : float
        Decoherence rate
    delta_eta : float
        Conformal time interval Δη
        
    Returns:
    --------
    P_ζ(k) with ring-down correction
    """
    damp = np.exp(-Gamma_k * delta_eta)
    osc = np.cos(2.0 * c_s * k * eta_0 + phi_k)
    return float(P0_k * (1.0 + A_ring * osc * damp))


def compute_spectrum_array(
    k_array: np.ndarray,
    P0_array: np.ndarray,
    eta_0: float,
    c_s: float,
    eps_H: float,
    eta_H: float,
    s: float,
    Gamma_over_H: float,
    H_star: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectrum with physical ring-down for array of k values.
    
    Returns:
    --------
    P_zeta : np.ndarray
        Power spectrum with ring-down
    nbar_k : np.ndarray
        Occupancy n̄_k for each k
    phi_k : np.ndarray
        Ring-down phase φ_k for each k
    A_ring : np.ndarray
        Ring-down amplitude A(k) for each k
    """
    n_k = len(k_array)
    P_zeta = np.zeros(n_k)
    nbar_k = np.zeros(n_k)
    phi_k = np.zeros(n_k)
    A_ring_arr = np.zeros(n_k)
    
    for i, k in enumerate(k_array):
        # Conformal time at freeze-out: η_*(k) ≈ -1/(c_s k)
        eta_star = -1.0 / (c_s * k)
        delta_eta = max(eta_star - eta_0, 0.0)
        
        # Decoherence rate: Γ_k = (Γ/H) × H_*
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
            A_ring_arr[i] = result.A_ring
            
            P_zeta[i] = power_spectrum_with_ringdown(
                P0_k=P0_array[i],
                k=k,
                eta_0=eta_0,
                c_s=c_s,
                A_ring=result.A_ring,
                phi_k=result.phi_k,
                Gamma_k=Gamma_k,
                delta_eta=delta_eta,
            )
        except Exception:
            # Fallback to base spectrum if ODE fails
            P_zeta[i] = P0_array[i]
            nbar_k[i] = 0.0
            phi_k[i] = 0.0
            A_ring_arr[i] = 0.0
    
    return P_zeta, nbar_k, phi_k, A_ring_arr


if __name__ == "__main__":
    # Test with example parameters
    print("Testing Mukhanov-Sasaki solver...")
    
    params = dict(
        k=0.05,
        eta_0=-120.0,
        c_s=0.98,
        eps_H=0.01,
        eta_H=0.005,
        s=0.0,
        Gamma_k=0.2,
        delta_eta=3.0,
    )
    
    result = compute_mode_result(**params)
    
    print(f"α_k     = {result.alpha_k}")
    print(f"β_k     = {result.beta_k}")
    print(f"r_k     = {result.r_k:.6f}")
    print(f"θ_k     = {result.theta_k:.6f} rad = {np.degrees(result.theta_k):.2f}°")
    print(f"n̄_k     = {result.nbar_k:.6f}")
    print(f"φ_k     = {result.phi_k:.6f} rad = {np.degrees(result.phi_k):.2f}°")
    print(f"A_ring  = {result.A_ring:.6f}")
    print(f"damping = {result.damping:.6f}")
    
    # Verify φ_k = -θ_k (approximately, for real α_k)
    print(f"\nVerification: φ_k + θ_k = {result.phi_k + result.theta_k:.6f} (should be ~0)")
    
    # Verify |α|² - |β|² = 1
    norm = abs(result.alpha_k)**2 - abs(result.beta_k)**2
    print(f"Bogoliubov norm: |α|² - |β|² = {norm:.6f} (should be 1)")
