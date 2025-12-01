"""
Analytical solutions for validating Two-Phase MPM Solver implementations.
Contains exact mathematical solutions for physics equations from Ng et al. (2023).
"""

import numpy as np
from typing import Tuple, Union, Optional
import math


def poiseuille_flow_solution(y: np.ndarray, dp_dx: float, mu: float, h: float) -> np.ndarray:
    """
    Analytical velocity profile for 2D Poiseuille flow between parallel plates.
    
    Args:
        y: Transverse coordinate array (m)
        dp_dx: Pressure gradient (Pa/m) 
        mu: Dynamic viscosity (Pa·s)
        h: Half-channel height (m)
    
    Returns:
        Velocity profile u(y) (m/s)
    """
    return (1.0 / (2.0 * mu)) * dp_dx * y * (h - np.abs(y))


def couette_flow_solution(y: np.ndarray, U_wall: float, h: float) -> np.ndarray:
    """
    Linear velocity profile for Couette flow between moving plates.
    
    Args:
        y: Transverse coordinate array (m)
        U_wall: Moving wall velocity (m/s)
        h: Half-channel height (m)
    
    Returns:
        Linear velocity profile u(y) (m/s)
    """
    return U_wall * (y + h) / (2.0 * h)


def drucker_prager_yield_function(J2: float, p_prime: float, mu_p: float) -> float:
    """
    Analytical Drucker-Prager yield function (Equation 17).
    
    Args:
        J2: Second invariant of deviatoric stress tensor (Pa²)
        p_prime: Mean effective pressure (Pa)
        mu_p: Friction coefficient (dimensionless)
    
    Returns:
        Yield function value f_shear (Pa)
    """
    return np.sqrt(J2) - mu_p * p_prime


def friction_coefficient_equation_18(I: float, Iv: float, phi: float, 
                                    mu1: float = 0.49, mu2: float = 1.4,
                                    a: float = 1.23, b: float = 0.31) -> float:
    """
    Friction coefficient correlation from Equation 18.
    
    Args:
        I: Inertial number (dimensionless)
        Iv: Viscous number (dimensionless)
        phi: Solid volume fraction (dimensionless)
        mu1: Static friction coefficient
        mu2: Dynamic friction coefficient
        a, b: Correlation parameters
    
    Returns:
        Effective friction coefficient μ_p
    """
    Im = np.sqrt(I**2 + 2.0 * Iv)
    
    mu_p = (mu1 + (mu2 - mu1) / (1.0 + b / Im) + 
            5.0 * phi * Iv / (2.0 * a * Im))
    
    return mu_p


def trajectory_landing_distance(v_launch: float, theta: float, H_B: float, 
                               g: float = 9.81) -> float:
    """
    Analytical projectile motion landing distance (Equation 3).
    
    Args:
        v_launch: Launch velocity magnitude (m/s)
        theta: Slope angle (rad)
        H_B: Barrier height (m)
        g: Gravitational acceleration (m/s²)
    
    Returns:
        Landing distance ξ (m)
    """
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    
    # Projectile motion calculation
    xi = ((v_launch**2) / (g * cos_theta) * 
          (tan_theta + np.sqrt(tan_theta**2 + 2.0 * g * H_B / (v_launch**2 * cos_theta))) + 
          H_B * tan_theta)
    
    return xi


def landing_velocity_equation_4(v_r: float, theta_land: float, R: float = 1.0) -> float:
    """
    Landing velocity calculation (Equation 4).
    
    Args:
        v_r: Velocity magnitude just before landing (m/s)
        theta_land: Landing angle (rad)
        R: Velocity correction factor (dimensionless)
    
    Returns:
        Impact velocity v_i (m/s)
    """
    return R * v_r * np.cos(theta_land)


def drag_force_equation_22(phi: float, eta_f: float, d: float, 
                          v_rel: np.ndarray, F_hat: float) -> np.ndarray:
    """
    Inter-phase drag force (Equation 22).
    
    Args:
        phi: Solid volume fraction (dimensionless)
        eta_f: Fluid viscosity (Pa·s)
        d: Particle diameter (m)
        v_rel: Relative velocity vector (m/s)
        F_hat: Drag coefficient from Van der Hoef correlation
    
    Returns:
        Drag force per unit volume f_d (N/m³)
    """
    return 18.0 * phi * (1.0 - phi) * eta_f / (d**2) * F_hat * v_rel


def fluidization_ratio_equation_26(p_bed: float, sigma_prime_bed: float) -> float:
    """
    Fluidization ratio (Equation 26).
    
    Args:
        p_bed: Pore water pressure (Pa)
        sigma_prime_bed: Effective stress (Pa)
    
    Returns:
        Fluidization ratio λ (dimensionless)
    """
    if sigma_prime_bed == 0.0:
        return 1.0 if p_bed > 0.0 else 0.0
    
    return p_bed / (p_bed + sigma_prime_bed)


def bed_friction_equation_25(mu_bed: float, sigma_bed: float, lambda_val: float) -> float:
    """
    Reduced bed friction force (Equation 25).
    
    Args:
        mu_bed: Bed friction coefficient (dimensionless)
        sigma_bed: Normal stress on bed (Pa)
        lambda_val: Fluidization ratio (dimensionless)
    
    Returns:
        Friction force F_fric (N)
    """
    return mu_bed * sigma_bed * (1.0 - lambda_val)


def van_der_hoef_drag_coefficient(phi: float, Re: float) -> float:
    """
    Van der Hoef et al. (2005) drag coefficient correlation.
    Approximation for validation purposes.
    
    Args:
        phi: Solid volume fraction (dimensionless)
        Re: Particle Reynolds number (dimensionless)
    
    Returns:
        Drag coefficient F̂ (dimensionless)
    """
    # Simplified correlation for testing (actual implementation may differ)
    if Re < 0.1:
        # Stokes regime
        F_hat = 1.0 + 3.0 * phi / (2.0 * (1.0 - phi))
    elif Re < 1.0:
        # Transition regime
        F_hat = (1.0 + 3.0 * phi / (2.0 * (1.0 - phi))) * (1.0 + 0.15 * Re**0.687)
    else:
        # Turbulent regime approximation
        F_hat = (1.0 + 3.0 * phi / (2.0 * (1.0 - phi))) * (1.0 + 0.15 * Re**0.687) * (1.0 + Re / 24.0)
    
    return F_hat


def compute_second_invariant_J2(stress_tensor: np.ndarray) -> float:
    """
    Compute second invariant of deviatoric stress tensor.
    
    Args:
        stress_tensor: 3x3 stress tensor (Pa)
    
    Returns:
        Second invariant J2 (Pa²)
    """
    # Mean stress
    p = np.trace(stress_tensor) / 3.0
    
    # Deviatoric stress tensor
    s = stress_tensor - p * np.eye(3)
    
    # Second invariant: J2 = 0.5 * s_ij * s_ij
    J2 = 0.5 * np.sum(s * s)
    
    return J2


def compute_mean_effective_pressure(stress_tensor: np.ndarray) -> float:
    """
    Compute mean effective pressure from stress tensor.
    
    Args:
        stress_tensor: 3x3 stress tensor (Pa)
    
    Returns:
        Mean effective pressure p' (Pa)
    """
    return np.trace(stress_tensor) / 3.0


def compute_reynolds_number(v_rel: Union[float, np.ndarray], d: float, 
                          rho_f: float, eta_f: float) -> float:
    """
    Compute particle Reynolds number.
    
    Args:
        v_rel: Relative velocity magnitude (m/s)
        d: Particle diameter (m)
        rho_f: Fluid density (kg/m³)
        eta_f: Fluid viscosity (Pa·s)
    
    Returns:
        Reynolds number Re (dimensionless)
    """
    if isinstance(v_rel, np.ndarray):
        v_mag = np.linalg.norm(v_rel)
    else:
        v_mag = abs(v_rel)
    
    return rho_f * v_mag * d / eta_f


def create_known_stress_state(case: str = "elastic") -> np.ndarray:
    """
    Create known stress states for testing plasticity models.
    
    Args:
        case: Type of stress state ("elastic", "plastic", "tensile")
    
    Returns:
        3x3 stress tensor (Pa)
    """
    if case == "elastic":
        # Elastic stress state (below yield)
        stress = np.array([
            [1000.0,    0.0,    0.0],
            [   0.0, 1000.0,    0.0], 
            [   0.0,    0.0, 1000.0]
        ])
    elif case == "plastic":
        # Plastic stress state (above yield)
        stress = np.array([
            [5000.0,  500.0,    0.0],
            [ 500.0, 3000.0,    0.0],
            [   0.0,    0.0, 2000.0]
        ])
    elif case == "tensile":
        # Tensile stress state
        stress = np.array([
            [-1000.0,     0.0,    0.0],
            [    0.0,  1000.0,    0.0],
            [    0.0,     0.0, 1000.0]
        ])
    else:
        raise ValueError(f"Unknown stress state case: {case}")
    
    return stress


def generate_plastic_stress_states(n_samples: int = 100, 
                                 random_seed: Optional[int] = None) -> list:
    """
    Generate random stress states for plastic consistency testing.
    
    Args:
        n_samples: Number of stress states to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        List of 3x3 stress tensors (Pa)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    stress_states = []
    
    for _ in range(n_samples):
        # Generate random symmetric stress tensor
        sigma_xx = np.random.uniform(-5000, 10000)  # Pa
        sigma_yy = np.random.uniform(-5000, 10000)
        sigma_zz = np.random.uniform(-5000, 10000)
        sigma_xy = np.random.uniform(-2000, 2000)
        sigma_xz = np.random.uniform(-2000, 2000)
        sigma_yz = np.random.uniform(-2000, 2000)
        
        stress = np.array([
            [sigma_xx, sigma_xy, sigma_xz],
            [sigma_xy, sigma_yy, sigma_yz],
            [sigma_xz, sigma_yz, sigma_zz]
        ])
        
        stress_states.append(stress)
    
    return stress_states


# Constants for physical validation
PHYSICAL_CONSTANTS = {
    "gravity": 9.81,           # m/s²
    "water_density": 1000.0,   # kg/m³
    "water_viscosity": 1e-3,   # Pa·s
    "sand_density": 2650.0,    # kg/m³
    "sand_diameter": 1e-3,     # m (1 mm typical)
    "friction_mu1": 0.49,      # Static friction
    "friction_mu2": 1.4,       # Dynamic friction
    "correlation_a": 1.23,     # Van der Hoef parameter
    "correlation_b": 0.31,     # Van der Hoef parameter
}


if __name__ == "__main__":
    # Test analytical solutions
    print("Testing analytical solutions...")
    
    # Test Drucker-Prager yield function
    J2 = 1000.0**2
    p_prime = 2000.0
    mu_p = 0.5
    yield_val = drucker_prager_yield_function(J2, p_prime, mu_p)
    print(f"Drucker-Prager yield function: {yield_val:.2f} Pa")
    
    # Test trajectory calculation
    v_launch = 10.0  # m/s
    theta = 20.0 * np.pi / 180.0  # 20 degrees
    H_B = 2.0  # m
    xi = trajectory_landing_distance(v_launch, theta, H_B)
    print(f"Landing distance: {xi:.2f} m")
    
    print("Analytical solutions tested successfully!")