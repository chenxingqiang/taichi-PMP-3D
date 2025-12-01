"""
Test utilities and helper functions for Two-Phase MPM Solver test suite.
Common functions used across multiple test modules.
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, Optional, Union, List
from pathlib import Path
import sys

# Add src directory for imports (when implemented)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def create_empty_grid_fields(nx: int = 32, ny: int = 32, nz: int = 32) -> Dict[str, np.ndarray]:
    """
    Create empty grid field arrays for testing.
    
    Args:
        nx, ny, nz: Grid dimensions
        
    Returns:
        Dictionary of grid field arrays
    """
    return {
        "velocity": np.zeros((nx, ny, nz, 3)),
        "momentum": np.zeros((nx, ny, nz, 3)),
        "mass": np.zeros((nx, ny, nz)),
        "pressure": np.zeros((nx, ny, nz)),
        "volume_fraction": np.zeros((nx, ny, nz))
    }


def initialize_test_particles(n_particles: int = 1000, 
                            domain_size: Tuple[float, float, float] = (3.2, 3.2, 3.2)) -> Dict[str, np.ndarray]:
    """
    Initialize particle arrays for testing.
    
    Args:
        n_particles: Number of particles
        domain_size: Domain size (Lx, Ly, Lz) in meters
        
    Returns:
        Dictionary of particle arrays
    """
    np.random.seed(42)  # Reproducible initialization
    
    # Random positions within domain
    positions = np.random.uniform([0, 0, 0], domain_size, (n_particles, 3))
    
    # Random velocities (small magnitude)
    velocities = np.random.normal(0, 0.1, (n_particles, 3))
    
    # Uniform particle mass
    masses = np.ones(n_particles) * 1e-6  # kg
    
    # Identity stress tensors
    stress = np.tile(np.eye(3), (n_particles, 1, 1)) * 1000.0  # Pa
    
    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "stress": stress,
        "volume": np.ones(n_particles) * 1e-9  # m³
    }


def compute_total_momentum(particles: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute total momentum from particle data.
    
    Args:
        particles: Dictionary with particle arrays
        
    Returns:
        Total momentum vector (3D)
    """
    masses = particles["masses"]
    velocities = particles["velocities"]
    return np.sum(masses[:, np.newaxis] * velocities, axis=0)


def compute_total_mass(particles: Dict[str, np.ndarray]) -> float:
    """
    Compute total mass from particle data.
    
    Args:
        particles: Dictionary with particle arrays
        
    Returns:
        Total mass
    """
    return np.sum(particles["masses"])


def compute_total_grid_momentum(grid_fields: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute total momentum from grid fields.
    
    Args:
        grid_fields: Dictionary with grid arrays
        
    Returns:
        Total momentum vector (3D)
    """
    return np.sum(grid_fields["momentum"], axis=(0, 1, 2))


def compute_total_grid_mass(grid_fields: Dict[str, np.ndarray]) -> float:
    """
    Compute total mass from grid fields.
    
    Args:
        grid_fields: Dictionary with grid arrays
        
    Returns:
        Total mass
    """
    return np.sum(grid_fields["mass"])


def compute_velocity_divergence(velocity_field: np.ndarray, dx: float = 0.1) -> np.ndarray:
    """
    Compute velocity divergence using finite differences.
    
    Args:
        velocity_field: 4D array (nx, ny, nz, 3)
        dx: Grid spacing
        
    Returns:
        Divergence field (3D array)
    """
    u = velocity_field[..., 0]  # x-velocity
    v = velocity_field[..., 1]  # y-velocity
    w = velocity_field[..., 2]  # z-velocity
    
    # Central differences
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    dwdz = np.gradient(w, dx, axis=2)
    
    return dudx + dvdy + dwdz


def initialize_divergent_velocity_field(shape: Tuple[int, int, int], 
                                      divergence_magnitude: float = 1.0) -> np.ndarray:
    """
    Create a velocity field with known divergence for testing.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        divergence_magnitude: Target divergence magnitude
        
    Returns:
        Velocity field with non-zero divergence
    """
    nx, ny, nz = shape
    velocity_field = np.zeros((nx, ny, nz, 3))
    
    # Create linearly expanding flow
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    velocity_field[..., 0] = divergence_magnitude * X
    velocity_field[..., 1] = divergence_magnitude * Y
    velocity_field[..., 2] = divergence_magnitude * Z
    
    return velocity_field


def create_analytical_poisson_rhs(shape: Tuple[int, int, int], 
                                 dx: float = 0.1) -> np.ndarray:
    """
    Create right-hand side for Poisson equation with analytical solution.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        dx: Grid spacing
        
    Returns:
        RHS array for Poisson equation
    """
    nx, ny, nz = shape
    
    # Create coordinate arrays
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dx, ny)
    z = np.linspace(0, (nz-1)*dx, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Analytical solution: phi = sin(πx/L) * sin(πy/L) * sin(πz/L)
    L = max(nx, ny, nz) * dx
    phi = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L) * np.sin(np.pi * Z / L)
    
    # RHS is Laplacian of phi: ∇²φ = -3π²/L² * φ
    rhs = -3.0 * (np.pi / L)**2 * phi
    
    return rhs


def solve_poisson_analytical(rhs: np.ndarray, dx: float = 0.1) -> np.ndarray:
    """
    Analytical solution for the Poisson equation created by create_analytical_poisson_rhs.
    
    Args:
        rhs: Right-hand side array
        dx: Grid spacing
        
    Returns:
        Analytical solution
    """
    nx, ny, nz = rhs.shape
    
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dx, ny) 
    z = np.linspace(0, (nz-1)*dx, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Analytical solution
    L = max(nx, ny, nz) * dx
    solution = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L) * np.sin(np.pi * Z / L)
    
    return solution


def benchmark_timer(name: str = "Operation"):
    """
    Context manager for timing code execution.
    
    Args:
        name: Name of the operation being timed
        
    Usage:
        with benchmark_timer("Matrix multiplication"):
            result = np.dot(A, B)
    """
    class Timer:
        def __init__(self, name):
            self.name = name
            
        def __enter__(self):
            self.start = time.time()
            return self
            
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
            print(f"{self.name} took {self.elapsed:.4f} seconds")
    
    return Timer(name)


def get_gpu_memory_usage() -> int:
    """
    Get GPU memory usage in bytes.
    
    Returns:
        Memory usage in bytes (placeholder implementation)
    """
    # Placeholder implementation - would use actual GPU monitoring
    # This would integrate with Taichi's memory profiling or CUDA/OpenCL APIs
    return 0  # TODO: Implement actual GPU memory monitoring


def get_gpu_type() -> str:
    """
    Get GPU type identifier.
    
    Returns:
        GPU type string
    """
    # Placeholder implementation
    return "Unknown"  # TODO: Implement actual GPU detection


def initialize_counter_flow(n_solid: int = 1000, n_fluid: int = 1000) -> Tuple[Dict, Dict]:
    """
    Initialize solid and fluid particles with relative motion for drag testing.
    
    Args:
        n_solid: Number of solid particles
        n_fluid: Number of fluid particles
        
    Returns:
        Tuple of (solid_particles, fluid_particles) dictionaries
    """
    # Solid particles moving right
    solid_particles = initialize_test_particles(n_solid)
    solid_particles["velocities"][:, 0] = 2.0  # 2 m/s to the right
    
    # Fluid particles moving left
    fluid_particles = initialize_test_particles(n_fluid)
    fluid_particles["velocities"][:, 0] = -1.0  # 1 m/s to the left
    
    return solid_particles, fluid_particles


def initialize_relative_velocity(v_rel_initial: float) -> Tuple[Dict, Dict]:
    """
    Initialize particles with specific relative velocity.
    
    Args:
        v_rel_initial: Initial relative velocity magnitude
        
    Returns:
        Tuple of (solid_particles, fluid_particles) dictionaries
    """
    solid_particles = initialize_test_particles(500)
    fluid_particles = initialize_test_particles(500)
    
    # Set relative velocity in x-direction
    solid_particles["velocities"][:, 0] = v_rel_initial / 2
    fluid_particles["velocities"][:, 0] = -v_rel_initial / 2
    
    return solid_particles, fluid_particles


def compute_relative_velocity(solid_particles: Dict, fluid_particles: Dict) -> float:
    """
    Compute average relative velocity between phases.
    
    Args:
        solid_particles: Solid particle dictionary
        fluid_particles: Fluid particle dictionary
        
    Returns:
        Average relative velocity magnitude
    """
    v_solid_avg = np.mean(solid_particles["velocities"], axis=0)
    v_fluid_avg = np.mean(fluid_particles["velocities"], axis=0)
    
    return np.linalg.norm(v_solid_avg - v_fluid_avg)


def setup_dam_break_initial_condition(domain_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup initial conditions for dam break simulation.
    
    Args:
        domain_config: Domain configuration dictionary
        
    Returns:
        Initial condition dictionary
    """
    # Extract parameters
    material = domain_config.get("material", "sand_water_mixture")
    initial_height = domain_config.get("initial_height", 0.3)
    flume_length = domain_config.get("flume_length", 5.0)
    flume_width = domain_config.get("flume_width", 0.2)
    
    # Initial particle region (dam)
    dam_length = 1.0  # m
    
    initial_conditions = {
        "material": material,
        "particle_region": {
            "x_min": 0.0,
            "x_max": dam_length,
            "y_min": 0.0, 
            "y_max": flume_width,
            "z_min": 0.0,
            "z_max": initial_height
        },
        "initial_velocity": np.array([0.0, 0.0, 0.0]),  # Initially at rest
        "gravity": np.array([0.0, 0.0, -9.81])
    }
    
    return initial_conditions


def setup_500m3_debris_flow(n_particles: int = 50000) -> Dict[str, Any]:
    """
    Setup large-scale debris flow for memory testing.
    
    Args:
        n_particles: Number of particles
        
    Returns:
        Large-scale configuration dictionary
    """
    # 500 m³ debris flow: ~10m x 10m x 5m initial geometry
    config = {
        "n_particles": n_particles,
        "domain_size": (50.0, 20.0, 20.0),  # Large computational domain
        "initial_volume": 500.0,  # m³
        "particle_density": 2650.0,  # kg/m³
        "total_mass": 500.0 * 2650.0,  # kg
        "memory_target": 8 * 1024**3  # 8 GB
    }
    
    return config


def run_to_time(target_time: float, dt: float = 1e-4) -> int:
    """
    Compute number of steps needed to reach target time.
    
    Args:
        target_time: Target simulation time
        dt: Time step size
        
    Returns:
        Number of time steps
    """
    return int(target_time / dt)


def run_impact_simulation(domain_config: Dict[str, Any], 
                         target_time: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder function to run impact simulation and return time history.
    
    Args:
        domain_config: Domain configuration
        target_time: Simulation time
        
    Returns:
        Tuple of (times, forces) arrays
    """
    # Placeholder implementation - would use actual solver
    times = np.linspace(0, target_time, 200)
    
    # Generate realistic-looking force history
    impact_start = 0.5
    impact_duration = 0.3
    
    forces = np.zeros_like(times)
    mask = (times >= impact_start) & (times <= impact_start + impact_duration)
    
    t_impact = times[mask] - impact_start
    forces[mask] = 3000 * np.exp(-(t_impact - 0.1)**2 / 0.02) * np.sin(10 * t_impact)
    forces[forces < 0] = 0  # No negative forces
    
    return times, forces


class MockReturnMappingResult:
    """Mock result class for return mapping algorithm testing."""
    
    def __init__(self, converged: bool = True, iterations: int = 5, residual: float = 1e-12):
        self.converged = converged
        self.iterations = iterations
        self.residual = residual


def assert_arrays_close(actual: np.ndarray, expected: np.ndarray, 
                       rtol: float = 1e-12, atol: float = 1e-14, 
                       msg: str = "Arrays not equal"):
    """
    Assert that two arrays are close within specified tolerances.
    
    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Error message
    """
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)


def assert_scalar_close(actual: float, expected: float, 
                       rtol: float = 1e-12, atol: float = 1e-14,
                       msg: str = "Scalars not equal"):
    """
    Assert that two scalars are close within specified tolerance.
    
    Args:
        actual: Actual value
        expected: Expected value  
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Error message
    """
    if expected == 0.0:
        assert abs(actual) < atol, f"{msg}: {actual} != {expected}"
    else:
        relative_error = abs(actual - expected) / abs(expected)
        assert relative_error < rtol or abs(actual - expected) < atol, \
               f"{msg}: {actual} != {expected} (relative error: {relative_error:.2e})"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test grid field creation
    grid_fields = create_empty_grid_fields(8, 8, 8)
    print(f"Created grid fields with shape: {grid_fields['velocity'].shape}")
    
    # Test particle initialization
    particles = initialize_test_particles(100)
    print(f"Initialized {len(particles['masses'])} particles")
    
    # Test momentum computation
    momentum = compute_total_momentum(particles)
    print(f"Total momentum: {momentum}")
    
    # Test divergence computation
    vel_field = initialize_divergent_velocity_field((16, 16, 16), 2.0)
    div_field = compute_velocity_divergence(vel_field)
    print(f"Max divergence: {np.max(div_field):.2f}")
    
    print("Utility functions tested successfully!")