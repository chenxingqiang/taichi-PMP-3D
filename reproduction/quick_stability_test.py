"""
Quick Numerical Stability Test
Tests the simulation with paper-accurate parameters to identify convergence issues
"""

import taichi as ti
import numpy as np
import yaml
import logging
from datetime import datetime

# Import our modules
from incompressible_mpm_solver import IncompressibleMPMSolver
from barrier_model import BarrierModel

def setup_logging():
    """Setup logging for stability test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/stability_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_paper_parameters():
    """Test with paper-accurate parameters."""
    logger = setup_logging()
    logger.info("=== QUICK STABILITY TEST WITH PAPER PARAMETERS ===")
    
    # Load paper-accurate configuration
    with open("physics_config_paper_accurate.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded")
    
    # Initialize Taichi with CPU for stability
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)
    logger.info("Taichi initialized with CPU backend")
    
    # Extract parameters
    domain_length = config['simulation']['domain_length']
    domain_width = config['simulation']['domain_width']
    domain_height = config['simulation']['domain_height']
    mesh_barrier_ratio = config['numerics']['mesh_barrier_ratio']
    barrier_height = config['simulation']['barrier_height']
    particles_per_cell = config['numerics']['particles_per_cell']
    dt = config['numerics']['max_timestep']
    
    # Calculate grid
    mesh_size = mesh_barrier_ratio * barrier_height
    nx = int(domain_length / mesh_size) + 1
    ny = int(domain_width / mesh_size) + 1
    nz = int(domain_height / mesh_size) + 1
    
    logger.info(f"Domain: {domain_length} x {domain_width} x {domain_height} m")
    logger.info(f"Grid: {nx} x {ny} x {nz} cells")
    logger.info(f"Mesh size: {mesh_size} m")
    logger.info(f"Particles per cell: {particles_per_cell}")
    logger.info(f"Time step: {dt} s")
    
    # Check if grid is too large
    total_cells = nx * ny * nz
    estimated_particles = total_cells * particles_per_cell
    
    logger.info(f"Total cells: {total_cells:,}")
    logger.info(f"Estimated particles: {estimated_particles:,}")
    
    if total_cells > 1000000:
        logger.warning(f"Grid is very large: {total_cells:,} cells")
        logger.warning("This may cause memory issues or slow performance")
    
    if estimated_particles > 10000000:
        logger.warning(f"Estimated particles is very large: {estimated_particles:,}")
        logger.warning("This may cause memory issues")
    
    # Test with smaller domain first
    logger.info("Testing with reduced domain for stability...")
    
    # Reduce domain size for testing
    test_domain_length = min(domain_length, 2.0)  # Max 2m
    test_domain_width = min(domain_width, 0.5)    # Max 0.5m
    test_domain_height = min(domain_height, 0.5)  # Max 0.5m
    
    test_nx = int(test_domain_length / mesh_size) + 1
    test_ny = int(test_domain_width / mesh_size) + 1
    test_nz = int(test_domain_height / mesh_size) + 1
    
    test_total_cells = test_nx * test_ny * test_nz
    test_estimated_particles = test_total_cells * particles_per_cell
    
    logger.info(f"Test domain: {test_domain_length} x {test_domain_width} x {test_domain_height} m")
    logger.info(f"Test grid: {test_nx} x {test_ny} x {test_nz} cells")
    logger.info(f"Test total cells: {test_total_cells:,}")
    logger.info(f"Test estimated particles: {test_estimated_particles:,}")
    
    try:
        # Initialize MPM solver with test domain
        logger.info("Initializing MPM solver...")
        solver = IncompressibleMPMSolver(
            nx=test_nx, ny=test_ny, nz=test_nz,
            dx=mesh_size,
            rho=config['fluid_phase']['density'],
            mu=config['fluid_phase']['viscosity'],
            g=config['simulation']['gravity'],
            dt=dt,
            max_particles=min(1000000, test_estimated_particles * 2)
        )
        logger.info("MPM solver initialized successfully")
        
        # Initialize particles
        logger.info("Initializing particles...")
        debris_length = min(test_domain_length * 0.3, 0.5)  # 30% of domain or 0.5m max
        debris_height = test_domain_height * 0.8
        
        solver.initialize_particles_dam_break(
            x_min=0.0, x_max=debris_length,
            y_min=0.0, y_max=test_domain_width,
            z_min=0.0, z_max=debris_height,
            ppc=particles_per_cell
        )
        
        total_particles = solver.n_particles[None]
        logger.info(f"Initialized {total_particles} particles")
        
        # Run stability test
        logger.info("Running stability test...")
        max_steps = 1000
        stability_issues = []
        
        for step in range(max_steps):
            try:
                # Run single step
                solver.step()
                
                # Check for numerical issues
                velocities = solver.v.to_numpy()
                # Note: pressure is managed by pcg_solver, not directly accessible
                
                # Check for NaN or infinite values
                has_nan_vel = np.any(np.isnan(velocities))
                has_inf_vel = np.any(np.isinf(velocities))
                has_nan_pressure = False  # Placeholder - pressure not directly accessible
                has_inf_pressure = False  # Placeholder - pressure not directly accessible
                
                if has_nan_vel or has_inf_vel or has_nan_pressure or has_inf_pressure:
                    stability_issues.append({
                        'step': step,
                        'nan_vel': has_nan_vel,
                        'inf_vel': has_inf_vel,
                        'nan_pressure': has_nan_pressure,
                        'inf_pressure': has_inf_pressure
                    })
                    logger.error(f"Numerical issue at step {step}: NaN/Inf detected")
                    break
                
                # Check velocity magnitude
                vel_magnitude = np.linalg.norm(velocities, axis=1)
                max_vel = np.max(vel_magnitude)
                
                if max_vel > 100.0:  # Unrealistic velocity
                    stability_issues.append({
                        'step': step,
                        'max_velocity': max_vel,
                        'issue': 'velocity_explosion'
                    })
                    logger.error(f"Velocity explosion at step {step}: max_vel = {max_vel}")
                    break
                
                # Log progress
                if step % 100 == 0:
                    logger.info(f"Step {step}: max_vel = {max_vel:.6f} m/s")
                
            except Exception as e:
                stability_issues.append({
                    'step': step,
                    'error': str(e)
                })
                logger.error(f"Error at step {step}: {str(e)}")
                break
        
        # Report results
        logger.info("=== STABILITY TEST RESULTS ===")
        if not stability_issues:
            logger.info("✅ STABILITY TEST PASSED")
            logger.info(f"Completed {max_steps} steps without numerical issues")
        else:
            logger.error("❌ STABILITY TEST FAILED")
            for issue in stability_issues:
                logger.error(f"Issue at step {issue['step']}: {issue}")
        
        return len(stability_issues) == 0
        
    except Exception as e:
        logger.error(f"Failed to initialize solver: {str(e)}")
        return False

def test_parameter_sensitivity():
    """Test parameter sensitivity for stability."""
    logger = setup_logging()
    logger.info("=== PARAMETER SENSITIVITY TEST ===")
    
    # Test different time steps
    time_steps = [1e-6, 1e-5, 1e-4]
    cfl_factors = [0.001, 0.01, 0.1]
    
    for dt in time_steps:
        for cfl in cfl_factors:
            logger.info(f"Testing dt={dt}, CFL={cfl}")
            # This would run a mini-simulation with these parameters
            # For now, just log the combination
            logger.info(f"  dt={dt}, CFL={cfl} - would need full test")

def main():
    """Main function for stability testing."""
    print("=== QUICK NUMERICAL STABILITY TEST ===")
    print("Testing paper-accurate parameters for numerical stability")
    print("="*50)
    
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run stability test
    success = test_paper_parameters()
    
    if success:
        print("✅ Stability test passed!")
        print("The paper parameters appear to be numerically stable.")
        print("You can proceed with the full simulation.")
    else:
        print("❌ Stability test failed!")
        print("Numerical issues detected. Check logs for details.")
        print("Consider adjusting numerical parameters.")
    
    # Run parameter sensitivity test
    test_parameter_sensitivity()
    
    print("\nCheck logs/ directory for detailed results.")

if __name__ == "__main__":
    main()
