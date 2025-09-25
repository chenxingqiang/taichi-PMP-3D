"""
Enhanced Two-Phase MPM Debris Flow Simulation with Detailed Logging
Diagnoses convergence issues and numerical stability problems
"""

import taichi as ti
import numpy as np
import yaml
import time
import os
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt

# Import our modules
from incompressible_mpm_solver import IncompressibleMPMSolver
from barrier_model import BarrierModel
from output_metrics import OutputMetricsCalculator
from level_set_method import LevelSetMethod

class DetailedLoggingSimulation:
    """
    Enhanced simulation with comprehensive logging for convergence diagnosis.
    """

    def __init__(self, config_path: str = "physics_config.yaml"):
        """Initialize simulation with detailed logging."""
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger.info("=== SIMULATION INITIALIZATION ===")
        self.logger.info(f"Config loaded from: {config_path}")
        
        # Initialize Taichi with detailed logging
        self._initialize_taichi()
        
        # Simulation parameters
        self.dt = self.config['numerics']['max_timestep']
        self.total_time = self.config['simulation']['total_time']
        self.output_interval = self.config['numerics']['vtk_output_interval']
        
        # Domain setup
        self.domain_length = self.config['simulation']['domain_length']
        self.domain_width = self.config['simulation']['domain_width']
        self.domain_height = self.config['simulation']['domain_height']
        
        self.logger.info(f"Domain: {self.domain_length} x {self.domain_width} x {self.domain_height} m")
        self.logger.info(f"Time step: {self.dt} s, Total time: {self.total_time} s")
        
        # Grid setup
        self.mesh_barrier_ratio = self.config['numerics']['mesh_barrier_ratio']
        self.barrier_height = self.config['simulation']['barrier_height']
        self.mesh_size = self.mesh_barrier_ratio * self.barrier_height
        
        self.nx = int(self.domain_length / self.mesh_size) + 1
        self.ny = int(self.domain_width / self.mesh_size) + 1
        self.nz = int(self.domain_height / self.mesh_size) + 1
        
        self.logger.info(f"Grid: {self.nx} x {self.ny} x {self.nz} cells")
        self.logger.info(f"Mesh size: {self.mesh_size} m")
        self.logger.info(f"Mesh-barrier ratio: {self.mesh_barrier_ratio}")
        
        # Initialize components
        self._initialize_components()
        
        # Initialize particles
        self._initialize_particles()
        
        # Initialize level set
        self.solver.level_set_method.initialize_box(
            x_min=0.0, x_max=self.domain_length,
            y_min=0.0, y_max=self.domain_width,
            z_min=0.0, z_max=self.domain_height
        )
        
        self.is_initialized = True
        self.logger.info("=== SIMULATION INITIALIZATION COMPLETE ===")

    def _setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/simulation_detailed_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger('MPMSimulation')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_filename}")

    def _initialize_taichi(self):
        """Initialize Taichi with detailed logging."""
        self.logger.info("Initializing Taichi...")
        
        # Log Taichi configuration
        self.logger.info(f"Taichi version: {ti.__version__}")
        # Note: ti.archs_supported() is not available in newer Taichi versions
        
        # Initialize with CPU for stability debugging
        ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)
        
        self.logger.info("Taichi initialized with CPU backend for debugging")

    def _initialize_components(self):
        """Initialize simulation components with detailed logging."""
        self.logger.info("Initializing simulation components...")
        
        # Initialize MPM solver
        self.logger.info("Initializing MPM solver...")
        self.solver = IncompressibleMPMSolver(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.mesh_size,
            rho=self.config['fluid_phase']['density'],
            mu=self.config['fluid_phase']['viscosity'],
            g=self.config['simulation']['gravity'],
            dt=self.dt,
            max_particles=1000000
        )
        self.logger.info("MPM solver initialized")
        
        # Initialize barrier model
        self.logger.info("Initializing barrier model...")
        self.barrier_model = BarrierModel(
            barrier_height=self.config['simulation']['barrier_height'],
            barrier_positions=self.config['simulation']['barrier_positions'],
            contact_stiffness=self.config['numerics']['contact_stiffness'],
            contact_damping=self.config['numerics']['contact_damping'],
            friction_coefficient=self.config['solid_phase']['static_friction']
        )
        self.logger.info("Barrier model initialized")
        
        # Initialize metrics calculator
        self.logger.info("Initializing metrics calculator...")
        self.metrics_calculator = OutputMetricsCalculator("physics_config.yaml")
        self.logger.info("Metrics calculator initialized")

    def _initialize_particles(self):
        """Initialize debris flow particles with detailed logging."""
        self.logger.info("Initializing particles...")
        
        # Debris flow initial geometry
        debris_length = self.config['simulation']['debris_volume'] / (self.domain_width * self.domain_height)
        debris_height = self.domain_height * 0.8  # 80% of domain height
        
        self.logger.info(f"Debris geometry: length={debris_length:.3f} m, height={debris_height:.3f} m")
        
        # Initialize particles in debris flow region
        self.solver.initialize_particles_dam_break(
            x_min=0.0, x_max=debris_length,
            y_min=0.0, y_max=self.domain_width,
            z_min=0.0, z_max=debris_height,
            ppc=self.config['numerics']['particles_per_cell']
        )
        
        self.total_particles = self.solver.n_particles[None]
        self.logger.info(f"Initialized {self.total_particles} particles")
        self.logger.info(f"Particles per cell: {self.config['numerics']['particles_per_cell']}")

    def run_simulation_with_detailed_logging(self, output_dir: str = "simulation_output") -> Dict[str, Any]:
        """
        Run simulation with comprehensive logging and convergence monitoring.
        """
        if not self.is_initialized:
            self.initialize_simulation()
        
        self.logger.info("=== STARTING SIMULATION ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation state tracking
        self.simulation_state = {
            'time': 0.0,
            'step': 0,
            'convergence_history': [],
            'pressure_residuals': [],
            'velocity_stats': [],
            'particle_stats': [],
            'barrier_forces': [],
            'errors': []
        }
        
        # Run simulation
        start_time = time.time()
        
        try:
            while self.simulation_state['time'] < self.total_time:
                step_start = time.time()
                
                # Log step information
                if self.simulation_state['step'] % 100 == 0:
                    self.logger.info(f"Step {self.simulation_state['step']}, Time: {self.simulation_state['time']:.6f} s")
                
                # Run single time step with detailed monitoring
                step_success = self._run_single_step_with_monitoring()
                
                if not step_success:
                    self.logger.error(f"Step {self.simulation_state['step']} failed!")
                    break
                
                # Update simulation state
                self.simulation_state['time'] += self.dt
                self.simulation_state['step'] += 1
                
                # Output and analysis
                if self.simulation_state['step'] % self.output_interval == 0:
                    self._output_and_analyze(output_dir)
                
                # Check for convergence issues
                self._check_convergence_issues()
                
                step_time = time.time() - step_start
                if step_time > 1.0:  # Log slow steps
                    self.logger.warning(f"Slow step {self.simulation_state['step']}: {step_time:.3f} s")
        
        except Exception as e:
            self.logger.error(f"Simulation failed at step {self.simulation_state['step']}: {str(e)}")
            self.logger.error("Exception details:", exc_info=True)
            raise
        
        total_time = time.time() - start_time
        self.logger.info(f"=== SIMULATION COMPLETED ===")
        self.logger.info(f"Total simulation time: {total_time:.2f} s")
        self.logger.info(f"Total steps: {self.simulation_state['step']}")
        if self.simulation_state['step'] > 0:
            self.logger.info(f"Average step time: {total_time/self.simulation_state['step']:.4f} s")
        else:
            self.logger.info("No steps completed - simulation failed immediately")
        
        return self.simulation_state

    def _run_single_step_with_monitoring(self) -> bool:
        """
        Run single time step with detailed convergence monitoring.
        """
        try:
            # Store initial state for comparison
            initial_velocities = self.solver.v.to_numpy()
            # Note: pressure is managed by pcg_solver, not directly accessible
            
            # Run MPM step
            self.solver.step()
            
            # Check for numerical issues
            final_velocities = self.solver.v.to_numpy()
            # Note: pressure is managed by pcg_solver, not directly accessible
            
            # Velocity statistics
            vel_magnitude = np.linalg.norm(final_velocities, axis=1)
            max_vel = np.max(vel_magnitude)
            mean_vel = np.mean(vel_magnitude)
            
            # Pressure statistics (not directly accessible, skip for now)
            max_pressure = 0.0  # Placeholder
            min_pressure = 0.0  # Placeholder
            mean_pressure = 0.0  # Placeholder
            
            # Check for NaN or infinite values
            has_nan_vel = np.any(np.isnan(final_velocities))
            has_inf_vel = np.any(np.isinf(final_velocities))
            has_nan_pressure = False  # Placeholder - pressure not directly accessible
            has_inf_pressure = False  # Placeholder - pressure not directly accessible
            
            # Log detailed statistics
            if self.simulation_state['step'] % 50 == 0:
                self.logger.debug(f"Step {self.simulation_state['step']} statistics:")
                self.logger.debug(f"  Velocity: max={max_vel:.6f}, mean={mean_vel:.6f}")
                self.logger.debug(f"  Pressure: max={max_pressure:.6f}, min={min_pressure:.6f}, mean={mean_pressure:.6f}")
                self.logger.debug(f"  NaN/Inf check: vel_nan={has_nan_vel}, vel_inf={has_inf_vel}, p_nan={has_nan_pressure}, p_inf={has_inf_pressure}")
            
            # Store statistics
            self.simulation_state['velocity_stats'].append({
                'step': self.simulation_state['step'],
                'time': self.simulation_state['time'],
                'max_velocity': max_vel,
                'mean_velocity': mean_vel,
                'has_nan': has_nan_vel,
                'has_inf': has_inf_vel
            })
            
            self.simulation_state['particle_stats'].append({
                'step': self.simulation_state['step'],
                'time': self.simulation_state['time'],
                'max_pressure': max_pressure,
                'min_pressure': min_pressure,
                'mean_pressure': mean_pressure,
                'has_nan': has_nan_pressure,
                'has_inf': has_inf_pressure
            })
            
            # Check for numerical explosion
            if max_vel > 100.0:  # Unrealistic velocity
                self.logger.error(f"Velocity explosion detected: max_vel = {max_vel}")
                return False
            
            if has_nan_vel or has_inf_vel:
                self.logger.error(f"NaN/Inf in velocities at step {self.simulation_state['step']}")
                return False
            
            if has_nan_pressure or has_inf_pressure:
                self.logger.error(f"NaN/Inf in pressures at step {self.simulation_state['step']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in step {self.simulation_state['step']}: {str(e)}")
            return False

    def _check_convergence_issues(self):
        """Check for convergence issues and log warnings."""
        if len(self.simulation_state['velocity_stats']) < 10:
            return
        
        # Check for increasing velocity trend
        recent_velocities = [v['max_velocity'] for v in self.simulation_state['velocity_stats'][-10:]]
        if len(recent_velocities) >= 5:
            vel_trend = np.polyfit(range(len(recent_velocities)), recent_velocities, 1)[0]
            if vel_trend > 0.1:  # Increasing trend
                self.logger.warning(f"Increasing velocity trend detected: {vel_trend:.6f}")
        
        # Check for pressure oscillations
        recent_pressures = [p['max_pressure'] for p in self.simulation_state['particle_stats'][-10:]]
        if len(recent_pressures) >= 5:
            pressure_variance = np.var(recent_pressures)
            if pressure_variance > 1e6:  # High variance
                self.logger.warning(f"High pressure variance detected: {pressure_variance:.2e}")

    def _output_and_analyze(self, output_dir: str):
        """Output results and perform analysis."""
        # Export VTK
        vtk_filename = f"{output_dir}/particles_{self.simulation_state['step']:06d}.vtk"
        self.solver.export_vtk(vtk_filename)
        
        # Compute output metrics
        try:
            metrics = self._compute_output_metrics()
            self.logger.info(f"Step {self.simulation_state['step']} metrics: {metrics}")
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")

    def _compute_output_metrics(self) -> Dict[str, float]:
        """Compute output metrics with error handling."""
        try:
            # Simplified metrics calculation
            velocities = self.solver.v.to_numpy()
            positions = self.solver.x.to_numpy()
            
            # Flow statistics
            vel_magnitude = np.linalg.norm(velocities, axis=1)
            max_velocity = np.max(vel_magnitude)
            mean_velocity = np.mean(vel_magnitude)
            
            # Position statistics
            x_positions = positions[:, 0]
            max_x_position = np.max(x_positions)
            mean_x_position = np.mean(x_positions)
            
            metrics = {
                'max_velocity': max_velocity,
                'mean_velocity': mean_velocity,
                'max_x_position': max_x_position,
                'mean_x_position': mean_x_position,
                'total_particles': self.total_particles
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in metrics calculation: {str(e)}")
            return {}

    def save_convergence_analysis(self, output_dir: str):
        """Save detailed convergence analysis."""
        self.logger.info("Saving convergence analysis...")
        
        # Create analysis directory
        analysis_dir = f"{output_dir}/convergence_analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save velocity statistics
        if self.simulation_state['velocity_stats']:
            vel_data = np.array([(v['step'], v['time'], v['max_velocity'], v['mean_velocity']) 
                               for v in self.simulation_state['velocity_stats']])
            np.savetxt(f"{analysis_dir}/velocity_stats.txt", vel_data, 
                      header="step time max_velocity mean_velocity")
        
        # Save pressure statistics
        if self.simulation_state['particle_stats']:
            p_data = np.array([(p['step'], p['time'], p['max_pressure'], p['min_pressure'], p['mean_pressure']) 
                             for p in self.simulation_state['particle_stats']])
            np.savetxt(f"{analysis_dir}/pressure_stats.txt", p_data,
                      header="step time max_pressure min_pressure mean_pressure")
        
        # Create convergence plots
        self._create_convergence_plots(analysis_dir)
        
        self.logger.info(f"Convergence analysis saved to {analysis_dir}")

    def _create_convergence_plots(self, analysis_dir: str):
        """Create convergence diagnostic plots."""
        try:
            if not self.simulation_state['velocity_stats']:
                return
            
            # Extract data
            steps = [v['step'] for v in self.simulation_state['velocity_stats']]
            times = [v['time'] for v in self.simulation_state['velocity_stats']]
            max_vels = [v['max_velocity'] for v in self.simulation_state['velocity_stats']]
            mean_vels = [v['mean_velocity'] for v in self.simulation_state['velocity_stats']]
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Velocity evolution
            ax1.plot(times, max_vels, 'r-', label='Max Velocity')
            ax1.plot(times, mean_vels, 'b-', label='Mean Velocity')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Velocity (m/s)')
            ax1.set_title('Velocity Evolution')
            ax1.legend()
            ax1.grid(True)
            
            # Velocity vs step
            ax2.plot(steps, max_vels, 'r-', label='Max Velocity')
            ax2.plot(steps, mean_vels, 'b-', label='Mean Velocity')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity vs Step')
            ax2.legend()
            ax2.grid(True)
            
            # Pressure evolution
            if self.simulation_state['particle_stats']:
                max_pressures = [p['max_pressure'] for p in self.simulation_state['particle_stats']]
                mean_pressures = [p['mean_pressure'] for p in self.simulation_state['particle_stats']]
                
                ax3.plot(times[:len(max_pressures)], max_pressures, 'r-', label='Max Pressure')
                ax3.plot(times[:len(mean_pressures)], mean_pressures, 'b-', label='Mean Pressure')
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Pressure (Pa)')
                ax3.set_title('Pressure Evolution')
                ax3.legend()
                ax3.grid(True)
            
            # Log scale velocity
            ax4.semilogy(times, max_vels, 'r-', label='Max Velocity')
            ax4.semilogy(times, mean_vels, 'b-', label='Mean Velocity')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Velocity (m/s) - Log Scale')
            ax4.set_title('Velocity Evolution (Log Scale)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{analysis_dir}/convergence_diagnostics.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Convergence plots created")
            
        except Exception as e:
            self.logger.error(f"Error creating convergence plots: {str(e)}")


def main():
    """Main function to run detailed simulation with logging."""
    print("=== DETAILED MPM SIMULATION WITH LOGGING ===")
    print("This will run the simulation with comprehensive logging")
    print("to diagnose convergence and numerical stability issues.")
    print("="*60)
    
    # Initialize simulation
    sim = DetailedLoggingSimulation("physics_config.yaml")
    
    # Run simulation
    results = sim.run_simulation_with_detailed_logging("simulation_output")
    
    # Save convergence analysis
    sim.save_convergence_analysis("simulation_output")
    
    print("=== SIMULATION COMPLETED ===")
    print("Check the logs/ directory for detailed logs")
    print("Check simulation_output/convergence_analysis/ for diagnostic plots")


if __name__ == "__main__":
    main()
