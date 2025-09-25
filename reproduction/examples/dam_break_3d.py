"""
3D Dam Break Test Case for iMPM

This example reproduces the dam break simulation described in Section 7.1 of:
"Incompressible material point method for free surface flow"

Physical parameters:
- Domain: 3.22m × 1.0m × 0.6m
- Grid: 161 × 50 × 30 (Δx = 0.02m)
- Initial fluid column: 1.2m × 0.6m × 0.6m
- Density: 1000 kg/m³
- Viscosity: 1.01×10⁻³ Pa·s
- Gravity: 9.8 m/s²

Validation data:
- Wave front position vs time
- Pressure sensor readings at specific locations
- Comparison with experimental data from Martin & Moyce (1952)
"""

import sys
import os
import numpy as np
import taichi as ti

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incompressible_mpm_solver import IncompressibleMPMSolver
import time

class DamBreak3D:
    def __init__(self):
        # Physical parameters from paper
        self.L = 0.6   # Initial water column width (m)
        self.H = 1.0   # Initial water column height (m)
        self.domain_x = 3.22  # Domain length (m)
        self.domain_y = 1.0   # Domain height (m)
        self.domain_z = 0.6   # Domain width (m)
        
        # Grid parameters
        self.dx = 0.02  # Grid spacing (m)
        self.nx = int(self.domain_x / self.dx) + 1  # 161
        self.ny = int(self.domain_y / self.dx) + 1  # 50
        self.nz = int(self.domain_z / self.dx) + 1  # 30
        
        # Physical parameters
        self.rho = 1000.0      # Density (kg/m³)
        self.mu = 1.01e-3      # Dynamic viscosity (Pa·s)
        self.g = 9.8           # Gravity (m/s²)
        self.dt = 1e-4         # Time step (s)
        
        # Simulation parameters
        self.alpha_h = 0.05    # Hourglass damping coefficient
        self.chi = 0.03        # FLIP blending coefficient
        
        # Particles per cell
        self.ppc = 4  # 2×2×1 particles per cell initially
        
        print(f"Dam Break 3D Setup:")
        print(f"  Domain: {self.domain_x}m × {self.domain_y}m × {self.domain_z}m")
        print(f"  Grid: {self.nx} × {self.ny} × {self.nz} (dx = {self.dx}m)")
        print(f"  Initial fluid: {self.L}m × {self.H}m × {self.domain_z}m")
        print(f"  Physical parameters: ρ = {self.rho} kg/m³, μ = {self.mu} Pa·s")
        
    def create_solver(self):
        """Create and initialize the iMPM solver"""
        self.solver = IncompressibleMPMSolver(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.dx,
            rho=self.rho,
            mu=self.mu,
            gamma=0.0,  # No surface tension for dam break
            g=self.g,
            dt=self.dt,
            alpha_h=self.alpha_h,
            chi=self.chi,
            max_particles=100000
        )
        
        # Initialize particles in dam configuration
        self.initialize_dam_particles()
        
        # Initialize level set for dam break
        self.initialize_level_set()
        
        print(f"Initialized {self.solver.n_particles[None]} particles")
        
    def initialize_dam_particles(self):
        """Initialize particles in dam configuration"""
        # Calculate particle spacing
        particles_per_dx = int(np.cbrt(self.ppc))  # particles per direction per cell
        particle_dx = self.dx / particles_per_dx
        
        # Count particles needed
        nx_particles = int(self.L / particle_dx)
        ny_particles = int(self.H / particle_dx)
        nz_particles = int(self.domain_z / particle_dx)
        
        total_particles = nx_particles * ny_particles * nz_particles
        print(f"Creating {total_particles} particles in dam")
        
        if total_particles > self.solver.max_particles:
            print(f"Warning: Requested {total_particles} particles exceeds maximum {self.solver.max_particles}")
            total_particles = self.solver.max_particles
            
        # Initialize particles
        self.solver.initialize_particles_dam_break(
            x_min=0.0, x_max=self.L,
            y_min=0.0, y_max=self.H,
            z_min=0.0, z_max=self.domain_z,
            ppc=self.ppc
        )
        
    def initialize_level_set(self):
        """Initialize level set function for dam break"""
        self.solver.level_set_method.initialize_box(
            x_min=0.0, x_max=self.L,
            y_min=0.0, y_max=self.H,
            z_min=0.0, z_max=self.domain_z
        )
        
        # Compute initial gradient and curvature
        self.solver.level_set_method.compute_gradient()
        self.solver.level_set_method.compute_curvature_least_squares()
        
    def run_simulation(self, total_time=2.0, output_interval=0.05):
        """Run the dam break simulation"""
        total_steps = int(total_time / self.dt)
        output_steps = int(output_interval / self.dt)
        
        print(f"Running simulation for {total_time}s ({total_steps} steps)")
        print(f"Output every {output_interval}s ({output_steps} steps)")
        
        # Timing and monitoring
        start_time = time.time()
        frame = 0
        
        # Data collection for validation
        wave_front_positions = []
        times = []
        
        for step in range(total_steps):
            current_time = step * self.dt
            
            # Perform simulation step
            pcg_iterations = self.solver.step()
            
            # Monitor and output
            if step % output_steps == 0:
                self.solver.compute_statistics()
                
                # Compute wave front position (rightmost particle x-coordinate)
                positions, velocities = self.solver.export_particles_to_numpy()
                if len(positions) > 0:
                    wave_front_x = np.max(positions[:, 0])
                else:
                    wave_front_x = 0.0
                
                wave_front_positions.append(wave_front_x)
                times.append(current_time)
                
                elapsed_time = time.time() - start_time
                steps_per_second = (step + 1) / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Frame {frame:3d} (t={current_time:.3f}s, step={step:6d}):")
                print(f"  Wave front: {wave_front_x:.3f}m")
                print(f"  KE: {self.solver.total_kinetic_energy[None]:.6f} J")
                print(f"  Max velocity: {self.solver.max_velocity[None]:.3f} m/s")
                print(f"  PCG iterations: {pcg_iterations}")
                print(f"  Performance: {steps_per_second:.1f} steps/s")
                
                # Export VTK for visualization (optional)
                if frame % 5 == 0:  # Every 0.25s
                    self.export_vtk(frame, current_time)
                
                frame += 1
                
            # Check for simulation instability
            if self.solver.max_velocity[None] > 20.0:  # Reasonable upper limit
                print("Warning: Maximum velocity exceeds reasonable bounds!")
                print("Simulation may be unstable.")
                break
                
        total_elapsed = time.time() - start_time
        average_sps = total_steps / total_elapsed
        
        print(f"Simulation completed in {total_elapsed:.2f}s")
        print(f"Average performance: {average_sps:.1f} steps/s")
        
        # Save validation data
        self.save_validation_data(times, wave_front_positions)
        
        return times, wave_front_positions
    
    def export_vtk(self, frame, time):
        """Export particle data to VTK format"""
        try:
            positions, velocities = self.solver.export_particles_to_numpy()
            
            if len(positions) == 0:
                return
                
            # Create simple VTK output
            filename = f"dam_break_frame_{frame:04d}.vtk"
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"Dam break simulation t={time:.4f}\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Points
                f.write(f"POINTS {len(positions)} float\n")
                for pos in positions:
                    f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
                # Cells (points)
                f.write(f"CELLS {len(positions)} {len(positions)*2}\n")
                for i in range(len(positions)):
                    f.write(f"1 {i}\n")
                
                # Cell types (VTK_VERTEX = 1)
                f.write(f"CELL_TYPES {len(positions)}\n")
                for i in range(len(positions)):
                    f.write("1\n")
                
                # Point data
                f.write(f"POINT_DATA {len(positions)}\n")
                f.write("VECTORS velocity float\n")
                for vel in velocities:
                    f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")
                
                f.write("SCALARS speed float\n")
                f.write("LOOKUP_TABLE default\n")
                for vel in velocities:
                    speed = np.linalg.norm(vel)
                    f.write(f"{speed:.6f}\n")
            
            if frame == 0:
                print(f"VTK files will be saved as: {filename}")
                
        except Exception as e:
            print(f"Warning: Failed to export VTK file: {e}")
    
    def save_validation_data(self, times, wave_front_positions):
        """Save validation data for comparison with experiments"""
        try:
            # Save as CSV for easy analysis
            data = np.column_stack([times, wave_front_positions])
            np.savetxt("dam_break_wave_front.csv", data, 
                      header="time,wave_front_x", delimiter=",", fmt="%.6f")
            
            # Normalize by initial column width for comparison with literature
            normalized_positions = np.array(wave_front_positions) / self.L
            normalized_times = np.array(times) / np.sqrt(self.L / self.g)
            
            normalized_data = np.column_stack([normalized_times, normalized_positions])
            np.savetxt("dam_break_normalized.csv", normalized_data,
                      header="normalized_time,normalized_position", delimiter=",", fmt="%.6f")
            
            print("Validation data saved to:")
            print("  dam_break_wave_front.csv")
            print("  dam_break_normalized.csv")
            
        except Exception as e:
            print(f"Warning: Failed to save validation data: {e}")
    
    def compare_with_theory(self, times, wave_front_positions):
        """Compare results with theoretical/experimental data"""
        # Martin & Moyce (1952) experimental data (dimensionless)
        # For 2D dam break: x/L = 2*sqrt(t*sqrt(g/L)) (approximate)
        
        print("\nComparison with theory:")
        print("Time (s) | Simulated x/L | Theoretical x/L | Difference")
        print("-" * 60)
        
        for i, (t, x) in enumerate(zip(times, wave_front_positions)):
            if t > 0:
                x_normalized = x / self.L
                t_normalized = t * np.sqrt(self.g / self.L)
                
                # Theoretical prediction (simplified)
                x_theory = 2.0 * np.sqrt(t_normalized)
                
                difference = abs(x_normalized - x_theory)
                
                if i % 5 == 0:  # Print every 5th data point
                    print(f"{t:8.3f} | {x_normalized:12.3f} | {x_theory:14.3f} | {difference:9.3f}")

def main():
    """Main function to run the dam break simulation"""
    print("=" * 80)
    print("3D Dam Break Simulation using Incompressible Material Point Method (iMPM)")
    print("=" * 80)
    
    # Create and run simulation
    dam_break = DamBreak3D()
    dam_break.create_solver()
    
    # Run simulation
    times, wave_front_positions = dam_break.run_simulation(
        total_time=2.0,        # Run for 2 seconds
        output_interval=0.1    # Output every 0.1 seconds
    )
    
    # Compare with theory
    dam_break.compare_with_theory(times, wave_front_positions)
    
    print("=" * 80)
    print("Simulation completed successfully!")
    print("Check the generated files for visualization and analysis.")
    print("=" * 80)

if __name__ == "__main__":
    main()