# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Saturated Granular Column Collapse Simulation

Reference: Ceccato, F., Leonardi, A., Girardi, V., Simonini, P., & Pirulli, M. (2020).
"Numerical and experimental investigation of saturated granular column collapse in air"
Soils and Foundations. doi:10.1016/j.sandf.2020.04.004

Experimental setup:
- Initial column width L₀ = 0.07 m (scaled from 0.7m in paper)
- Initial column height H₀ = 0.12 m  
- Container depth w = 0.05 m
- Solid volume fraction φ_s = 0.4 (porosity n = 0.6)

This script runs the two-phase MPM simulation and outputs results for visualization.
"""

import taichi as ti
import numpy as np
import os
import time
import sys

# Initialize Taichi with GPU if available
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4)

# Import the collapse solver
from collapse import (
    parametersetting, initialize, load_step, fetchinfo, fetchinfo_l,
    x_s, v_s, x_l, v_l, n_s_particles, n_l_particles, dt, timelimit,
    grid_sv, grid_lv, grid_phi_s, n_s, pore_3D, sig_principle_s
)


def setup_simulation():
    """Configure simulation parameters matching Ceccato et al. (2020) experiment"""
    
    # Scaled parameters for numerical simulation
    # Original: L₀=0.7m, H₀=0.12m, w=0.05m
    # Scaled: L₀=0.4m, H₀=0.7m (aspect ratio ~5.8)
    
    parametersetting(
        friction=0.5,              # Bottom boundary friction
        friction_side=0.3,         # Side wall friction  
        FLIPcoeff=0.95,            # FLIP blending coefficient (mostly FLIP for stability)
        friction_angle=30.0,       # Internal friction angle (degrees)
        mu_2=0.6,                  # Dynamic friction coefficient μ₂
        xifactor=0.1,              # Rheology parameter
        sand_E=1.0e6,              # Young's modulus (Pa)
        sand_nu=0.3,               # Poisson's ratio
        board_inclination=0,       # Horizontal base (no slope)
        timestep=1e-5,             # Time step (s)
        DEM_contact=False,         # No DEM obstacles
    )


def export_vtk(positions_s, velocities_s, positions_l, velocities_l, 
               porosity, pore_pressure, frame_id, output_dir):
    """Export particle data to VTK format for visualization"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export solid phase
    n_solid = len(positions_s)
    if n_solid > 0:
        filename_s = os.path.join(output_dir, f"solid_{frame_id:06d}.vtk")
        with open(filename_s, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Solid Phase Particles\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {n_solid} float\n")
            for pos in positions_s:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            f.write(f"\nPOINT_DATA {n_solid}\n")
            f.write("VECTORS velocity float\n")
            for vel in velocities_s:
                f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")
            
            f.write("SCALARS porosity float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for phi in porosity:
                f.write(f"{phi:.6f}\n")
    
    # Export liquid phase
    n_liquid = len(positions_l)
    if n_liquid > 0:
        filename_l = os.path.join(output_dir, f"liquid_{frame_id:06d}.vtk")
        with open(filename_l, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Liquid Phase Particles\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {n_liquid} float\n")
            for pos in positions_l:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            f.write(f"\nPOINT_DATA {n_liquid}\n")
            f.write("VECTORS velocity float\n")
            for vel in velocities_l:
                f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")
            
            f.write("SCALARS pore_pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for pp in pore_pressure:
                f.write(f"{pp:.6f}\n")
    
    print(f"  Exported frame {frame_id}: {n_solid} solid, {n_liquid} liquid particles")


def compute_front_position(positions, threshold_height=0.01):
    """Compute wave front position (maximum x where particles exist)"""
    if len(positions) == 0:
        return 0.0
    
    # Filter particles above threshold height
    valid_mask = positions[:, 1] > threshold_height
    if np.sum(valid_mask) == 0:
        return 0.0
    
    return np.max(positions[valid_mask, 0])


def run_simulation():
    """Main simulation loop"""
    
    print("=" * 60)
    print("Saturated Granular Column Collapse Simulation")
    print("Reference: Ceccato et al. (2020), Soils and Foundations")
    print("=" * 60)
    
    # Setup parameters
    print("\n[1/4] Setting up simulation parameters...")
    setup_simulation()
    
    # Initialize particles
    print("\n[2/4] Initializing particles...")
    initialize()
    
    n_solid = n_s_particles[None]
    n_liquid = n_l_particles[None]
    print(f"  Solid particles: {n_solid}")
    print(f"  Liquid particles: {n_liquid}")
    
    # Output directory
    output_dir = "../../two_phase_collapse_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulation parameters
    dt_val = dt
    total_time = 0.5  # Simulation duration (s)
    output_interval = 100  # Steps between VTK outputs
    log_interval = 50  # Steps between log outputs
    
    # Reference time scale: t_ref = sqrt(H₀/g) ≈ 0.11 s for H₀=0.12m
    H0 = 0.7  # Initial column height in simulation
    g = 9.81
    t_ref = np.sqrt(H0 / g)
    print(f"\n  Reference time t_ref = sqrt(H₀/g) = {t_ref:.4f} s")
    
    # Data storage for analysis
    time_history = []
    front_position_history = []
    max_velocity_history = []
    
    # Initial export
    positions_s = x_s.to_numpy()[:n_solid]
    velocities_s = v_s.to_numpy()[:n_solid]
    positions_l = x_l.to_numpy()[:n_liquid]
    velocities_l = v_l.to_numpy()[:n_liquid]
    porosity = n_s.to_numpy()[:n_solid]
    pore_press = np.zeros(n_liquid)  # Initial pore pressure = 0
    
    # Get initial front position
    initial_front = compute_front_position(positions_s)
    print(f"  Initial front position: {initial_front:.4f} m")
    
    export_vtk(positions_s, velocities_s, positions_l, velocities_l,
               porosity, pore_press, 0, output_dir)
    
    # Run simulation
    print("\n[3/4] Running simulation...")
    print("-" * 60)
    
    step = 0
    current_time = 0.0
    frame_id = 1
    start_wall_time = time.time()
    
    while current_time < total_time:
        # Perform one time step
        load_step(step)
        
        current_time += dt_val
        step += 1
        
        # Log progress
        if step % log_interval == 0:
            # Get current state
            positions_s = x_s.to_numpy()[:n_solid]
            velocities_s = v_s.to_numpy()[:n_solid]
            
            # Compute metrics
            front_pos = compute_front_position(positions_s)
            max_vel = np.max(np.linalg.norm(velocities_s, axis=1)) if n_solid > 0 else 0
            t_normalized = current_time / t_ref
            
            time_history.append(current_time)
            front_position_history.append(front_pos)
            max_velocity_history.append(max_vel)
            
            elapsed = time.time() - start_wall_time
            print(f"  Step {step:6d} | t={current_time:.5f}s (t/t_ref={t_normalized:.2f}) | "
                  f"Front: {front_pos:.4f}m | MaxVel: {max_vel:.3f}m/s | "
                  f"Elapsed: {elapsed:.1f}s")
        
        # Export VTK
        if step % output_interval == 0:
            positions_s = x_s.to_numpy()[:n_solid]
            velocities_s = v_s.to_numpy()[:n_solid]
            positions_l = x_l.to_numpy()[:n_liquid]
            velocities_l = v_l.to_numpy()[:n_liquid]
            porosity = n_s.to_numpy()[:n_solid]
            
            # Get pore pressure (simplified)
            pore_press = np.zeros(n_liquid)
            
            export_vtk(positions_s, velocities_s, positions_l, velocities_l,
                       porosity, pore_press, frame_id, output_dir)
            frame_id += 1
    
    # Final statistics
    print("-" * 60)
    total_wall_time = time.time() - start_wall_time
    print(f"\n[4/4] Simulation completed!")
    print(f"  Total steps: {step}")
    print(f"  Total simulation time: {current_time:.4f} s")
    print(f"  Wall clock time: {total_wall_time:.1f} s")
    print(f"  Output frames: {frame_id}")
    
    # Save time history data
    history_file = os.path.join(output_dir, "time_history.csv")
    np.savetxt(history_file, 
               np.column_stack([time_history, front_position_history, max_velocity_history]),
               header="time,front_position,max_velocity",
               delimiter=",", comments="")
    print(f"  Time history saved to: {history_file}")
    
    # Generate comparison plot
    generate_comparison_plot(time_history, front_position_history, t_ref, 
                            initial_front, output_dir)
    
    return output_dir


def generate_comparison_plot(time_history, front_history, t_ref, L0, output_dir):
    """Generate comparison plot with experimental data reference"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Front position vs time
        ax1 = axes[0]
        t_norm = np.array(time_history) / t_ref
        front_norm = np.array(front_history) / L0
        
        ax1.plot(t_norm, front_norm, 'b-', linewidth=2, label='Simulation')
        ax1.set_xlabel('$t/t_{ref}$', fontsize=12)
        ax1.set_ylabel('$x_{front}/L_0$', fontsize=12)
        ax1.set_title('Wave Front Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, 6])
        
        # Add reference points from Ceccato et al. (approximate)
        ref_t = [0.44, 1.10, 2.20, 5.0]
        ref_x = [1.0, 1.5, 2.0, 2.5]  # Approximate from figure
        ax1.scatter(ref_t, ref_x, c='red', s=100, marker='s', 
                   label='Ceccato et al. (2020)', zorder=5)
        ax1.legend()
        
        # Plot 2: Snapshot illustration
        ax2 = axes[1]
        ax2.set_aspect('equal')
        
        # Draw schematic of collapse stages
        ax2.text(0.5, 0.95, 'Collapse Stages (Schematic)', 
                transform=ax2.transAxes, ha='center', fontsize=14)
        
        # Initial column
        rect1 = patches.Rectangle((0.1, 0), 0.15, 0.5, 
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.7)
        ax2.add_patch(rect1)
        ax2.text(0.175, 0.55, '$t=0$', ha='center', fontsize=10)
        
        # Intermediate stage
        from matplotlib.patches import Polygon
        verts2 = [(0.35, 0), (0.35, 0.35), (0.55, 0.1), (0.55, 0)]
        poly2 = Polygon(verts2, linewidth=2, edgecolor='green',
                       facecolor='lightgreen', alpha=0.7)
        ax2.add_patch(poly2)
        ax2.text(0.45, 0.4, '$t/t_{ref}≈1$', ha='center', fontsize=10)
        
        # Final stage  
        verts3 = [(0.65, 0), (0.65, 0.15), (0.95, 0.02), (0.95, 0)]
        poly3 = Polygon(verts3, linewidth=2, edgecolor='red',
                       facecolor='lightsalmon', alpha=0.7)
        ax2.add_patch(poly3)
        ax2.text(0.8, 0.2, '$t/t_{ref}≈5$', ha='center', fontsize=10)
        
        ax2.set_xlim([0, 1.1])
        ax2.set_ylim([0, 0.7])
        ax2.set_xlabel('$x/L_0$', fontsize=12)
        ax2.set_ylabel('$y/H_0$', fontsize=12)
        ax2.axhline(y=0, color='brown', linewidth=3)  # Ground
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, "collapse_comparison.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Comparison plot saved to: {plot_file}")
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    output_dir = run_simulation()
    print(f"\n Results saved to: {output_dir}")
    print("=" * 60)

