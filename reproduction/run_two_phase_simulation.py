"""
Two-Phase Debris Flow Simulation Runner

This script runs a dam break simulation with two-phase MPM,
showing both solid particles (debris) and fluid particles (water).
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import yaml
from datetime import datetime

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

from two_phase_mpm_solver import TwoPhaseMPMSolver


def load_config(config_path='physics_config_paper_accurate.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_simulation():
    """Run two-phase debris flow simulation"""
    print("=" * 60)
    print("Two-Phase Debris Flow Simulation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Simulation parameters
    domain_length = config['simulation']['domain_length']
    domain_height = config['simulation']['domain_height']
    domain_width = config['simulation']['domain_width']
    dx = config['numerics']['dx']
    dt = config['numerics']['max_timestep']
    total_time = min(config['simulation']['total_time'], 0.5)  # Limit for testing
    
    # Grid dimensions
    nx = int(domain_length / dx) + 4
    ny = int(domain_height / dx) + 4
    nz = int(domain_width / dx) + 4
    
    # Initial debris dimensions
    debris_length = config['simulation']['initial_debris_length']
    debris_height = config['simulation']['initial_debris_height']
    
    # Calculate max particles
    ppc = 4  # particles per cell
    particle_dx = dx / np.sqrt(ppc)
    num_particles = int(debris_length / particle_dx * debris_height / particle_dx * domain_width / particle_dx * 1.2)
    max_particles = max(50000, num_particles)
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"dx = {dx:.4f} m, dt = {dt:.2e} s")
    print(f"Debris: {debris_length:.2f} x {debris_height:.2f} x {domain_width:.2f} m")
    print(f"Max particles per phase: {max_particles}")
    
    # Create solver
    solver = TwoPhaseMPMSolver(
        nx=nx, ny=ny, nz=nz,
        dx=dx,
        rho_s=float(config['solid_phase']['density']),
        E_s=float(config['solid_phase']['young_modulus']),
        nu_s=float(config['solid_phase']['poisson_ratio']),
        friction_angle=np.arctan(float(config['solid_phase']['static_friction'])) * 180 / np.pi,
        rho_f=float(config['fluid_phase']['density']),
        mu_f=float(config['fluid_phase']['viscosity']),
        d_s=float(config['solid_phase']['particle_diameter']),
        phi_s0=float(config['simulation'].get('solid_volume_fraction', 0.55)),
        g=float(config['simulation']['gravity']),
        dt=dt,
        max_particles=max_particles,
        flip_ratio=1.0 - float(config['numerics']['pic_flip_ratio'])
    )
    
    # Initialize particles
    print("\nInitializing particles...")
    solver.init_particles(
        x_min=dx * 2,
        x_max=dx * 2 + debris_length,
        y_min=dx * 2,
        y_max=dx * 2 + debris_height,
        z_min=dx * 2,
        z_max=dx * 2 + domain_width,
        ppc=ppc
    )
    
    n_solid = solver.n_solid[None]
    n_fluid = solver.n_fluid[None]
    print(f"Initialized {n_solid} solid particles and {n_fluid} fluid particles")
    
    # Setup output directory
    output_dir = 'simulation_output/two_phase_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulation loop
    n_steps = int(total_time / dt)
    save_interval = max(1, n_steps // 20)  # Save 20 frames
    
    history = {
        'time': [],
        'solid_positions': [],
        'fluid_positions': [],
        'solid_velocities': [],
        'fluid_velocities': [],
        'wave_front': []
    }
    
    print(f"\nRunning simulation for {n_steps} steps ({total_time:.3f} s)...")
    print("-" * 60)
    
    for step in range(n_steps):
        # Advance simulation
        solver.step()
        
        # Save data periodically
        if step % save_interval == 0 or step == n_steps - 1:
            t = step * dt
            
            # Get particle data
            data = solver.export_particles()
            
            # Compute statistics
            solid_pos = data['solid']['positions']
            fluid_pos = data['fluid']['positions']
            solid_vel = data['solid']['velocities']
            fluid_vel = data['fluid']['velocities']
            
            if len(solid_pos) > 0:
                wave_front = np.max(solid_pos[:, 0])
                max_vel_s = np.max(np.linalg.norm(solid_vel, axis=1))
            else:
                wave_front = 0.0
                max_vel_s = 0.0
            
            if len(fluid_pos) > 0:
                max_vel_f = np.max(np.linalg.norm(fluid_vel, axis=1))
            else:
                max_vel_f = 0.0
            
            print(f"Step {step:5d} | t = {t:.4f}s | Wave front: {wave_front:.3f}m | "
                  f"Max vel: solid={max_vel_s:.2f}, fluid={max_vel_f:.2f} m/s")
            
            # Store history
            history['time'].append(t)
            history['solid_positions'].append(solid_pos.copy())
            history['fluid_positions'].append(fluid_pos.copy())
            history['solid_velocities'].append(solid_vel.copy())
            history['fluid_velocities'].append(fluid_vel.copy())
            history['wave_front'].append(wave_front)
    
    print("-" * 60)
    print("Simulation completed!")
    
    # Generate visualization
    print("\nGenerating plots...")
    generate_two_phase_plots(history, output_dir, domain_length, domain_height)
    
    return history


def generate_two_phase_plots(history, output_dir, domain_length, domain_height):
    """Generate visualization plots for two-phase simulation"""
    
    # Set style similar to reference plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    times = np.array(history['time'])
    n_frames = len(times)
    
    # ========== 1. Flow Morphology - Solid vs Fluid ==========
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Two-Phase Debris Flow: Solid (Brown) vs Fluid (Blue)', fontsize=14, fontweight='bold')
    
    frame_indices = np.linspace(0, n_frames - 1, 6).astype(int)
    
    for idx, (ax, frame_idx) in enumerate(zip(axes.flatten(), frame_indices)):
        t = times[frame_idx]
        solid_pos = history['solid_positions'][frame_idx]
        fluid_pos = history['fluid_positions'][frame_idx]
        
        # Plot fluid first (blue, transparent)
        if len(fluid_pos) > 0:
            ax.scatter(fluid_pos[:, 0], fluid_pos[:, 1], 
                      c='steelblue', s=1, alpha=0.3, label='Fluid' if idx == 0 else '')
        
        # Plot solid on top (brown)
        if len(solid_pos) > 0:
            ax.scatter(solid_pos[:, 0], solid_pos[:, 1], 
                      c='saddlebrown', s=2, alpha=0.7, label='Solid' if idx == 0 else '')
        
        ax.set_xlim(0, domain_length)
        ax.set_ylim(0, domain_height)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.3f} s', fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        if idx == 0:
            ax.legend(loc='upper right', markerscale=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_phase_morphology.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'two_phase_morphology.pdf'), bbox_inches='tight')
    plt.close()
    
    # ========== 2. Velocity Field Comparison ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Select a mid-time frame
    mid_frame = n_frames // 2
    t = times[mid_frame]
    solid_pos = history['solid_positions'][mid_frame]
    solid_vel = history['solid_velocities'][mid_frame]
    fluid_pos = history['fluid_positions'][mid_frame]
    fluid_vel = history['fluid_velocities'][mid_frame]
    
    # Solid velocity magnitude
    ax = axes[0, 0]
    if len(solid_pos) > 0:
        vel_mag_s = np.linalg.norm(solid_vel, axis=1)
        sc = ax.scatter(solid_pos[:, 0], solid_pos[:, 1], c=vel_mag_s, 
                       cmap='YlOrRd', s=3, vmin=0, vmax=max(2, np.percentile(vel_mag_s, 95)))
        plt.colorbar(sc, ax=ax, label='Velocity (m/s)')
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, domain_height)
    ax.set_aspect('equal')
    ax.set_title(f'Solid Phase Velocity | t = {t:.3f}s', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # Fluid velocity magnitude
    ax = axes[0, 1]
    if len(fluid_pos) > 0:
        vel_mag_f = np.linalg.norm(fluid_vel, axis=1)
        sc = ax.scatter(fluid_pos[:, 0], fluid_pos[:, 1], c=vel_mag_f, 
                       cmap='Blues', s=3, vmin=0, vmax=max(2, np.percentile(vel_mag_f, 95)))
        plt.colorbar(sc, ax=ax, label='Velocity (m/s)')
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, domain_height)
    ax.set_aspect('equal')
    ax.set_title(f'Fluid Phase Velocity | t = {t:.3f}s', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # Relative velocity (solid - fluid)
    ax = axes[1, 0]
    if len(solid_pos) > 0 and len(fluid_pos) > 0:
        # For co-located particles, compute relative velocity
        rel_vel = solid_vel - fluid_vel[:len(solid_vel)]  # Assuming same order
        rel_vel_mag = np.linalg.norm(rel_vel, axis=1)
        sc = ax.scatter(solid_pos[:, 0], solid_pos[:, 1], c=rel_vel_mag, 
                       cmap='RdPu', s=3, vmin=0, vmax=max(0.5, np.percentile(rel_vel_mag, 95)))
        plt.colorbar(sc, ax=ax, label='|v_s - v_f| (m/s)')
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, domain_height)
    ax.set_aspect('equal')
    ax.set_title('Relative Velocity (Solid - Fluid)', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # Velocity vectors (subsampled)
    ax = axes[1, 1]
    subsample = 20
    if len(solid_pos) > 0:
        ax.quiver(solid_pos[::subsample, 0], solid_pos[::subsample, 1],
                 solid_vel[::subsample, 0], solid_vel[::subsample, 1],
                 color='brown', alpha=0.7, scale=30, label='Solid')
    if len(fluid_pos) > 0:
        ax.quiver(fluid_pos[::subsample, 0], fluid_pos[::subsample, 1],
                 fluid_vel[::subsample, 0], fluid_vel[::subsample, 1],
                 color='steelblue', alpha=0.5, scale=30, label='Fluid')
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, domain_height)
    ax.set_aspect('equal')
    ax.set_title('Velocity Vectors', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_phase_velocity.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'two_phase_velocity.pdf'), bbox_inches='tight')
    plt.close()
    
    # ========== 3. Wave Front Evolution ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    
    wave_fronts = np.array(history['wave_front'])
    ax.plot(times, wave_fronts, 'k-', linewidth=2, marker='o', markersize=4, label='Simulation')
    
    # Theoretical free fall (rough estimate)
    g = 9.81
    t_theory = np.linspace(0, max(times), 100)
    # Simplified theoretical model
    ax.plot(t_theory, 0.04 + 0.5 * g * np.sin(20 * np.pi / 180) * t_theory**2, 
           'r--', linewidth=1.5, label='Theoretical (simplified)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Wave Front Position (m)', fontsize=12)
    ax.set_title('Debris Flow Front Propagation', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wave_front_evolution.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'wave_front_evolution.pdf'), bbox_inches='tight')
    plt.close()
    
    # ========== 4. Combined 3D View ==========
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use final frame
    solid_pos = history['solid_positions'][-1]
    fluid_pos = history['fluid_positions'][-1]
    
    if len(solid_pos) > 0:
        ax.scatter(solid_pos[::5, 0], solid_pos[::5, 2], solid_pos[::5, 1],
                  c='saddlebrown', s=3, alpha=0.6, label='Solid')
    if len(fluid_pos) > 0:
        ax.scatter(fluid_pos[::5, 0], fluid_pos[::5, 2], fluid_pos[::5, 1],
                  c='steelblue', s=2, alpha=0.3, label='Fluid')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title(f'3D View of Two-Phase Flow | t = {times[-1]:.3f}s', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_phase_3d_view.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    run_simulation()

