"""
Saturated Granular Column Collapse Simulation
Reproducing: Ceccato et al. (2020), Soils and Foundations

Experimental setup from Figure:
- Initial column length L₀ = 0.07 m (scaled from 0.7m)
- Initial column height H₀ = 0.12 m
- Container width w = 0.05 m
- Solid volume fraction φ_s ≈ 0.4 (porosity n = 0.6)

Reference time scale: t_ref = √(H₀/g) ≈ 0.11 s

Time instants from experimental images:
- t/t_ref = 0.44, 1.10, 2.20, 5.0
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import os
from datetime import datetime

# Initialize Taichi - use CPU with f64 for accuracy (Metal GPU doesn't support f64)
ti.init(arch=ti.cpu, default_fp=ti.f64)

from two_phase_mpm_solver import TwoPhaseMPMSolver


def run_ceccato_simulation():
    """Run simulation matching Ceccato et al. (2020) experiment"""
    
    print("=" * 70)
    print("Saturated Granular Column Collapse - Ceccato et al. (2020)")
    print("=" * 70)
    
    # =====================================================
    # Physical parameters from Ceccato et al. (2020)
    # =====================================================
    
    # Geometry (scaled for computation)
    L0 = 0.07            # Initial column length (m) - scaled from 0.7m
    H0 = 0.12            # Initial column height (m)
    W0 = 0.05            # Container width (m)
    aspect_ratio = L0 / H0  # ~0.58
    
    # Material properties
    rho_s = 2650.0       # Solid density (kg/m³) - typical sand
    rho_f = 1000.0       # Fluid density (kg/m³) - water
    phi_s0 = 0.40        # Initial solid volume fraction
    d_s = 0.001          # Particle diameter (m) - 1mm sand
    
    # Mechanical properties
    E_s = 1.0e6          # Young's modulus (Pa)
    nu_s = 0.3           # Poisson's ratio
    friction_angle = 30.0  # Friction angle (degrees)
    mu_f = 0.001         # Water viscosity (Pa·s)
    
    # Gravity
    g = 9.81             # Gravitational acceleration (m/s²)
    
    # Reference time scale
    t_ref = np.sqrt(H0 / g)  # ≈ 0.11 s
    
    # Domain size
    domain_length = 0.5   # Total domain length (m)
    domain_height = 0.25  # Total domain height (m)
    domain_width = W0     # Domain width (m)
    
    # Numerical parameters
    dx = 0.008           # Grid spacing (m) - coarser for faster testing
    dt = 5e-5            # Time step (s)
    total_time = 0.15    # Total simulation time (s) ≈ 1.4 * t_ref (quick test)
    
    # Grid dimensions
    nx = int(domain_length / dx) + 4
    ny = int(domain_height / dx) + 4
    nz = int(domain_width / dx) + 4
    
    # Particles
    ppc = 4  # Particles per cell
    particle_dx = dx / np.sqrt(ppc)
    max_particles = int(L0 / particle_dx * H0 / particle_dx * W0 / particle_dx * 1.5)
    max_particles = max(50000, max_particles)
    
    print(f"\n[Experimental Parameters]", flush=True)
    print(f"  Initial column: L₀={L0:.3f}m × H₀={H0:.3f}m × W={W0:.3f}m", flush=True)
    print(f"  Aspect ratio: a = L₀/H₀ = {aspect_ratio:.2f}", flush=True)
    print(f"  Reference time: t_ref = √(H₀/g) = {t_ref:.4f} s", flush=True)
    print(f"  Solid volume fraction: φ_s = {phi_s0:.2f}", flush=True)
    
    print(f"\n[Numerical Parameters]", flush=True)
    print(f"  Grid: {nx} × {ny} × {nz}", flush=True)
    print(f"  dx = {dx:.4f} m, dt = {dt:.2e} s", flush=True)
    print(f"  Max particles per phase: {max_particles}", flush=True)
    print(f"  Total simulation time: {total_time:.3f} s ({total_time/t_ref:.1f} × t_ref)", flush=True)
    
    # =====================================================
    # Create solver
    # =====================================================
    print(f"\n[Creating solver...]")
    
    solver = TwoPhaseMPMSolver(
        nx=nx, ny=ny, nz=nz,
        dx=dx,
        rho_s=rho_s,
        E_s=E_s,
        nu_s=nu_s,
        friction_angle=friction_angle,
        rho_f=rho_f,
        mu_f=mu_f,
        d_s=d_s,
        phi_s0=phi_s0,
        g=g,
        dt=dt,
        max_particles=max_particles,
        flip_ratio=0.95
    )
    
    # =====================================================
    # Initialize particles
    # =====================================================
    print(f"\n[Initializing particles...]")
    
    # Initial column position (near left wall)
    x_min = dx * 2
    x_max = x_min + L0
    y_min = dx * 2
    y_max = y_min + H0
    z_min = dx * 2
    z_max = z_min + W0
    
    solver.init_particles(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        ppc=ppc
    )
    
    n_solid = solver.n_solid[None]
    n_fluid = solver.n_fluid[None]
    print(f"  Solid particles: {n_solid}")
    print(f"  Fluid particles: {n_fluid}")
    
    # =====================================================
    # Setup output
    # =====================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'simulation_output/ceccato_collapse_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================
    # Run simulation
    # =====================================================
    n_steps = int(total_time / dt)
    
    # Save at specific t/t_ref values from the paper
    target_times = np.array([0.0, 0.44, 1.10, 2.20, 5.0]) * t_ref
    
    # Additional regular saves - more frequent for testing
    save_interval = max(1, n_steps // 30)
    
    history = {
        'time': [],
        't_normalized': [],
        'solid_positions': [],
        'fluid_positions': [],
        'solid_velocities': [],
        'fluid_velocities': [],
        'wave_front': [],
        'runout': [],
        'max_height': []
    }
    
    print(f"\n[Running simulation...]")
    print(f"  Total steps: {n_steps}")
    print("-" * 70)
    
    saved_frames = []
    frame_id = 0
    
    for step in range(n_steps + 1):
        t = step * dt
        t_norm = t / t_ref
        
        # Check if we should save (at target times or regular interval)
        should_save = False
        if step % save_interval == 0:
            should_save = True
        for target_t in target_times:
            if abs(t - target_t) < dt:
                should_save = True
                break
        
        if should_save or step == 0:
            # Get particle data
            data = solver.export_particles()
            solid_pos = data['solid']['positions']
            fluid_pos = data['fluid']['positions']
            solid_vel = data['solid']['velocities']
            fluid_vel = data['fluid']['velocities']
            
            # Compute metrics
            if len(solid_pos) > 0:
                wave_front = np.max(solid_pos[:, 0])
                runout = wave_front - x_min
                max_height = np.max(solid_pos[:, 1])
                max_vel_s = np.max(np.linalg.norm(solid_vel, axis=1))
            else:
                wave_front = x_min
                runout = 0
                max_height = 0
                max_vel_s = 0
            
            # Store
            history['time'].append(t)
            history['t_normalized'].append(t_norm)
            history['solid_positions'].append(solid_pos.copy())
            history['fluid_positions'].append(fluid_pos.copy())
            history['solid_velocities'].append(solid_vel.copy())
            history['fluid_velocities'].append(fluid_vel.copy())
            history['wave_front'].append(wave_front)
            history['runout'].append(runout)
            history['max_height'].append(max_height)
            
            saved_frames.append(t_norm)
            
            print(f"  Step {step:6d} | t/t_ref = {t_norm:5.2f} | "
                  f"Front: {wave_front:.4f}m | Runout: {runout/L0:.2f}×L₀ | "
                  f"Height: {max_height/H0:.2f}×H₀ | MaxVel: {max_vel_s:.2f}m/s")
            
            frame_id += 1
        
        # Advance simulation
        if step < n_steps:
            solver.step()
    
    print("-" * 70)
    print(f"\n[Simulation completed!]")
    print(f"  Saved {len(history['time'])} frames")
    
    # =====================================================
    # Generate comparison figures
    # =====================================================
    print(f"\n[Generating comparison figures...]")
    
    generate_ceccato_comparison(history, output_dir, L0, H0, t_ref, 
                               domain_length, domain_height)
    
    # Save data
    np.savez(os.path.join(output_dir, 'simulation_data.npz'),
             time=history['time'],
             t_normalized=history['t_normalized'],
             wave_front=history['wave_front'],
             runout=history['runout'],
             max_height=history['max_height'])
    
    print(f"\n[Results saved to: {output_dir}]")
    print("=" * 70)
    
    return history, output_dir


def generate_ceccato_comparison(history, output_dir, L0, H0, t_ref, 
                                domain_length, domain_height):
    """Generate comparison figures matching Ceccato et al. (2020)"""
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })
    
    times = np.array(history['time'])
    t_norm = np.array(history['t_normalized'])
    n_frames = len(times)
    
    # ========== Figure 1: Morphology at key time instants ==========
    # Match the experimental images: t/t_ref = 0.44, 1.10, 2.20, 5.0
    target_t_norm = [0.44, 1.10, 2.20, 5.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Saturated Granular Column Collapse\n'
                 'Simulation Results (cf. Ceccato et al., 2020)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (ax, target_t) in enumerate(zip(axes.flatten(), target_t_norm)):
        # Find closest frame
        frame_idx = np.argmin(np.abs(t_norm - target_t))
        actual_t = t_norm[frame_idx]
        
        solid_pos = history['solid_positions'][frame_idx]
        fluid_pos = history['fluid_positions'][frame_idx]
        
        # Plot in x-y plane (side view)
        if len(fluid_pos) > 0:
            ax.scatter(fluid_pos[:, 0] / H0, fluid_pos[:, 1] / H0,
                      c='steelblue', s=1, alpha=0.3, label='Fluid')
        
        if len(solid_pos) > 0:
            ax.scatter(solid_pos[:, 0] / H0, solid_pos[:, 1] / H0,
                      c='saddlebrown', s=2, alpha=0.8, label='Solid')
        
        # Add ground line
        ax.axhline(y=0, color='black', linewidth=2)
        ax.fill_between([0, domain_length/H0], [-0.1, -0.1], [0, 0], 
                       color='gray', alpha=0.3)
        
        # Initial column outline (dashed)
        ax.plot([0, L0/H0, L0/H0, 0, 0], 
               [0, 0, 1, 1, 0], 
               'r--', linewidth=1.5, alpha=0.5, label='Initial')
        
        ax.set_xlim(-0.1, 3.5)
        ax.set_ylim(-0.15, 1.3)
        ax.set_aspect('equal')
        ax.set_xlabel('x/H₀', fontsize=11)
        ax.set_ylabel('y/H₀', fontsize=11)
        ax.set_title(f't/t$_{{ref}}$ = {actual_t:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', markerscale=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collapse_morphology.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'collapse_morphology.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: collapse_morphology.png/pdf")
    
    # ========== Figure 2: Wave front evolution ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wave front position
    wave_fronts = np.array(history['wave_front'])
    runouts = np.array(history['runout'])
    
    ax1.plot(t_norm, runouts / L0, 'b-', linewidth=2, marker='o', 
            markersize=4, markevery=5, label='Simulation')
    
    # Add reference points (approximate from Ceccato figure)
    ref_t = [0.44, 1.10, 2.20, 5.0]
    ref_runout = [0.1, 0.8, 1.5, 2.2]  # Approximate x_front/L0 values
    ax1.scatter(ref_t, ref_runout, c='red', s=100, marker='s', 
               zorder=5, label='Ceccato et al. (approx.)')
    
    ax1.set_xlabel('t/t$_{ref}$', fontsize=12)
    ax1.set_ylabel('Runout / L₀', fontsize=12)
    ax1.set_title('Wave Front Propagation', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 6])
    ax1.set_ylim([0, 3])
    
    # Height evolution
    max_heights = np.array(history['max_height'])
    ax2.plot(t_norm, max_heights / H0, 'g-', linewidth=2, marker='s',
            markersize=4, markevery=5)
    ax2.set_xlabel('t/t$_{ref}$', fontsize=12)
    ax2.set_ylabel('Max Height / H₀', fontsize=12)
    ax2.set_title('Maximum Height Evolution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 6])
    ax2.set_ylim([0, 1.2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wave_front_evolution.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'wave_front_evolution.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: wave_front_evolution.png/pdf")
    
    # ========== Figure 3: Velocity field ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Select frames at t/t_ref ≈ 1.0 and 2.0
    for idx, target_t in enumerate([1.0, 2.0]):
        frame_idx = np.argmin(np.abs(t_norm - target_t))
        solid_pos = history['solid_positions'][frame_idx]
        solid_vel = history['solid_velocities'][frame_idx]
        fluid_pos = history['fluid_positions'][frame_idx]
        fluid_vel = history['fluid_velocities'][frame_idx]
        
        # Solid velocity
        ax = axes[idx, 0]
        if len(solid_pos) > 0:
            vel_mag = np.linalg.norm(solid_vel, axis=1)
            sc = ax.scatter(solid_pos[:, 0]/H0, solid_pos[:, 1]/H0, 
                          c=vel_mag, cmap='YlOrRd', s=3,
                          vmin=0, vmax=max(1.5, np.percentile(vel_mag, 95)))
            plt.colorbar(sc, ax=ax, label='|v| (m/s)')
        ax.set_xlim(-0.1, 3.5)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('x/H₀')
        ax.set_ylabel('y/H₀')
        ax.set_title(f'Solid Velocity | t/t$_{{ref}}$ = {t_norm[frame_idx]:.2f}')
        ax.axhline(y=0, color='black', linewidth=1)
        
        # Fluid velocity  
        ax = axes[idx, 1]
        if len(fluid_pos) > 0:
            vel_mag = np.linalg.norm(fluid_vel, axis=1)
            sc = ax.scatter(fluid_pos[:, 0]/H0, fluid_pos[:, 1]/H0,
                          c=vel_mag, cmap='Blues', s=3,
                          vmin=0, vmax=max(1.5, np.percentile(vel_mag, 95)))
            plt.colorbar(sc, ax=ax, label='|v| (m/s)')
        ax.set_xlim(-0.1, 3.5)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('x/H₀')
        ax.set_ylabel('y/H₀')
        ax.set_title(f'Fluid Velocity | t/t$_{{ref}}$ = {t_norm[frame_idx]:.2f}')
        ax.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_fields.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'velocity_fields.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: velocity_fields.png/pdf")
    
    # ========== Figure 4: Comparison with experiment schematic ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw experimental reference (schematic)
    ax.text(0.5, 0.98, 'Comparison: Simulation vs Experiment (Ceccato et al., 2020)',
           transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Draw multiple snapshots
    colors = ['blue', 'green', 'orange', 'red']
    for idx, (target_t, color) in enumerate(zip([0.44, 1.10, 2.20, 5.0], colors)):
        frame_idx = np.argmin(np.abs(t_norm - target_t))
        solid_pos = history['solid_positions'][frame_idx]
        
        if len(solid_pos) > 0:
            # Get convex hull approximation
            x_pos = solid_pos[:, 0] / H0
            y_pos = solid_pos[:, 1] / H0
            
            # Plot as scatter with transparency
            ax.scatter(x_pos, y_pos, c=color, s=1, alpha=0.3,
                      label=f't/t$_{{ref}}$ = {target_t:.2f}')
    
    # Initial column
    ax.plot([0, L0/H0, L0/H0, 0, 0], [0, 0, 1, 1, 0], 
           'k--', linewidth=2, label='Initial')
    
    ax.axhline(y=0, color='brown', linewidth=3)
    ax.set_xlim(-0.2, 4.0)
    ax.set_ylim(-0.2, 1.4)
    ax.set_aspect('equal')
    ax.set_xlabel('x/H₀', fontsize=12)
    ax.set_ylabel('y/H₀', fontsize=12)
    ax.legend(loc='upper right', markerscale=5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: experiment_comparison.png/pdf")


if __name__ == '__main__':
    history, output_dir = run_ceccato_simulation()

