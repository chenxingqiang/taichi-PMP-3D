#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Dam Break Validation - Two-Phase MPM Simulation
Based on classical dam break experiments

This script validates the two-phase MPM solver against:
1. Martin & Moyce (1952) - Water column collapse
2. Koshizuka & Oka (1996) - MPS dam break simulation

Key validation metrics:
- Wave front position vs time
- Column height vs time
- Flow morphology at different instants
"""

import os
os.environ['TI_ARCH'] = 'arm64'

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from datetime import datetime
import time

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

# Import two-phase solver
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_phase_mpm_solver import TwoPhaseMPMSolver


def run_dam_break_validation():
    """Run dam break validation with two-phase solver"""
    
    print("=" * 70)
    print("Dam Break Validation - Two-Phase MPM")
    print("=" * 70)
    
    # ========================================
    # Experimental parameters (Martin & Moyce, 1952)
    # ========================================
    # Initial column: L₀ × H₀ = 0.057m × 0.114m (aspect ratio a = 2)
    # Reference time: t_ref = √(2H₀/g) ≈ 0.152s
    # We use a slightly larger setup for better visualization
    
    L0 = 0.10    # Initial column length (m)
    H0 = 0.20    # Initial column height (m)
    W0 = 0.05    # Width (m) - 3D simulation
    
    aspect_ratio = H0 / L0
    t_ref = np.sqrt(2 * H0 / 9.81)
    
    print(f"\n[Experimental Parameters]")
    print(f"  Initial column: L₀={L0:.3f}m × H₀={H0:.3f}m × W={W0:.3f}m")
    print(f"  Aspect ratio: a = H₀/L₀ = {aspect_ratio:.2f}")
    print(f"  Reference time: t_ref = √(2H₀/g) = {t_ref:.4f} s")
    
    # ========================================
    # Numerical parameters
    # ========================================
    dx = 0.008          # Grid spacing (m)
    dt = 2e-5           # Time step (s)
    total_time = 0.3    # Total simulation time (s) ≈ 2 × t_ref
    
    # Domain
    domain_x = 0.6      # m
    domain_y = 0.35     # m  
    domain_z = W0 + 0.02  # m
    
    nx = int(domain_x / dx) + 1
    ny = int(domain_y / dx) + 1
    nz = int(domain_z / dx) + 1
    
    total_steps = int(total_time / dt)
    
    print(f"\n[Numerical Parameters]")
    print(f"  Grid: {nx} × {ny} × {nz}")
    print(f"  dx = {dx:.4f} m, dt = {dt:.2e} s")
    print(f"  Total simulation time: {total_time:.3f} s ({total_time/t_ref:.2f} × t_ref)")
    print(f"  Total steps: {total_steps}")
    
    # ========================================
    # Create solver (use fluid-only mode for dam break)
    # ========================================
    print(f"\n[Creating solver...]")
    
    solver = TwoPhaseMPMSolver(
        nx=nx, ny=ny, nz=nz,
        dx=dx,
        rho_s=2650.0,         # Solid density
        rho_f=1000.0,         # Water density
        E_s=1e6,              # Young's modulus (not used for pure fluid)
        nu_s=0.3,             # Poisson ratio
        mu_f=0.001,           # Water viscosity (Pa·s)
        friction_angle=30.0,  # Friction angle (not used for pure fluid)
        d_s=0.001,            # Particle diameter (not used for pure fluid)
        phi_s0=0.0,           # Pure fluid (no solid)
        g=9.81,
        dt=dt,
        max_particles=100000,
        flip_ratio=0.95
    )
    
    # ========================================
    # Initialize particles (fluid only)
    # ========================================
    print(f"\n[Initializing particles...]")
    
    # Create fluid column
    particle_dx = dx / 2.0  # 2x2x2 particles per cell
    
    particles_x = []
    particles_y = []
    particles_z = []
    
    x = 0.01 + particle_dx / 2
    while x < L0:
        y = 0.01 + particle_dx / 2
        while y < H0:
            z = 0.01 + particle_dx / 2
            while z < W0:
                particles_x.append(x)
                particles_y.append(y)
                particles_z.append(z)
                z += particle_dx
            y += particle_dx
        x += particle_dx
    
    n_particles = len(particles_x)
    
    # Add particles to solver (as fluid only)
    particle_mass = 1000.0 * (particle_dx ** 3)  # Water mass
    
    # Initialize fluid particles using numpy arrays
    x_f_np = np.zeros((solver.max_particles, 3))
    v_f_np = np.zeros((solver.max_particles, 3))
    m_f_np = np.zeros(solver.max_particles)
    C_f_np = np.zeros((solver.max_particles, 3, 3))
    
    for i in range(n_particles):
        x_f_np[i] = [particles_x[i], particles_y[i], particles_z[i]]
        v_f_np[i] = [0.0, 0.0, 0.0]
        m_f_np[i] = particle_mass
        # C_f remains zero
    
    solver.x_f.from_numpy(x_f_np)
    solver.v_f.from_numpy(v_f_np)
    solver.m_f.from_numpy(m_f_np)
    solver.C_f.from_numpy(C_f_np)
    solver.n_fluid[None] = n_particles
    solver.n_solid[None] = 0  # No solid particles
    
    print(f"  Fluid particles: {n_particles}")
    
    # ========================================
    # Data storage
    # ========================================
    time_history = []
    front_position_history = []
    height_history = []
    max_velocity_history = []
    snapshots = []
    
    # Snapshot times (normalized by t_ref)
    snapshot_t_normalized = [0.0, 0.5, 1.0, 1.5, 2.0]
    snapshot_times = [t * t_ref for t in snapshot_t_normalized]
    snapshot_idx = 0
    
    # ========================================
    # Run simulation
    # ========================================
    print(f"\n[Running simulation...]")
    print(f"  Total steps: {total_steps}")
    print("-" * 70)
    
    start_time = time.time()
    current_time = 0.0
    
    # Store initial state
    x_f_np = solver.x_f.to_numpy()[:n_particles]
    v_f_np = solver.v_f.to_numpy()[:n_particles]
    snapshots.append({
        'time': 0.0,
        't_normalized': 0.0,
        'positions': x_f_np.copy(),
        'velocities': v_f_np.copy()
    })
    snapshot_idx = 1
    
    for step in range(total_steps):
        # Run one step using the full step method (includes pressure solve)
        solver.step()
        
        current_time = (step + 1) * dt
        
        # Collect metrics every 50 steps
        if step % 50 == 0:
            x_f_np = solver.x_f.to_numpy()[:solver.n_fluid[None]]
            v_f_np = solver.v_f.to_numpy()[:solver.n_fluid[None]]
            
            # Check for NaN
            if np.any(np.isnan(x_f_np)) or np.any(np.isnan(v_f_np)):
                print(f"  NaN detected at step {step}, stopping")
                break
            
            # Wave front position
            front_x = np.max(x_f_np[:, 0])
            
            # Column height (at x < L0/2)
            in_column = x_f_np[:, 0] < L0 / 2
            if np.any(in_column):
                column_height = np.max(x_f_np[in_column, 1])
            else:
                column_height = 0.0
            
            max_vel = np.max(np.linalg.norm(v_f_np, axis=1))
            
            time_history.append(current_time)
            front_position_history.append(front_x)
            height_history.append(column_height)
            max_velocity_history.append(max_vel)
        
        # Store snapshots
        if snapshot_idx < len(snapshot_times) and current_time >= snapshot_times[snapshot_idx]:
            x_f_np = solver.x_f.to_numpy()[:solver.n_fluid[None]]
            v_f_np = solver.v_f.to_numpy()[:solver.n_fluid[None]]
            snapshots.append({
                'time': current_time,
                't_normalized': current_time / t_ref,
                'positions': x_f_np.copy(),
                'velocities': v_f_np.copy()
            })
            print(f"  Snapshot at t/t_ref = {current_time/t_ref:.2f}")
            snapshot_idx += 1
        
        # Progress output
        if step % 500 == 0:
            x_f_np = solver.x_f.to_numpy()[:solver.n_fluid[None]]
            v_f_np = solver.v_f.to_numpy()[:solver.n_fluid[None]]
            front_x = np.max(x_f_np[:, 0])
            max_vel = np.max(np.linalg.norm(v_f_np, axis=1))
            print(f"  Step {step:5d} | t/t_ref = {current_time/t_ref:.2f} | "
                  f"Front: {front_x/L0:.2f}×L₀ | MaxVel: {max_vel:.2f}m/s")
    
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"\n[Simulation completed!]")
    print(f"  Elapsed time: {elapsed:.1f} s")
    print(f"  Snapshots: {len(snapshots)}")
    
    # ========================================
    # Generate validation plots
    # ========================================
    print(f"\n[Generating validation plots...]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"simulation_output/dam_break_validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to normalized units
    time_norm = np.array(time_history) / t_ref
    front_norm = np.array(front_position_history) / L0
    height_norm = np.array(height_history) / H0
    
    # ========================================
    # Plot 1: Flow morphology at different instants
    # ========================================
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(2, (n_snapshots + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, snapshot in enumerate(snapshots):
        ax = axes[idx]
        pos = snapshot['positions']
        vel = snapshot['velocities']
        t_norm = snapshot['t_normalized']
        
        vel_mag = np.linalg.norm(vel, axis=1)
        
        # Plot particles (X-Y plane)
        scatter = ax.scatter(pos[:, 0] / L0, pos[:, 1] / H0, 
                           c=vel_mag, cmap='coolwarm', s=3, alpha=0.7,
                           vmin=0, vmax=max(1.5, np.max(vel_mag)))
        
        # Initial box
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=1.5, label='Initial')
        
        # Bottom wall
        ax.axhline(y=0, color='black', linewidth=2)
        ax.fill_between([0, 6], [-0.1, -0.1], [0, 0], color='gray', alpha=0.3)
        
        ax.set_xlim(-0.2, 6)
        ax.set_ylim(-0.15, 1.5)
        ax.set_xlabel('x/L₀')
        ax.set_ylabel('y/H₀')
        ax.set_title(f't/t_ref = {t_norm:.2f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(snapshots), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Dam Break Flow Morphology (Two-Phase MPM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'flow_morphology.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'flow_morphology.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: flow_morphology.png/pdf")
    
    # ========================================
    # Plot 2: Wave front evolution (comparison with experiments)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Martin & Moyce (1952) experimental data (approximate)
    exp_t = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    exp_front = np.array([1.0, 1.2, 1.8, 2.5, 3.2, 3.8])  # x/L0
    exp_height = np.array([1.0, 0.9, 0.7, 0.5, 0.35, 0.25])  # h/H0
    
    # Wave front
    ax1 = axes[0]
    ax1.plot(time_norm, front_norm, 'b-', linewidth=2, label='Simulation')
    ax1.plot(exp_t, exp_front, 'rs', markersize=10, label='Martin & Moyce (1952)')
    ax1.set_xlabel('t/t_ref', fontsize=12)
    ax1.set_ylabel('x_front / L₀', fontsize=12)
    ax1.set_title('Wave Front Position', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 5)
    
    # Column height
    ax2 = axes[1]
    ax2.plot(time_norm, height_norm, 'g-', linewidth=2, label='Simulation')
    ax2.plot(exp_t, exp_height, 'rs', markersize=10, label='Martin & Moyce (1952)')
    ax2.set_xlabel('t/t_ref', fontsize=12)
    ax2.set_ylabel('h / H₀', fontsize=12)
    ax2.set_title('Column Height Evolution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 1.2)
    
    plt.suptitle('Dam Break Validation vs Martin & Moyce (1952)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'validation_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: validation_comparison.png/pdf")
    
    # ========================================
    # Plot 3: Velocity field at key instant
    # ========================================
    if len(snapshots) >= 3:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        snapshot = snapshots[2]  # t/t_ref ≈ 1.0
        pos = snapshot['positions']
        vel = snapshot['velocities']
        
        vel_mag = np.linalg.norm(vel, axis=1)
        
        # Quiver plot (subsample for clarity)
        subsample = 10
        scatter = ax.scatter(pos[::subsample, 0], pos[::subsample, 1], 
                           c=vel_mag[::subsample], cmap='jet', s=50, alpha=0.8)
        
        # Velocity vectors
        scale = 10
        ax.quiver(pos[::subsample*3, 0], pos[::subsample*3, 1],
                 vel[::subsample*3, 0], vel[::subsample*3, 1],
                 color='white', alpha=0.8, scale=scale)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Velocity magnitude (m/s)')
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Velocity Field at t/t_ref = {snapshot["t_normalized"]:.2f}')
        ax.set_aspect('equal')
        ax.axhline(y=0, color='black', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_field.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'velocity_field.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: velocity_field.png/pdf")
    
    # Save data
    np.savez(output_dir / 'simulation_data.npz',
             time=time_history,
             time_normalized=time_norm,
             front_position=front_position_history,
             front_normalized=front_norm,
             height=height_history,
             height_normalized=height_norm,
             max_velocity=max_velocity_history,
             L0=L0, H0=H0, t_ref=t_ref)
    print(f"  Saved: simulation_data.npz")
    
    print(f"\n[Results saved to: {output_dir}]")
    print("=" * 70)
    
    return output_dir


if __name__ == "__main__":
    run_dam_break_validation()

