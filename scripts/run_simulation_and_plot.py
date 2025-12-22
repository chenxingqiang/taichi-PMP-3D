#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Dam Break Simulation with Visualization
Based on Ng et al. (2023) - Two-Phase MPM Debris Flow Impact

This script runs a dam break simulation and generates paper-style figures:
1. Flow morphology at different time instants
2. Velocity field visualization
3. Impact force time series
4. Pressure distribution
5. Wave front position vs time
"""

import os
os.environ['TI_ARCH'] = 'arm64'

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import yaml
import time
from pathlib import Path

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

# Import solver
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from incompressible_mpm_solver import IncompressibleMPMSolver

class SimulationRunner:
    """Run simulation and collect data for visualization"""
    
    def __init__(self, config_path='physics_config_paper_accurate.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract parameters
        sim = self.config['simulation']
        num = self.config['numerics']
        fluid = self.config['fluid_phase']
        
        self.total_time = sim['total_time']
        self.dt = num['max_timestep']
        self.dx = num['dx']
        
        # Domain setup
        self.domain_length = sim['domain_length']
        self.domain_width = sim['domain_width']
        self.domain_height = sim['domain_height']
        
        # Grid dimensions
        self.nx = int(self.domain_length / self.dx) + 1
        self.ny = int(self.domain_height / self.dx) + 1
        self.nz = int(self.domain_width / self.dx) + 1
        
        # Initial debris setup
        self.debris_length = sim.get('initial_debris_length', 0.4)
        self.debris_height = sim.get('initial_debris_height', 0.3)
        
        # Barrier setup
        self.barrier_height = sim['barrier_height']
        self.barrier_positions = sim['barrier_positions']
        
        # Calculate required number of particles
        particle_dx = self.dx / (num['particles_per_cell'] ** 0.5)
        n_particles_x = int(self.debris_length / particle_dx)
        n_particles_y = int(self.debris_height / particle_dx)
        n_particles_z = int(self.domain_width / particle_dx)
        estimated_particles = n_particles_x * n_particles_y * n_particles_z
        max_particles = max(estimated_particles + 10000, 200000)  # Add buffer
        
        print(f"  Estimated particles: {estimated_particles}")
        print(f"  Max particles: {max_particles}")
        
        # Create solver
        self.solver = IncompressibleMPMSolver(
            nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx,
            rho=fluid['density'],
            mu=fluid['viscosity'],
            gamma=fluid.get('surface_tension', 0.0),
            g=sim['gravity'],
            dt=self.dt,
            max_particles=max_particles,
            preconditioner=num.get('preconditioner', 'jacobi')
        )
        
        # Data storage
        self.time_history = []
        self.wave_front_history = []
        self.max_velocity_history = []
        self.kinetic_energy_history = []
        self.impact_force_history = []
        self.snapshots = []  # Store particle snapshots at key times
        
        # Output directory
        self.output_dir = Path('simulation_output/paper_figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Simulation Configuration:")
        print(f"  Domain: {self.domain_length}m x {self.domain_height}m x {self.domain_width}m")
        print(f"  Grid: {self.nx} x {self.ny} x {self.nz}")
        print(f"  dx: {self.dx}m, dt: {self.dt}s")
        print(f"  Total time: {self.total_time}s")
    
    def initialize(self):
        """Initialize particles and level set"""
        # Initialize dam break
        self.solver.initialize_particles_dam_break(
            x_min=0.0, x_max=self.debris_length,
            y_min=0.0, y_max=self.debris_height,
            z_min=0.0, z_max=self.domain_width,
            ppc=self.config['numerics']['particles_per_cell']
        )
        
        # Initialize level set
        self.solver.level_set_method.initialize_box(
            x_min=0.0, x_max=self.debris_length,
            y_min=0.0, y_max=self.debris_height,
            z_min=0.0, z_max=self.domain_width
        )
        self.solver.level_set_method.compute_gradient()
        
        print(f"Initialized {self.solver.n_particles[None]} particles")
    
    def run(self, snapshot_times=[0.0, 0.2, 0.5, 1.0, 1.5, 2.0]):
        """Run simulation and collect data"""
        total_steps = int(self.total_time / self.dt)
        current_time = 0.0
        snapshot_idx = 0
        
        print(f"\nRunning simulation for {total_steps} steps...")
        print("-" * 60)
        
        start_time = time.time()
        
        # Store initial snapshot
        if snapshot_times and current_time >= snapshot_times[snapshot_idx]:
            self._store_snapshot(current_time)
            snapshot_idx += 1
        
        for step in range(total_steps):
            # Run one step
            try:
                pcg_iters = self.solver.step(
                    pcg_max_iter=self.config['numerics']['max_pressure_iterations'],
                    pcg_tol=self.config['numerics']['pressure_tolerance']
                )
            except Exception as e:
                print(f"Error at step {step}: {e}")
                break
            
            current_time = (step + 1) * self.dt
            
            # Collect metrics
            pos, vel = self.solver.export_particles_to_numpy()
            
            # Check for NaN
            if np.any(np.isnan(vel)):
                print(f"NaN detected at step {step}, t={current_time:.4f}s")
                break
            
            # Store time history
            if step % 10 == 0:
                self.time_history.append(current_time)
                self.wave_front_history.append(np.max(pos[:, 0]))
                self.max_velocity_history.append(np.max(np.linalg.norm(vel, axis=1)))
                self.kinetic_energy_history.append(0.5 * np.sum(np.linalg.norm(vel, axis=1)**2))
                
                # Simplified impact force (momentum flux at barrier)
                if len(self.barrier_positions) > 0:
                    barrier_x = self.barrier_positions[0]
                    near_barrier = pos[:, 0] > (barrier_x - 0.1)
                    if np.any(near_barrier):
                        impact_vel = vel[near_barrier, 0]
                        impact_force = np.sum(np.abs(impact_vel)) * 1000  # ρ * v
                    else:
                        impact_force = 0.0
                else:
                    impact_force = 0.0
                self.impact_force_history.append(impact_force)
            
            # Store snapshots at specified times
            if snapshot_idx < len(snapshot_times) and current_time >= snapshot_times[snapshot_idx]:
                self._store_snapshot(current_time)
                snapshot_idx += 1
            
            # Progress output
            if step % 100 == 0:
                wave_x = np.max(pos[:, 0])
                max_vel = np.max(np.linalg.norm(vel, axis=1))
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | t={current_time:.4f}s | wave_x={wave_x:.3f}m | "
                      f"max_vel={max_vel:.3f}m/s | PCG={pcg_iters} | elapsed={elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"Simulation completed in {elapsed:.1f}s")
        print(f"Final time: {current_time:.4f}s")
        print(f"Snapshots stored: {len(self.snapshots)}")
    
    def _store_snapshot(self, t):
        """Store particle snapshot"""
        pos, vel = self.solver.export_particles_to_numpy()
        self.snapshots.append({
            'time': t,
            'positions': pos.copy(),
            'velocities': vel.copy()
        })
        print(f"  Snapshot stored at t={t:.4f}s")
    
    def plot_flow_morphology(self):
        """Plot flow morphology at different time instants (Figure 3 style)"""
        if len(self.snapshots) == 0:
            print("No snapshots available for plotting")
            return
        
        n_snapshots = len(self.snapshots)
        fig, axes = plt.subplots(n_snapshots, 1, figsize=(12, 3*n_snapshots))
        if n_snapshots == 1:
            axes = [axes]
        
        for idx, (ax, snapshot) in enumerate(zip(axes, self.snapshots)):
            pos = snapshot['positions']
            vel = snapshot['velocities']
            t = snapshot['time']
            
            # Velocity magnitude for coloring
            vel_mag = np.linalg.norm(vel, axis=1)
            
            # Plot particles (X-Y plane, averaged over Z)
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=vel_mag, 
                               cmap='jet', s=2, alpha=0.7,
                               vmin=0, vmax=max(1.0, np.max(vel_mag)))
            
            # Draw barriers
            for barrier_x in self.barrier_positions:
                if barrier_x < self.domain_length:
                    rect = patches.Rectangle(
                        (barrier_x - 0.02, 0), 0.04, self.barrier_height,
                        linewidth=1, edgecolor='black', facecolor='gray'
                    )
                    ax.add_patch(rect)
            
            # Draw ground
            ax.axhline(y=0, color='brown', linewidth=2)
            
            ax.set_xlim(-0.1, self.domain_length + 0.1)
            ax.set_ylim(-0.05, self.domain_height)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f't = {t:.2f} s')
            ax.set_aspect('equal')
            
            plt.colorbar(scatter, ax=ax, label='Velocity (m/s)')
        
        plt.suptitle('Flow Morphology Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flow_morphology.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'flow_morphology.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: flow_morphology.png/pdf")
    
    def plot_velocity_field(self):
        """Plot velocity field at key time instant (Figure 4 style)"""
        if len(self.snapshots) < 2:
            print("Not enough snapshots for velocity field plot")
            return
        
        # Use the middle snapshot
        snapshot = self.snapshots[len(self.snapshots)//2]
        pos = snapshot['positions']
        vel = snapshot['velocities']
        t = snapshot['time']
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Velocity magnitude
        vel_mag = np.linalg.norm(vel, axis=1)
        
        # Scatter plot with velocity magnitude
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=vel_mag,
                           cmap='coolwarm', s=5, alpha=0.8,
                           vmin=0, vmax=max(1.0, np.percentile(vel_mag, 95)))
        
        # Add velocity vectors (subsampled)
        subsample = max(1, len(pos) // 200)
        ax.quiver(pos[::subsample, 0], pos[::subsample, 1],
                 vel[::subsample, 0], vel[::subsample, 1],
                 scale=10, alpha=0.5, width=0.003)
        
        # Draw barriers
        for barrier_x in self.barrier_positions:
            if barrier_x < self.domain_length:
                rect = patches.Rectangle(
                    (barrier_x - 0.02, 0), 0.04, self.barrier_height,
                    linewidth=1, edgecolor='black', facecolor='gray'
                )
                ax.add_patch(rect)
        
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.set_xlim(-0.1, self.domain_length + 0.1)
        ax.set_ylim(-0.05, self.domain_height)
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'Velocity Field at t = {t:.2f} s', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Velocity Magnitude (m/s)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'velocity_field.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'velocity_field.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: velocity_field.png/pdf")
    
    def plot_wave_front_position(self):
        """Plot wave front position vs time (Figure 5 style)"""
        if len(self.time_history) == 0:
            print("No time history data for wave front plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulation data
        ax.plot(self.time_history, self.wave_front_history, 'b-', 
               linewidth=2, label='Simulation')
        
        # Analytical solution (simplified dam break)
        # x_front = x0 + 2*sqrt(g*h0)*t for ideal dam break
        g = self.config['simulation']['gravity']
        h0 = self.debris_height
        t_analytical = np.array(self.time_history)
        x_analytical = self.debris_length + 2 * np.sqrt(g * h0) * t_analytical
        ax.plot(t_analytical, x_analytical, 'r--', 
               linewidth=2, label='Analytical (ideal)')
        
        # Mark barrier positions
        for i, barrier_x in enumerate(self.barrier_positions):
            ax.axhline(y=barrier_x, color='gray', linestyle=':', alpha=0.7)
            ax.text(max(self.time_history)*0.95, barrier_x + 0.02, 
                   f'Barrier {i+1}', ha='right', fontsize=10)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Wave Front Position (m)', fontsize=12)
        ax.set_title('Wave Front Propagation', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.time_history))
        ax.set_ylim(0, self.domain_length)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'wave_front_position.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'wave_front_position.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: wave_front_position.png/pdf")
    
    def plot_impact_force(self):
        """Plot impact force time series (Figure 6 style)"""
        if len(self.time_history) == 0:
            print("No time history data for impact force plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize impact force
        impact_force = np.array(self.impact_force_history)
        if np.max(impact_force) > 0:
            impact_force_normalized = impact_force / np.max(impact_force)
        else:
            impact_force_normalized = impact_force
        
        ax.plot(self.time_history, impact_force_normalized, 'b-', 
               linewidth=2, label='Impact Force')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Normalized Impact Force', fontsize=12)
        ax.set_title('Impact Force on First Barrier', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.time_history))
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'impact_force.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'impact_force.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: impact_force.png/pdf")
    
    def plot_energy_evolution(self):
        """Plot kinetic energy evolution"""
        if len(self.time_history) == 0:
            print("No time history data for energy plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Kinetic energy
        ax1.plot(self.time_history, self.kinetic_energy_history, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Kinetic Energy (J)', fontsize=12)
        ax1.set_title('Kinetic Energy Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Maximum velocity
        ax2.plot(self.time_history, self.max_velocity_history, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Maximum Velocity (m/s)', fontsize=12)
        ax2.set_title('Maximum Velocity Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_evolution.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'energy_evolution.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: energy_evolution.png/pdf")
    
    def plot_pressure_distribution(self):
        """Plot pressure distribution from the last snapshot"""
        if len(self.snapshots) == 0:
            print("No snapshots available for pressure plot")
            return
        
        # Use the last snapshot
        snapshot = self.snapshots[-1]
        pos = snapshot['positions']
        t = snapshot['time']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Estimate pressure from height (hydrostatic approximation)
        rho = self.config['fluid_phase']['density']
        g = self.config['simulation']['gravity']
        max_y = np.max(pos[:, 1])
        pressure = rho * g * (max_y - pos[:, 1])  # Hydrostatic pressure
        
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=pressure/1000,  # kPa
                           cmap='viridis', s=5, alpha=0.8)
        
        # Draw barriers
        for barrier_x in self.barrier_positions:
            if barrier_x < self.domain_length:
                rect = patches.Rectangle(
                    (barrier_x - 0.02, 0), 0.04, self.barrier_height,
                    linewidth=1, edgecolor='black', facecolor='gray'
                )
                ax.add_patch(rect)
        
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.set_xlim(-0.1, self.domain_length + 0.1)
        ax.set_ylim(-0.05, self.domain_height)
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'Pressure Distribution at t = {t:.2f} s (Hydrostatic Estimate)', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pressure (kPa)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pressure_distribution.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pressure_distribution.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: pressure_distribution.png/pdf")
    
    def generate_all_plots(self):
        """Generate all paper figures"""
        print("\n" + "="*60)
        print("Generating Paper Figures")
        print("="*60 + "\n")
        
        self.plot_flow_morphology()
        self.plot_velocity_field()
        self.plot_wave_front_position()
        self.plot_impact_force()
        self.plot_energy_evolution()
        self.plot_pressure_distribution()
        
        print(f"\nAll figures saved to: {self.output_dir}")


def main():
    """Main entry point"""
    print("="*60)
    print("Two-Phase MPM Dam Break Simulation")
    print("Based on Ng et al. (2023)")
    print("="*60 + "\n")
    
    # Create simulation runner
    config_path = os.path.join(os.path.dirname(__file__), 'physics_config_paper_accurate.yaml')
    runner = SimulationRunner(config_path)
    
    # Initialize
    runner.initialize()
    
    # Run simulation with snapshots at specific times
    snapshot_times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    runner.run(snapshot_times=snapshot_times)
    
    # Generate all figures
    runner.generate_all_plots()
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

