#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
3D Two-Phase MPM Simulation with 2D Paper-Style Figure Generation

This script runs 3D simulations on GPU and generates 2D visualizations
similar to the figures in the reference paper (main.pdf).

Features:
1. Saturated Granular Column Collapse (Ceccato et al. 2020)
2. Dam Break Simulation
3. Two-Phase Debris Flow

Output:
- 2D XY-plane visualization (side view)
- 2D XZ-plane visualization (top view)
- Velocity field contours
- Solid volume fraction distribution
- Time evolution plots
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Taichi based on platform
import platform
import taichi as ti

if platform.system() == 'Darwin':
    # macOS - use CPU
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    print("[INFO] Running on macOS CPU")
else:
    # Linux/Windows - use CUDA GPU
    ti.init(arch=ti.cuda, default_fp=ti.f64, device_memory_fraction=0.8)
    print("[INFO] Running on CUDA GPU")

from taichi_mpm.core.two_phase_solver import TwoPhaseMPMSolver


# ============================================================
# Visualization Settings
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class TwoPhaseSimulator:
    """3D Two-Phase MPM Simulator with 2D visualization"""
    
    def __init__(self, scenario='collapse'):
        """
        Initialize simulator
        
        Args:
            scenario: 'collapse' for column collapse, 'dambreak' for dam break
        """
        self.scenario = scenario
        self.setup_scenario()
        
    def setup_scenario(self):
        """Setup simulation parameters based on scenario"""
        
        if self.scenario == 'collapse':
            # Saturated Granular Column Collapse (Ceccato et al. 2020)
            self.L0 = 0.07          # Initial column length (m)
            self.H0 = 0.12          # Initial column height (m)
            self.W0 = 0.05          # Column width (m)
            
            self.domain_length = 0.5
            self.domain_height = 0.25
            self.domain_width = 0.10
            
            self.dx = 0.005
            self.dt = 1e-5  # Smaller timestep for stability
            self.total_time = 0.25  # ~2.3 * t_ref for full flow development
            
            self.rho_s = 2650.0
            self.rho_f = 1000.0
            self.E_s = 2e4           # Lower stiffness for more flow
            self.nu_s = 0.35         # Slightly higher for more lateral spread
            self.friction_angle = 22.0  # Lower friction for more flow
            self.phi_s0 = 0.48       # More fluid content for liquefied flow
            self.d_s = 0.0003        # Smaller grain size for more fluid-like behavior
            
        elif self.scenario == 'dambreak':
            # Dam Break Simulation
            self.L0 = 0.10          # Water column length (m)
            self.H0 = 0.15          # Water column height (m)
            self.W0 = 0.05          # Column width (m)
            
            self.domain_length = 0.8  # Longer domain for runout
            self.domain_height = 0.30
            self.domain_width = 0.10
            
            self.dx = 0.005  # Finer grid
            self.dt = 1e-5   # Smaller timestep for stability
            self.total_time = 0.30  # Longer simulation
            
            self.rho_s = 2000.0     # Lighter for dam break
            self.rho_f = 1000.0
            self.E_s = 3e4          # Slightly stiffer
            self.nu_s = 0.3
            self.friction_angle = 22.0  # Lower friction for more flow
            self.phi_s0 = 0.40      # More fluid content for dam break
            self.d_s = 0.0003
            
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        
        # Gravity and reference scales
        self.g = 9.81
        self.t_ref = np.sqrt(self.H0 / self.g)
        
        # Grid dimensions
        self.nx = int(self.domain_length / self.dx) + 4
        self.ny = int(self.domain_height / self.dx) + 4
        self.nz = int(self.domain_width / self.dx) + 4
        
        # Max particles
        self.max_particles = 80000
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f'simulation_output/{self.scenario}_3d_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # History storage
        self.history = {
            'time': [],
            't_normalized': [],
            'solid_pos': [],
            'fluid_pos': [],
            'solid_vel': [],
            'fluid_vel': [],
            'wave_front': [],
            'max_height': [],
            'runout': []
        }
        
    def create_solver(self):
        """Create the two-phase MPM solver"""
        
        print(f"\n{'='*60}")
        print(f"Creating Two-Phase MPM Solver")
        print(f"{'='*60}")
        print(f"Scenario: {self.scenario}")
        print(f"Grid: {self.nx} x {self.ny} x {self.nz}")
        print(f"dx = {self.dx:.4f} m, dt = {self.dt:.2e} s")
        print(f"Domain: {self.domain_length:.2f} x {self.domain_height:.2f} x {self.domain_width:.2f} m")
        print(f"Reference time t_ref = {self.t_ref:.4f} s")
        
        self.solver = TwoPhaseMPMSolver(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.dx,
            dt=self.dt,
            max_particles=self.max_particles,
            rho_s=self.rho_s,
            rho_f=self.rho_f,
            E_s=self.E_s,
            nu_s=self.nu_s,
            friction_angle=self.friction_angle,
            d_s=self.d_s,
            phi_s0=self.phi_s0,
            g=self.g,
            flip_ratio=0.95  # Higher for less numerical viscosity
        )
        
    def initialize_particles(self):
        """Initialize particles based on scenario"""
        
        margin = self.dx * 2
        
        x_min = margin
        x_max = margin + self.L0
        y_min = margin
        y_max = margin + self.H0
        z_min = margin
        z_max = margin + self.W0
        
        print(f"\nInitializing particles in region:")
        print(f"  x: [{x_min:.3f}, {x_max:.3f}] m")
        print(f"  y: [{y_min:.3f}, {y_max:.3f}] m")
        print(f"  z: [{z_min:.3f}, {z_max:.3f}] m")
        
        self.solver.initialize_particles_two_phase(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
            ppc=2
        )
        
        self.n_solid = self.solver.n_solid[None]
        self.n_fluid = self.solver.n_fluid[None]
        
        print(f"  Solid particles: {self.n_solid}")
        print(f"  Fluid particles: {self.n_fluid}")
        
        # Store initial position for reference
        self.x_min_init = x_min
        self.y_min_init = y_min
        
    def run_simulation(self, n_snapshots=20):
        """Run simulation and save snapshots"""
        
        n_steps = int(self.total_time / self.dt)
        save_interval = max(1, n_steps // n_snapshots)
        
        print(f"\n{'='*60}")
        print(f"Running 3D Simulation")
        print(f"{'='*60}")
        print(f"Total steps: {n_steps}")
        print(f"Total time: {self.total_time:.3f} s ({self.total_time/self.t_ref:.2f} t_ref)")
        print(f"Snapshots: ~{n_snapshots}")
        print("-" * 60)
        
        for step in range(n_steps + 1):
            t = step * self.dt
            t_norm = t / self.t_ref
            
            # Save snapshot
            if step % save_interval == 0 or step == n_steps:
                self._save_snapshot(t, t_norm)
                
            # Advance simulation
            if step < n_steps:
                self.solver.clear_grid()
                self.solver.p2g_solid()
                self.solver.p2g_fluid()
                self.solver.grid_operations(1.0)
                self.solver.g2p_solid()
                self.solver.g2p_fluid()
        
        print("-" * 60)
        print(f"Simulation completed! Saved {len(self.history['time'])} snapshots")
        
    def _save_snapshot(self, t, t_norm):
        """Save a simulation snapshot"""
        
        # Get particle data
        solid_pos = self.solver.x_s.to_numpy()[:self.n_solid]
        fluid_pos = self.solver.x_f.to_numpy()[:self.n_fluid]
        solid_vel = self.solver.v_s.to_numpy()[:self.n_solid]
        fluid_vel = self.solver.v_f.to_numpy()[:self.n_fluid]
        
        # Compute metrics
        if len(solid_pos) > 0:
            wave_front = np.max(solid_pos[:, 0])
            max_height = np.max(solid_pos[:, 1])
            runout = wave_front - self.x_min_init
        else:
            wave_front = self.x_min_init
            max_height = 0
            runout = 0
        
        # Store
        self.history['time'].append(t)
        self.history['t_normalized'].append(t_norm)
        self.history['solid_pos'].append(solid_pos.copy())
        self.history['fluid_pos'].append(fluid_pos.copy())
        self.history['solid_vel'].append(solid_vel.copy())
        self.history['fluid_vel'].append(fluid_vel.copy())
        self.history['wave_front'].append(wave_front)
        self.history['max_height'].append(max_height)
        self.history['runout'].append(runout)
        
        # Progress output
        max_vel_s = np.max(np.linalg.norm(solid_vel, axis=1)) if len(solid_vel) > 0 else 0
        print(f"  t/t_ref={t_norm:5.2f} | runout/L0={runout/self.L0:5.2f} | "
              f"h_max/H0={max_height/self.H0:5.2f} | max_vel={max_vel_s:.2f} m/s")
    
    def generate_2d_figures(self):
        """Generate 2D paper-style figures from 3D simulation"""
        
        print(f"\n{'='*60}")
        print(f"Generating 2D Paper-Style Figures")
        print(f"{'='*60}")
        
        # Figure 1: Morphology evolution (XY side view)
        self._plot_morphology_evolution()
        
        # Figure 2: Velocity field analysis
        self._plot_velocity_fields()
        
        # Figure 3: Wave front and height evolution
        self._plot_time_evolution()
        
        # Figure 4: Combined snapshot comparison
        self._plot_combined_comparison()
        
        # Figure 5: Solid volume fraction distribution
        self._plot_volume_fraction()
        
        print(f"\nAll figures saved to: {self.output_dir}")
        
    def _plot_morphology_evolution(self):
        """Plot flow morphology at key time instants (2D XY view)"""
        
        # Select 4 key frames
        n_frames = len(self.history['time'])
        indices = [0, n_frames//3, 2*n_frames//3, n_frames-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax_idx, frame_idx in enumerate(indices):
            ax = axes[ax_idx]
            
            t_norm = self.history['t_normalized'][frame_idx]
            solid_pos = self.history['solid_pos'][frame_idx]
            fluid_pos = self.history['fluid_pos'][frame_idx]
            
            # Plot fluid phase (back layer)
            if len(fluid_pos) > 0:
                ax.scatter(fluid_pos[:, 0]/self.H0, fluid_pos[:, 1]/self.H0,
                          c='steelblue', s=1, alpha=0.3, label='Fluid')
            
            # Plot solid phase (front layer)
            if len(solid_pos) > 0:
                ax.scatter(solid_pos[:, 0]/self.H0, solid_pos[:, 1]/self.H0,
                          c='saddlebrown', s=2, alpha=0.7, label='Solid')
            
            # Initial column outline
            ax.plot([self.x_min_init/self.H0, (self.x_min_init + self.L0)/self.H0,
                    (self.x_min_init + self.L0)/self.H0, self.x_min_init/self.H0,
                    self.x_min_init/self.H0],
                   [self.y_min_init/self.H0, self.y_min_init/self.H0,
                    (self.y_min_init + self.H0)/self.H0, (self.y_min_init + self.H0)/self.H0,
                    self.y_min_init/self.H0],
                   'r--', linewidth=1.5, alpha=0.5, label='Initial')
            
            # Ground line
            ax.axhline(y=0, color='brown', linewidth=3)
            ax.fill_between([0, self.domain_length/self.H0], [-0.1, -0.1], [0, 0],
                           color='tan', alpha=0.5)
            
            ax.set_xlim(-0.1, self.domain_length/self.H0)
            ax.set_ylim(-0.15, (self.domain_height*0.9)/self.H0)
            ax.set_aspect('equal')
            ax.set_xlabel('x/H₀', fontsize=12)
            ax.set_ylabel('y/H₀', fontsize=12)
            ax.set_title(f't/t$_{{ref}}$ = {t_norm:.2f}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if ax_idx == 0:
                ax.legend(loc='upper right', markerscale=3)
        
        fig.suptitle(f'Flow Morphology Evolution - {self.scenario.title()} Simulation\n'
                    f'(3D→2D XY Projection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_morphology_evolution.png', dpi=300)
        plt.savefig(self.output_dir / 'fig1_morphology_evolution.pdf')
        plt.close()
        print("  Saved: fig1_morphology_evolution.png/pdf")
        
    def _plot_velocity_fields(self):
        """Plot velocity field contours"""
        
        # Select 2 key frames
        n_frames = len(self.history['time'])
        indices = [n_frames//3, 2*n_frames//3]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for row, frame_idx in enumerate(indices):
            t_norm = self.history['t_normalized'][frame_idx]
            solid_pos = self.history['solid_pos'][frame_idx]
            solid_vel = self.history['solid_vel'][frame_idx]
            fluid_pos = self.history['fluid_pos'][frame_idx]
            fluid_vel = self.history['fluid_vel'][frame_idx]
            
            # Solid velocity field
            ax = axes[row, 0]
            if len(solid_pos) > 0:
                vel_mag = np.linalg.norm(solid_vel, axis=1)
                vmax = max(1.0, np.percentile(vel_mag, 95))
                sc = ax.scatter(solid_pos[:, 0]/self.H0, solid_pos[:, 1]/self.H0,
                              c=vel_mag, cmap='YlOrRd', s=3, vmin=0, vmax=vmax)
                plt.colorbar(sc, ax=ax, label='|v| (m/s)', shrink=0.8)
            
            ax.set_xlim(-0.1, self.domain_length/self.H0)
            ax.set_ylim(-0.1, (self.domain_height*0.8)/self.H0)
            ax.set_aspect('equal')
            ax.set_xlabel('x/H₀')
            ax.set_ylabel('y/H₀')
            ax.set_title(f'Solid Phase Velocity | t/t$_{{ref}}$ = {t_norm:.2f}')
            ax.axhline(y=0, color='black', linewidth=2)
            
            # Fluid velocity field
            ax = axes[row, 1]
            if len(fluid_pos) > 0:
                vel_mag = np.linalg.norm(fluid_vel, axis=1)
                vmax = max(1.0, np.percentile(vel_mag, 95))
                sc = ax.scatter(fluid_pos[:, 0]/self.H0, fluid_pos[:, 1]/self.H0,
                              c=vel_mag, cmap='Blues', s=3, vmin=0, vmax=vmax)
                plt.colorbar(sc, ax=ax, label='|v| (m/s)', shrink=0.8)
            
            ax.set_xlim(-0.1, self.domain_length/self.H0)
            ax.set_ylim(-0.1, (self.domain_height*0.8)/self.H0)
            ax.set_aspect('equal')
            ax.set_xlabel('x/H₀')
            ax.set_ylabel('y/H₀')
            ax.set_title(f'Fluid Phase Velocity | t/t$_{{ref}}$ = {t_norm:.2f}')
            ax.axhline(y=0, color='black', linewidth=2)
        
        fig.suptitle(f'Velocity Field Analysis - {self.scenario.title()} Simulation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_velocity_fields.png', dpi=300)
        plt.savefig(self.output_dir / 'fig2_velocity_fields.pdf')
        plt.close()
        print("  Saved: fig2_velocity_fields.png/pdf")
        
    def _plot_time_evolution(self):
        """Plot wave front and height evolution over time"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        t_norm = np.array(self.history['t_normalized'])
        runout = np.array(self.history['runout'])
        max_height = np.array(self.history['max_height'])
        wave_front = np.array(self.history['wave_front'])
        
        # Runout evolution
        ax = axes[0]
        ax.plot(t_norm, runout/self.L0, 'b-', linewidth=2.5, marker='o', 
               markersize=4, markevery=3)
        ax.set_xlabel('t/t$_{ref}$', fontsize=12)
        ax.set_ylabel('Runout / L₀', fontsize=12)
        ax.set_title('Wave Front Runout', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(t_norm)])
        
        # Height evolution
        ax = axes[1]
        ax.plot(t_norm, max_height/self.H0, 'g-', linewidth=2.5, marker='s',
               markersize=4, markevery=3)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Initial H₀')
        ax.set_xlabel('t/t$_{ref}$', fontsize=12)
        ax.set_ylabel('Max Height / H₀', fontsize=12)
        ax.set_title('Maximum Height Evolution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, max(t_norm)])
        
        # Normalized runout vs aspect ratio (final state)
        ax = axes[2]
        aspect_ratio = self.L0 / self.H0
        final_runout = runout[-1] / self.L0 if len(runout) > 0 else 0
        
        # Plot empirical relation
        a_values = np.linspace(0.1, 3, 50)
        # Empirical: runout/L0 ≈ k * (L0/H0)^n for saturated collapse
        runout_emp = 2.5 * a_values ** 0.7
        ax.plot(a_values, runout_emp, 'k--', linewidth=1.5, 
               label='Empirical: $2.5(L_0/H_0)^{0.7}$')
        ax.scatter([aspect_ratio], [final_runout], c='red', s=150, marker='*',
                  zorder=5, label=f'This simulation')
        
        ax.set_xlabel('Aspect Ratio L₀/H₀', fontsize=12)
        ax.set_ylabel('Final Runout / L₀', fontsize=12)
        ax.set_title('Runout vs Aspect Ratio', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 3])
        ax.set_ylim([0, 5])
        
        fig.suptitle(f'Time Evolution Analysis - {self.scenario.title()} Simulation',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_time_evolution.png', dpi=300)
        plt.savefig(self.output_dir / 'fig3_time_evolution.pdf')
        plt.close()
        print("  Saved: fig3_time_evolution.png/pdf")
        
    def _plot_combined_comparison(self):
        """Plot combined comparison with all snapshots overlaid"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color gradient for time progression
        n_frames = len(self.history['time'])
        colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
        
        for idx in range(0, n_frames, max(1, n_frames//8)):  # Plot ~8 snapshots
            t_norm = self.history['t_normalized'][idx]
            solid_pos = self.history['solid_pos'][idx]
            
            if len(solid_pos) > 0:
                ax.scatter(solid_pos[:, 0]/self.H0, solid_pos[:, 1]/self.H0,
                          c=[colors[idx]], s=1, alpha=0.4,
                          label=f't/t$_{{ref}}$ = {t_norm:.2f}')
        
        # Initial column
        ax.plot([self.x_min_init/self.H0, (self.x_min_init + self.L0)/self.H0,
                (self.x_min_init + self.L0)/self.H0, self.x_min_init/self.H0,
                self.x_min_init/self.H0],
               [self.y_min_init/self.H0, self.y_min_init/self.H0,
                (self.y_min_init + self.H0)/self.H0, (self.y_min_init + self.H0)/self.H0,
                self.y_min_init/self.H0],
               'k--', linewidth=2.5, label='Initial Column')
        
        # Ground
        ax.axhline(y=0, color='brown', linewidth=4)
        
        ax.set_xlim(-0.2, self.domain_length/self.H0)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x/H₀', fontsize=12)
        ax.set_ylabel('y/H₀', fontsize=12)
        ax.set_title(f'Flow Evolution Comparison - {self.scenario.title()} Simulation\n'
                    f'(Time Progression: Light → Dark)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2, fontsize=9, markerscale=5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_combined_comparison.png', dpi=300)
        plt.savefig(self.output_dir / 'fig4_combined_comparison.pdf')
        plt.close()
        print("  Saved: fig4_combined_comparison.png/pdf")
        
    def _plot_volume_fraction(self):
        """Plot solid volume fraction distribution"""
        
        # Select middle frame
        n_frames = len(self.history['time'])
        frame_idx = n_frames // 2
        
        t_norm = self.history['t_normalized'][frame_idx]
        solid_pos = self.history['solid_pos'][frame_idx]
        fluid_pos = self.history['fluid_pos'][frame_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create 2D histogram for solid volume fraction
        ax = axes[0]
        x_bins = np.linspace(0, self.domain_length, 50)
        y_bins = np.linspace(0, self.domain_height, 30)
        
        if len(solid_pos) > 0 and len(fluid_pos) > 0:
            # Count solid particles in each bin
            H_solid, _, _ = np.histogram2d(solid_pos[:, 0], solid_pos[:, 1],
                                          bins=[x_bins, y_bins])
            H_fluid, _, _ = np.histogram2d(fluid_pos[:, 0], fluid_pos[:, 1],
                                          bins=[x_bins, y_bins])
            
            # Compute solid volume fraction
            total = H_solid + H_fluid + 1e-10
            phi_s = H_solid / total
            
            # Plot
            X, Y = np.meshgrid(x_bins[:-1]/self.H0, y_bins[:-1]/self.H0)
            pcm = ax.pcolormesh(X, Y, phi_s.T, cmap='YlOrBr', vmin=0, vmax=1)
            plt.colorbar(pcm, ax=ax, label='Solid Volume Fraction φ_s')
        
        ax.set_xlabel('x/H₀')
        ax.set_ylabel('y/H₀')
        ax.set_title(f'Solid Volume Fraction Distribution\nt/t$_{{ref}}$ = {t_norm:.2f}')
        ax.set_aspect('equal')
        
        # Phase distribution plot
        ax = axes[1]
        if len(solid_pos) > 0:
            ax.scatter(solid_pos[:, 0]/self.H0, solid_pos[:, 1]/self.H0,
                      c='brown', s=2, alpha=0.5, label='Solid')
        if len(fluid_pos) > 0:
            ax.scatter(fluid_pos[:, 0]/self.H0, fluid_pos[:, 1]/self.H0,
                      c='blue', s=1, alpha=0.3, label='Fluid')
        
        ax.set_xlim(-0.1, self.domain_length/self.H0)
        ax.set_ylim(-0.1, (self.domain_height*0.8)/self.H0)
        ax.set_aspect('equal')
        ax.set_xlabel('x/H₀')
        ax.set_ylabel('y/H₀')
        ax.set_title(f'Two-Phase Distribution\nt/t$_{{ref}}$ = {t_norm:.2f}')
        ax.axhline(y=0, color='black', linewidth=2)
        ax.legend(loc='upper right', markerscale=5)
        
        fig.suptitle(f'Volume Fraction Analysis - {self.scenario.title()} Simulation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_volume_fraction.png', dpi=300)
        plt.savefig(self.output_dir / 'fig5_volume_fraction.pdf')
        plt.close()
        print("  Saved: fig5_volume_fraction.png/pdf")
        
    def save_data(self):
        """Save simulation data"""
        np.savez(self.output_dir / 'simulation_data.npz',
                 time=self.history['time'],
                 t_normalized=self.history['t_normalized'],
                 wave_front=self.history['wave_front'],
                 max_height=self.history['max_height'],
                 runout=self.history['runout'],
                 L0=self.L0, H0=self.H0, t_ref=self.t_ref)
        print(f"  Saved: simulation_data.npz")


def main():
    """Main entry point"""
    
    import argparse
    parser = argparse.ArgumentParser(description='3D Two-Phase MPM Simulation with 2D Figures')
    parser.add_argument('--scenario', type=str, default='collapse',
                       choices=['collapse', 'dambreak'],
                       help='Simulation scenario')
    parser.add_argument('--snapshots', type=int, default=20,
                       help='Number of snapshots to save')
    args = parser.parse_args()
    
    print("=" * 70)
    print("3D Two-Phase MPM Simulation with 2D Paper-Style Figures")
    print("=" * 70)
    
    # Create simulator
    sim = TwoPhaseSimulator(scenario=args.scenario)
    
    # Create solver and initialize
    sim.create_solver()
    sim.initialize_particles()
    
    # Run simulation
    sim.run_simulation(n_snapshots=args.snapshots)
    
    # Generate figures
    sim.generate_2d_figures()
    
    # Save data
    sim.save_data()
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print(f"Output directory: {sim.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

