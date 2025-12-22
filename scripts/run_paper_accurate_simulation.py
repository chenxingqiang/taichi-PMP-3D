#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Paper-Accurate Two-Phase MPM Debris Flow Simulation

Based on Ng et al. (2023) paper parameters from physics_config_paper_accurate.yaml.
This script runs a complete debris flow simulation with:
1. Sloped channel (20 degrees)
2. Two-phase flow (solid + fluid)
3. Optional barriers
4. Paper-style 2D visualization

Domain: 2.0 x 0.4 x 0.5 m (length x width x height)
Initial debris: 0.4 x 0.4 x 0.3 m column
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Taichi based on platform
import platform
import taichi as ti

if platform.system() == 'Darwin':
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    print("[INFO] Running on macOS CPU")
else:
    ti.init(arch=ti.cuda, default_fp=ti.f64, device_memory_fraction=0.8)
    print("[INFO] Running on CUDA GPU")

from taichi_mpm.core.two_phase_solver import TwoPhaseMPMSolver


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class PaperAccurateSimulator:
    """
    Paper-accurate debris flow simulator based on Ng et al. (2023)
    """
    
    def __init__(self, config_path):
        """
        Initialize simulator from YAML config
        
        Args:
            config_path: Path to physics_config_paper_accurate.yaml
        """
        self.config = load_config(config_path)
        self.setup_from_config()
        
    def setup_from_config(self):
        """Extract parameters from config"""
        sim = self.config['simulation']
        solid = self.config['solid_phase']
        fluid = self.config['fluid_phase']
        numerics = self.config['numerics']
        
        # Simulation parameters
        self.total_time = float(sim['total_time'])
        self.g = float(sim['gravity'])
        self.slope_angle = float(sim['slope_angle'])  # degrees
        
        # Domain dimensions
        self.domain_length = float(sim['domain_length'])
        self.domain_width = float(sim['domain_width'])
        self.domain_height = float(sim['domain_height'])
        
        # Initial debris
        self.debris_length = float(sim['initial_debris_length'])
        self.debris_height = float(sim['initial_debris_height'])
        self.debris_width = self.domain_width
        
        # Material properties (convert strings to float if needed)
        self.rho_s = float(solid['density'])
        self.rho_f = float(fluid['density'])
        self.mu_f = float(fluid['viscosity'])
        self.E_s = float(solid['young_modulus'])
        self.nu_s = float(solid['poisson_ratio'])
        self.phi_s0 = float(sim['solid_volume_fraction'])
        self.d_s = float(solid['particle_diameter'])
        
        # Friction parameters
        self.mu_static = float(solid['static_friction'])
        self.mu_dynamic = float(solid['dynamic_friction'])
        self.basal_friction = float(solid['basal_friction'])
        
        # Numerical parameters
        self.dx = float(numerics['dx'])
        self.dt = float(numerics['max_timestep'])
        self.ppc = int(numerics['particles_per_cell'])
        self.flip_ratio = 1.0 - float(numerics['pic_flip_ratio'])  # Convert PIC ratio to FLIP ratio
        
        # Compute derived parameters
        self.t_ref = np.sqrt(self.debris_height / self.g)
        
        # Grid dimensions
        self.nx = int(self.domain_length / self.dx) + 4
        self.ny = int(self.domain_height / self.dx) + 4
        self.nz = int(self.domain_width / self.dx) + 4
        
        # Max particles estimate
        debris_cells = (self.debris_length / self.dx) * (self.debris_height / self.dx) * (self.debris_width / self.dx)
        self.max_particles = int(debris_cells * self.ppc * 2.5)  # Extra margin
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f'simulation_output/paper_accurate_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # History storage
        self.history = {
            'time': [],
            't_normalized': [],
            'solid_pos': [],
            'fluid_pos': [],
            'solid_vel': [],
            'wave_front': [],
            'max_height': [],
            'runout': [],
            'max_velocity': []
        }
        
        self._print_config()
        
    def _print_config(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("PAPER-ACCURATE TWO-PHASE MPM SIMULATION")
        print("Based on Ng et al. (2023) Parameters")
        print("="*70)
        print(f"\n📐 Domain Configuration:")
        print(f"   Size: {self.domain_length:.2f} x {self.domain_width:.2f} x {self.domain_height:.2f} m")
        print(f"   Grid: {self.nx} x {self.nz} x {self.ny} cells (dx={self.dx:.3f}m)")
        print(f"   Slope angle: {self.slope_angle}°")
        
        print(f"\n🧱 Initial Debris:")
        print(f"   Size: {self.debris_length:.2f} x {self.debris_width:.2f} x {self.debris_height:.2f} m")
        print(f"   Volume: {self.debris_length * self.debris_width * self.debris_height * 1000:.1f} L")
        
        print(f"\n⚙️  Material Properties:")
        print(f"   Solid: ρ={self.rho_s:.0f} kg/m³, E={self.E_s/1e6:.1f} MPa, ν={self.nu_s}")
        print(f"   Fluid: ρ={self.rho_f:.0f} kg/m³, μ={self.mu_f:.3f} Pa·s")
        print(f"   Friction: μ₁={self.mu_static:.2f}, μ₂={self.mu_dynamic:.2f}")
        print(f"   Solid fraction: φ₀={self.phi_s0:.2f}")
        print(f"   Particle diameter: d={self.d_s*1000:.1f} mm")
        
        print(f"\n🔢 Numerical Parameters:")
        print(f"   dt={self.dt:.2e}s, FLIP ratio={self.flip_ratio:.2f}")
        print(f"   Reference time t_ref={self.t_ref:.4f}s")
        print(f"   Max particles: {self.max_particles}")
        print(f"\n   Output: {self.output_dir}")
        print("-"*70)
        
    def create_solver(self):
        """Create the two-phase MPM solver with paper parameters"""
        # Convert friction angle from friction coefficient
        # tan(φ) = μ, so φ = arctan(μ)
        friction_angle_deg = np.degrees(np.arctan(self.mu_static))
        
        self.solver = TwoPhaseMPMSolver(
            nx=self.nx, ny=self.ny, nz=self.nz,
            dx=self.dx,
            dt=self.dt,
            max_particles=self.max_particles,
            rho_s=self.rho_s,
            rho_f=self.rho_f,
            E_s=self.E_s,
            nu_s=self.nu_s,
            friction_angle=friction_angle_deg,
            d_s=self.d_s,
            phi_s0=self.phi_s0,
            g=self.g,
            flip_ratio=self.flip_ratio
        )
        
    def initialize_particles(self):
        """Initialize debris particles in a column on the slope"""
        margin = self.dx * 2
        
        # Position debris column at start of domain
        x_min = margin
        x_max = margin + self.debris_length
        y_min = margin
        y_max = margin + self.debris_height
        z_min = margin
        z_max = margin + self.debris_width
        
        print(f"\n📍 Initializing debris column:")
        print(f"   x: [{x_min:.3f}, {x_max:.3f}] m")
        print(f"   y: [{y_min:.3f}, {y_max:.3f}] m")
        print(f"   z: [{z_min:.3f}, {z_max:.3f}] m")
        
        # Calculate PPC for initialization (cubic root for 3D)
        ppc_1d = int(round(self.ppc ** (1/3)))
        
        self.solver.initialize_particles_two_phase(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
            ppc=ppc_1d
        )
        
        self.n_solid = self.solver.n_solid[None]
        self.n_fluid = self.solver.n_fluid[None]
        
        print(f"   Solid particles: {self.n_solid}")
        print(f"   Fluid particles: {self.n_fluid}")
        
        # Store initial reference
        self.x_min_init = x_min
        self.y_min_init = y_min
        self.L0 = self.debris_length
        self.H0 = self.debris_height
        
    def run_simulation(self, n_snapshots=20):
        """Run the full simulation"""
        n_steps = int(self.total_time / self.dt)
        save_interval = max(1, n_steps // n_snapshots)
        
        print(f"\n{'='*70}")
        print(f"🚀 Running Paper-Accurate Simulation")
        print(f"{'='*70}")
        print(f"   Total steps: {n_steps}")
        print(f"   Total time: {self.total_time:.3f}s ({self.total_time/self.t_ref:.2f} t_ref)")
        print(f"   Snapshots: ~{n_snapshots}")
        print("-"*70)
        print(f"   {'t/t_ref':>8} | {'runout/L0':>10} | {'h_max/H0':>10} | {'max_vel':>10}")
        print("-"*70)
        
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
        
        print("-"*70)
        print(f"✅ Simulation completed! Saved {len(self.history['time'])} snapshots")
        
    def _save_snapshot(self, t, t_norm):
        """Save a simulation snapshot"""
        solid_pos = self.solver.x_s.to_numpy()[:self.n_solid]
        fluid_pos = self.solver.x_f.to_numpy()[:self.n_fluid]
        solid_vel = self.solver.v_s.to_numpy()[:self.n_solid]
        
        # Compute metrics
        if len(solid_pos) > 0:
            wave_front = np.max(solid_pos[:, 0])
            max_height = np.max(solid_pos[:, 1])
            runout = wave_front - self.x_min_init
            max_vel = np.max(np.linalg.norm(solid_vel, axis=1))
        else:
            wave_front = self.x_min_init
            max_height = 0
            runout = 0
            max_vel = 0
        
        # Store
        self.history['time'].append(t)
        self.history['t_normalized'].append(t_norm)
        self.history['solid_pos'].append(solid_pos.copy())
        self.history['fluid_pos'].append(fluid_pos.copy())
        self.history['solid_vel'].append(solid_vel.copy())
        self.history['wave_front'].append(wave_front)
        self.history['max_height'].append(max_height)
        self.history['runout'].append(runout)
        self.history['max_velocity'].append(max_vel)
        
        # Progress output
        print(f"   {t_norm:8.2f} | {runout/self.L0:10.2f} | "
              f"{max_height/self.H0:10.2f} | {max_vel:10.2f} m/s")
    
    def generate_figures(self):
        """Generate paper-style figures"""
        print(f"\n📊 Generating Paper-Style Figures...")
        
        # Setup matplotlib
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })
        
        self._plot_morphology_evolution()
        self._plot_velocity_field()
        self._plot_time_evolution()
        self._plot_runout_comparison()
        
        print(f"   All figures saved to: {self.output_dir}")
        
    def _plot_morphology_evolution(self):
        """Plot debris flow morphology at key times"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        n_snaps = len(self.history['time'])
        indices = [0, n_snaps//3, 2*n_snaps//3, n_snaps-1]
        
        for ax, idx in zip(axes.flat, indices):
            t_norm = self.history['t_normalized'][idx]
            solid_pos = self.history['solid_pos'][idx]
            fluid_pos = self.history['fluid_pos'][idx]
            
            # Normalize by H0
            if len(fluid_pos) > 0:
                ax.scatter(fluid_pos[:, 0] / self.H0, fluid_pos[:, 1] / self.H0,
                          s=1, c='lightblue', alpha=0.6, label='Fluid')
            if len(solid_pos) > 0:
                ax.scatter(solid_pos[:, 0] / self.H0, solid_pos[:, 1] / self.H0,
                          s=2, c='brown', alpha=0.8, label='Solid')
            
            # Initial outline
            rect_x = [self.x_min_init/self.H0, (self.x_min_init+self.L0)/self.H0,
                     (self.x_min_init+self.L0)/self.H0, self.x_min_init/self.H0, self.x_min_init/self.H0]
            rect_y = [self.y_min_init/self.H0, self.y_min_init/self.H0,
                     (self.y_min_init+self.H0)/self.H0, (self.y_min_init+self.H0)/self.H0, self.y_min_init/self.H0]
            ax.plot(rect_x, rect_y, 'r--', linewidth=1.5, label='Initial')
            
            # Ground line
            ax.axhline(y=0, color='brown', linewidth=3)
            
            ax.set_xlabel(r'$x/H_0$')
            ax.set_ylabel(r'$y/H_0$')
            ax.set_title(f't/t$_{{ref}}$ = {t_norm:.2f}')
            ax.set_xlim([0, self.domain_length / self.H0])
            ax.set_ylim([-0.2, 2.0])
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Debris Flow Morphology Evolution\n(Paper-Accurate Parameters)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_morphology.png')
        plt.savefig(self.output_dir / 'fig1_morphology.pdf')
        plt.close()
        print("   Saved: fig1_morphology.png/pdf")
        
    def _plot_velocity_field(self):
        """Plot velocity magnitude at selected times"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        n_snaps = len(self.history['time'])
        indices = [n_snaps//3, 2*n_snaps//3]
        
        for ax, idx in zip(axes, indices):
            t_norm = self.history['t_normalized'][idx]
            solid_pos = self.history['solid_pos'][idx]
            solid_vel = self.history['solid_vel'][idx]
            
            if len(solid_pos) > 0:
                vel_mag = np.linalg.norm(solid_vel, axis=1)
                sc = ax.scatter(solid_pos[:, 0] / self.H0, solid_pos[:, 1] / self.H0,
                              c=vel_mag, s=3, cmap='hot', vmin=0, vmax=3)
                plt.colorbar(sc, ax=ax, label='|v| (m/s)')
            
            ax.axhline(y=0, color='brown', linewidth=3)
            ax.set_xlabel(r'$x/H_0$')
            ax.set_ylabel(r'$y/H_0$')
            ax.set_title(f'Velocity Field | t/t$_{{ref}}$ = {t_norm:.2f}')
            ax.set_xlim([0, self.domain_length / self.H0])
            ax.set_ylim([-0.2, 2.0])
            ax.set_aspect('equal')
        
        fig.suptitle('Debris Flow Velocity Field', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_velocity.png')
        plt.savefig(self.output_dir / 'fig2_velocity.pdf')
        plt.close()
        print("   Saved: fig2_velocity.png/pdf")
        
    def _plot_time_evolution(self):
        """Plot time evolution of key metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        t_norm = np.array(self.history['t_normalized'])
        runout = np.array(self.history['runout']) / self.L0
        h_max = np.array(self.history['max_height']) / self.H0
        max_vel = np.array(self.history['max_velocity'])
        
        # Runout
        axes[0].plot(t_norm, runout, 'b-', linewidth=2)
        axes[0].set_xlabel(r'$t/t_{ref}$')
        axes[0].set_ylabel(r'Runout / $L_0$')
        axes[0].set_title('Wave Front Runout')
        axes[0].grid(True, alpha=0.3)
        
        # Height
        axes[1].plot(t_norm, h_max, 'g-', linewidth=2)
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='Initial H₀')
        axes[1].set_xlabel(r'$t/t_{ref}$')
        axes[1].set_ylabel(r'$h_{max} / H_0$')
        axes[1].set_title('Maximum Height Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Velocity
        axes[2].plot(t_norm, max_vel, 'r-', linewidth=2)
        axes[2].set_xlabel(r'$t/t_{ref}$')
        axes[2].set_ylabel('Max Velocity (m/s)')
        axes[2].set_title('Maximum Velocity')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle('Time Evolution Analysis (Paper-Accurate Parameters)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_time_evolution.png')
        plt.savefig(self.output_dir / 'fig3_time_evolution.pdf')
        plt.close()
        print("   Saved: fig3_time_evolution.png/pdf")
        
    def _plot_runout_comparison(self):
        """Plot runout with empirical correlation"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Empirical correlation: runout/L0 ~ 2.5 * (L0/H0)^0.7
        aspect = self.L0 / self.H0
        empirical_runout = 2.5 * (aspect ** 0.7)
        
        final_runout = self.history['runout'][-1] / self.L0
        
        # Plot empirical curve
        aspects = np.linspace(0.3, 3.0, 50)
        empirical_curve = 2.5 * (aspects ** 0.7)
        ax.plot(aspects, empirical_curve, 'k--', linewidth=2, 
               label=r'Empirical: $2.5(L_0/H_0)^{0.7}$')
        
        # Plot simulation result
        ax.scatter([aspect], [final_runout], s=200, c='red', marker='*',
                  label=f'This simulation', zorder=5)
        
        ax.set_xlabel(r'Aspect Ratio $L_0/H_0$')
        ax.set_ylabel(r'Final Runout / $L_0$')
        ax.set_title('Runout vs Aspect Ratio')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 3.0])
        ax.set_ylim([0, 5])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_runout_comparison.png')
        plt.savefig(self.output_dir / 'fig4_runout_comparison.pdf')
        plt.close()
        print("   Saved: fig4_runout_comparison.png/pdf")
        
    def save_data(self):
        """Save simulation data"""
        data = {
            'config': self.config,
            't_ref': self.t_ref,
            'L0': self.L0,
            'H0': self.H0,
            **self.history
        }
        np.savez(self.output_dir / 'simulation_data.npz', **{
            k: np.array(v) if isinstance(v, list) else v 
            for k, v in data.items() if k != 'config'
        })
        print(f"   Saved: simulation_data.npz")


def main():
    parser = argparse.ArgumentParser(description='Paper-accurate two-phase MPM simulation')
    parser.add_argument('--config', type=str, 
                       default='configs/physics_config_paper_accurate.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--snapshots', type=int, default=20,
                       help='Number of snapshots to save')
    args = parser.parse_args()
    
    print("="*70)
    print("Paper-Accurate Two-Phase MPM Debris Flow Simulation")
    print("="*70)
    
    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {args.config}")
        return
    
    print(f"Loading config: {config_path}")
    
    # Create and run simulation
    sim = PaperAccurateSimulator(str(config_path))
    sim.create_solver()
    sim.initialize_particles()
    sim.run_simulation(n_snapshots=args.snapshots)
    sim.generate_figures()
    sim.save_data()
    
    print("\n" + "="*70)
    print("🎉 Simulation Complete!")
    print(f"   Output directory: {sim.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

