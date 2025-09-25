"""
Output Metrics Calculator for Two-Phase MPM Debris Flow Impact
Implements key output quantities from Ng et al. (2023)
"""

import taichi as ti
import numpy as np
from typing import Dict, Any, Tuple
import yaml

@ti.data_oriented
class OutputMetricsCalculator:
    """
    Calculator for key output metrics in debris flow impact simulations.
    
    Implements:
    - Fluidization ratio calculation (Equation 26)
    - Impact force analysis (Equation 1)
    - Flow velocity and depth statistics
    - Barrier effectiveness metrics
    """
    
    def __init__(self, config_path: str = "physics_config.yaml"):
        """Initialize with physics configuration."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Physical parameters
        self.rho_s = self.config['solid_phase']['density']
        self.rho_f = self.config['fluid_phase']['density']
        self.phi_m = self.config['solid_phase']['critical_solid_fraction']
        self.mu_1 = self.config['solid_phase']['static_friction']
        self.mu_2 = self.config['solid_phase']['limiting_friction']
        self.d = self.config['solid_phase']['particle_diameter']
        self.eta_f = self.config['fluid_phase']['viscosity']
        
        # Simulation parameters
        self.g = self.config['simulation']['gravity']
        self.slope_angle = self.config['simulation']['slope_angle']
        
        # Output fields
        self.fluidization_ratio = ti.field(ti.f64, shape=())
        self.bed_friction_reduction = ti.field(ti.f64, shape=())
        self.flow_velocity_magnitude = ti.field(ti.f64, shape=())
        self.flow_depth = ti.field(ti.f64, shape=())
        self.solid_volume_fraction = ti.field(ti.f64, shape=())
        
        # Impact force metrics
        self.hydrodynamic_force = ti.field(ti.f64, shape=())
        self.static_force = ti.field(ti.f64, shape=())
        self.total_impact_force = ti.field(ti.f64, shape=())
        
        # Barrier effectiveness
        self.capture_efficiency = ti.field(ti.f64, shape=())
        self.overflow_ratio = ti.field(ti.f64, shape=())
        self.energy_dissipation = ti.field(ti.f64, shape=())
        
        # Statistics
        self.mean_fluidization_ratio = ti.field(ti.f64, shape=())
        self.max_fluidization_ratio = ti.field(ti.f64, shape=())
        self.mean_flow_velocity = ti.field(ti.f64, shape=())
        self.max_flow_velocity = ti.field(ti.f64, shape=())
        
    @ti.kernel
    def compute_fluidization_ratio(self,
                                  pressure_field: ti.template(),
                                  stress_field: ti.template(),
                                  n_particles: int):
        """
        Calculate fluidization ratio λ = p_bed/(p_bed + σ'_bed) (Equation 26).
        
        Args:
            pressure_field: Pore fluid pressure field
            stress_field: Effective stress field
            n_particles: Number of particles
        """
        total_fluidization = 0.0
        max_fluidization = 0.0
        valid_particles = 0
        
        for i in range(n_particles):
            p_bed = pressure_field[i]
            sigma_bed = stress_field[i]
            
            # Avoid division by zero
            if p_bed + sigma_bed > 1e-12:
                lambda_val = p_bed / (p_bed + sigma_bed)
                
                # Clamp to physical bounds [0, 1]
                lambda_val = ti.max(0.0, ti.min(1.0, lambda_val))
                
                total_fluidization += lambda_val
                max_fluidization = ti.max(max_fluidization, lambda_val)
                valid_particles += 1
        
        if valid_particles > 0:
            self.mean_fluidization_ratio[None] = total_fluidization / valid_particles
            self.max_fluidization_ratio[None] = max_fluidization
        else:
            self.mean_fluidization_ratio[None] = 0.0
            self.max_fluidization_ratio[None] = 0.0
    
    @ti.kernel
    def compute_bed_friction_reduction(self,
                                     mu_bed: ti.f64,
                                     sigma_bed: ti.f64,
                                     lambda_val: ti.f64):
        """
        Calculate bed friction reduction F_fric = μ_bed × σ_bed × (1 - λ) (Equation 25).
        
        Args:
            mu_bed: Bed friction coefficient
            sigma_bed: Normal stress at bed
            lambda_val: Fluidization ratio
        """
        self.bed_friction_reduction[None] = mu_bed * sigma_bed * (1.0 - lambda_val)
    
    @ti.kernel
    def compute_impact_forces(self,
                            flow_velocity: ti.template(),
                            flow_depth: ti.template(),
                            flow_density: ti.template(),
                            n_particles: int):
        """
        Calculate impact forces using Equation 1.
        
        F = αρv²h + (k/2)h²ρ||g||
        where:
        - α = dynamic impact coefficient (= 1.0)
        - k = static impact coefficient (= 1.0)
        - ρ = debris flow density
        - v = flow velocity at barrier
        - h = flow depth at barrier
        """
        total_hydrodynamic = 0.0
        total_static = 0.0
        total_force = 0.0
        
        alpha = 1.0  # Dynamic impact coefficient
        k = 1.0      # Static impact coefficient
        
        for i in range(n_particles):
            v = flow_velocity[i]
            h = flow_depth[i]
            rho = flow_density[i]
            
            # Hydrodynamic component: αρv²h
            hydrodynamic = alpha * rho * v * v * h
            total_hydrodynamic += hydrodynamic
            
            # Static component: (k/2)h²ρ||g||
            static = (k / 2.0) * h * h * rho * self.g
            total_static += static
            
            # Total impact force
            total_force += hydrodynamic + static
        
        self.hydrodynamic_force[None] = total_hydrodynamic
        self.static_force[None] = total_static
        self.total_impact_force[None] = total_force
    
    @ti.kernel
    def compute_flow_statistics(self,
                              velocity_field: ti.template(),
                              depth_field: ti.template(),
                              volume_fraction_field: ti.template(),
                              n_particles: int):
        """
        Compute flow velocity, depth, and volume fraction statistics.
        """
        total_velocity = 0.0
        total_depth = 0.0
        total_volume_fraction = 0.0
        max_velocity = 0.0
        max_depth = 0.0
        
        for i in range(n_particles):
            v = velocity_field[i]
            h = depth_field[i]
            phi = volume_fraction_field[i]
            
            total_velocity += v
            total_depth += h
            total_volume_fraction += phi
            
            max_velocity = ti.max(max_velocity, v)
            max_depth = ti.max(max_depth, h)
        
        if n_particles > 0:
            self.mean_flow_velocity[None] = total_velocity / n_particles
            self.max_flow_velocity[None] = max_velocity
            self.flow_depth[None] = total_depth / n_particles
            self.solid_volume_fraction[None] = total_volume_fraction / n_particles
        else:
            self.mean_flow_velocity[None] = 0.0
            self.max_flow_velocity[None] = 0.0
            self.flow_depth[None] = 0.0
            self.solid_volume_fraction[None] = 0.0
    
    @ti.kernel
    def compute_barrier_effectiveness(self,
                                    total_particles: int,
                                    captured_particles: int,
                                    overflow_particles: int,
                                    initial_kinetic_energy: ti.f64,
                                    final_kinetic_energy: ti.f64):
        """
        Calculate barrier effectiveness metrics.
        
        Args:
            total_particles: Total number of particles
            captured_particles: Particles captured by barriers
            overflow_particles: Particles that overflowed barriers
            initial_kinetic_energy: Initial kinetic energy
            final_kinetic_energy: Final kinetic energy
        """
        # Capture efficiency: fraction of particles captured
        if total_particles > 0:
            self.capture_efficiency[None] = captured_particles / total_particles
        else:
            self.capture_efficiency[None] = 0.0
        
        # Overflow ratio: fraction of particles that overflowed
        if total_particles > 0:
            self.overflow_ratio[None] = overflow_particles / total_particles
        else:
            self.overflow_ratio[None] = 0.0
        
        # Energy dissipation: fraction of kinetic energy dissipated
        if initial_kinetic_energy > 1e-12:
            self.energy_dissipation[None] = (initial_kinetic_energy - final_kinetic_energy) / initial_kinetic_energy
        else:
            self.energy_dissipation[None] = 0.0
    
    def compute_drag_coefficient(self, solid_fraction: float, reynolds_number: float) -> float:
        """
        Compute drag coefficient F̂(Re, φ) for inter-phase coupling.
        
        Implements the drag correlation from the paper.
        """
        # Simplified drag correlation (can be enhanced with full correlation)
        if solid_fraction < 0.1:
            # Dilute regime
            return 1.0 + 0.15 * reynolds_number**0.687
        elif solid_fraction < 0.4:
            # Intermediate regime
            return 1.0 + 0.15 * reynolds_number**0.687 * (1.0 + 4.0 * solid_fraction)
        else:
            # Dense regime
            return 1.0 + 0.15 * reynolds_number**0.687 * (1.0 + 4.0 * solid_fraction) / (1.0 - solid_fraction)**2
    
    def compute_reynolds_number(self, velocity: float, particle_diameter: float, 
                              fluid_density: float, fluid_viscosity: float) -> float:
        """
        Compute particle Reynolds number.
        
        Re = ρ_f * v * d / η_f
        """
        return fluid_density * velocity * particle_diameter / fluid_viscosity
    
    def get_fluidization_statistics(self) -> Dict[str, float]:
        """Get fluidization ratio statistics."""
        return {
            'mean_fluidization_ratio': self.mean_fluidization_ratio[None],
            'max_fluidization_ratio': self.max_fluidization_ratio[None],
            'bed_friction_reduction': self.bed_friction_reduction[None]
        }
    
    def get_impact_statistics(self) -> Dict[str, float]:
        """Get impact force statistics."""
        return {
            'hydrodynamic_force': self.hydrodynamic_force[None],
            'static_force': self.static_force[None],
            'total_impact_force': self.total_impact_force[None]
        }
    
    def get_flow_statistics(self) -> Dict[str, float]:
        """Get flow statistics."""
        return {
            'mean_flow_velocity': self.mean_flow_velocity[None],
            'max_flow_velocity': self.max_flow_velocity[None],
            'flow_depth': self.flow_depth[None],
            'solid_volume_fraction': self.solid_volume_fraction[None]
        }
    
    def get_barrier_effectiveness(self) -> Dict[str, float]:
        """Get barrier effectiveness metrics."""
        return {
            'capture_efficiency': self.capture_efficiency[None],
            'overflow_ratio': self.overflow_ratio[None],
            'energy_dissipation': self.energy_dissipation[None]
        }
    
    def export_all_metrics(self) -> Dict[str, Any]:
        """Export all computed metrics."""
        return {
            'fluidization': self.get_fluidization_statistics(),
            'impact_forces': self.get_impact_statistics(),
            'flow_statistics': self.get_flow_statistics(),
            'barrier_effectiveness': self.get_barrier_effectiveness(),
            'physical_parameters': {
                'solid_density': self.rho_s,
                'fluid_density': self.rho_f,
                'critical_volume_fraction': self.phi_m,
                'particle_diameter': self.d,
                'slope_angle': self.slope_angle
            }
        }
    
    def save_metrics_to_file(self, filename: str):
        """Save all metrics to YAML file."""
        metrics = self.export_all_metrics()
        
        with open(filename, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False, indent=2)
        
        print(f"Metrics saved to {filename}")
    
    def print_metrics_summary(self):
        """Print a summary of all computed metrics."""
        print("\n" + "="*60)
        print("OUTPUT METRICS SUMMARY")
        print("="*60)
        
        # Fluidization metrics
        fluidization = self.get_fluidization_statistics()
        print(f"Fluidization Ratio:")
        print(f"  Mean: {fluidization['mean_fluidization_ratio']:.4f}")
        print(f"  Max:  {fluidization['max_fluidization_ratio']:.4f}")
        print(f"  Bed Friction Reduction: {fluidization['bed_friction_reduction']:.2f} N")
        
        # Impact forces
        impact = self.get_impact_statistics()
        print(f"\nImpact Forces:")
        print(f"  Hydrodynamic: {impact['hydrodynamic_force']:.2f} N")
        print(f"  Static:       {impact['static_force']:.2f} N")
        print(f"  Total:        {impact['total_impact_force']:.2f} N")
        
        # Flow statistics
        flow = self.get_flow_statistics()
        print(f"\nFlow Statistics:")
        print(f"  Mean Velocity: {flow['mean_flow_velocity']:.3f} m/s")
        print(f"  Max Velocity:  {flow['max_flow_velocity']:.3f} m/s")
        print(f"  Flow Depth:    {flow['flow_depth']:.3f} m")
        print(f"  Solid Fraction: {flow['solid_volume_fraction']:.3f}")
        
        # Barrier effectiveness
        barrier = self.get_barrier_effectiveness()
        print(f"\nBarrier Effectiveness:")
        print(f"  Capture Efficiency: {barrier['capture_efficiency']:.3f}")
        print(f"  Overflow Ratio:     {barrier['overflow_ratio']:.3f}")
        print(f"  Energy Dissipation: {barrier['energy_dissipation']:.3f}")
        
        print("="*60)
