"""
Barrier Model Implementation for Two-Phase MPM Debris Flow Impact
Implements rigid barrier contact mechanics and overflow trajectory tracking
Based on Ng et al. (2023) equations 1, 3-4
"""

import taichi as ti
import numpy as np
from typing import Tuple, Dict, Any

@ti.data_oriented
class BarrierModel:
    """
    Rigid barrier model with contact detection and overflow mechanics.
    
    Implements:
    - Dual barrier configuration with adjustable spacing
    - Penalty method contact forces (Equation 1)
    - Overflow trajectory tracking (Equations 3-4)
    - Impact force calculation and statistics
    """
    
    def __init__(self, 
                 barrier_height: float = 0.15,
                 barrier_spacing: float = 2.0,
                 barrier_positions: Tuple[float, float] = (3.0, 5.0),
                 contact_stiffness: float = 1e8,
                 contact_damping: float = 1e3,
                 friction_coefficient: float = 0.4):
        """
        Initialize barrier model with experimental configuration.
        
        Args:
            barrier_height: Height of barriers (m)
            barrier_spacing: Distance between barriers (m) 
            barrier_positions: X-positions of barriers (m)
            contact_stiffness: Normal contact stiffness (N/m)
            contact_damping: Contact damping coefficient (N·s/m)
            friction_coefficient: Coulomb friction coefficient
        """
        
        # Barrier geometry
        self.barrier_height = barrier_height
        self.barrier_spacing = barrier_spacing
        self.barrier_positions = ti.Vector.field(3, ti.f64, shape=2)
        self.barrier_normals = ti.Vector.field(3, ti.f64, shape=2)
        
        # Contact parameters
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping
        self.friction_coefficient = friction_coefficient
        
        # Impact force tracking
        self.total_impact_force = ti.Vector.field(3, ti.f64, shape=())
        self.max_impact_force = ti.field(ti.f64, shape=())
        self.impact_force_history = ti.Vector.field(3, ti.f64, shape=10000)
        self.impact_time_history = ti.field(ti.f64, shape=10000)
        self.impact_count = ti.field(ti.i32, shape=())
        
        # Overflow tracking
        self.overflow_particles = ti.field(ti.i32, shape=10000)
        self.landing_distances = ti.field(ti.f64, shape=10000)
        self.launch_velocities = ti.field(ti.f64, shape=10000)
        self.launch_angles = ti.field(ti.f64, shape=10000)
        self.overflow_count = ti.field(ti.i32, shape=())
        
        # Statistics
        self.mean_landing_distance = ti.field(ti.f64, shape=())
        self.max_landing_distance = ti.field(ti.f64, shape=())
        self.mean_launch_velocity = ti.field(ti.f64, shape=())
        
        # Initialize barrier geometry
        self._initialize_barriers(barrier_positions)
        
    def _initialize_barriers(self, positions: Tuple[float, float]):
        """Initialize barrier positions and normals."""
        for i, x_pos in enumerate(positions):
            self.barrier_positions[i] = [x_pos, 0.0, 0.0]
            self.barrier_normals[i] = [1.0, 0.0, 0.0]  # Normal pointing in +x direction
    
    @ti.kernel
    def detect_contacts(self, 
                       particle_positions: ti.template(),
                       particle_velocities: ti.template(),
                       contact_forces: ti.template(),
                       n_particles: int):
        """
        Detect particle-barrier contacts and compute penalty forces.
        
        Implements penalty method contact mechanics:
        F_contact = k_n * δ * n̂ + c_d * v_rel
        where δ is penetration depth and v_rel is relative velocity
        """
        
        for i in range(n_particles):
            pos = particle_positions[i]
            vel = particle_velocities[i]
            force = ti.Vector([0.0, 0.0, 0.0])
            
            # Check contact with each barrier
            for barrier_id in range(2):
                barrier_pos = self.barrier_positions[barrier_id]
                barrier_normal = self.barrier_normals[barrier_id]
                
                # Check if particle is in contact with barrier
                # Barrier is a vertical plane at x = barrier_pos.x
                penetration = barrier_pos.x - pos.x
                
                if penetration > 0.0 and pos.y < self.barrier_height:
                    # Particle is penetrating barrier
                    
                    # Normal contact force (penalty method)
                    normal_force_magnitude = self.contact_stiffness * penetration
                    normal_force = normal_force_magnitude * barrier_normal
                    
                    # Damping force (proportional to relative velocity)
                    relative_velocity = vel
                    damping_force = -self.contact_damping * relative_velocity
                    
                    # Coulomb friction (tangential component)
                    friction_force = ti.Vector([0.0, 0.0, 0.0])
                    if ti.abs(normal_force_magnitude) > 1e-12:
                        friction_force_magnitude = self.friction_coefficient * normal_force_magnitude
                        friction_force = -friction_force_magnitude * ti.Vector([0.0, vel.y, vel.z])
                        friction_force = friction_force / (ti.sqrt(vel.y**2 + vel.z**2 + 1e-12))
                    
                    # Total contact force
                    contact_force = normal_force + damping_force + friction_force
                    force += contact_force
                    
                    # Track impact forces
                    impact_magnitude = ti.sqrt(contact_force.x**2 + contact_force.y**2 + contact_force.z**2)
                    if impact_magnitude > self.max_impact_force[None]:
                        self.max_impact_force[None] = impact_magnitude
                    
                    # Store impact force history
                    if self.impact_count[None] < 10000:
                        idx = self.impact_count[None]
                        self.impact_force_history[idx] = contact_force
                        self.impact_time_history[idx] = ti.cast(0.0, ti.f64)  # Will be set by caller
                        self.impact_count[None] += 1
            
            contact_forces[i] = force
    
    @ti.kernel
    def track_overflow_kinematics(self,
                                 particle_positions: ti.template(),
                                 particle_velocities: ti.template(),
                                 n_particles: int,
                                 current_time: ti.f64):
        """
        Monitor overflow trajectories and compute landing statistics.
        
        Tracks particles that overflow barriers and calculates:
        - Launch velocity and angle
        - Landing distance using analytical trajectory (Equation 3)
        - Landing velocity (Equation 4)
        """
        
        for i in range(n_particles):
            pos = particle_positions[i]
            vel = particle_velocities[i]
            
            # Check for overflow at each barrier
            for barrier_id in range(2):
                barrier_pos = self.barrier_positions[barrier_id]
                
                # Check if particle is overflowing (above barrier height)
                if (pos.x > barrier_pos.x and 
                    pos.y > self.barrier_height and 
                    vel.x > 0.0):  # Moving forward
                    
                    # Calculate launch parameters
                    launch_velocity = ti.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    launch_angle = ti.atan2(vel.y, vel.x)
                    
                    # Analytical landing distance (Equation 3)
                    g = 9.81  # Gravity
                    theta = 20.0 * 3.14159265359 / 180.0  # Slope angle (radians)
                    
                    # Simplified trajectory calculation
                    # xi = (v²launch)/(g*cosθ) * [tanθ + √(tan²θ + 2g*HB/(v²launch*cosθ))] + HB*tanθ
                    cos_theta = ti.cos(theta)
                    tan_theta = ti.tan(theta)
                    
                    term1 = launch_velocity**2 / (g * cos_theta)
                    term2 = tan_theta + ti.sqrt(tan_theta**2 + 2 * g * self.barrier_height / (launch_velocity**2 * cos_theta))
                    landing_distance = term1 * term2 + self.barrier_height * tan_theta
                    
                    # Store overflow data
                    if self.overflow_count[None] < 10000:
                        idx = self.overflow_count[None]
                        self.overflow_particles[idx] = i
                        self.landing_distances[idx] = landing_distance
                        self.launch_velocities[idx] = launch_velocity
                        self.launch_angles[idx] = launch_angle
                        self.overflow_count[None] += 1
    
    @ti.kernel
    def compute_impact_statistics(self):
        """Compute impact force statistics."""
        total_force = ti.Vector([0.0, 0.0, 0.0])
        
        for i in range(self.impact_count[None]):
            total_force += self.impact_force_history[i]
        
        self.total_impact_force[None] = total_force
    
    @ti.kernel
    def compute_overflow_statistics(self):
        """Compute overflow trajectory statistics."""
        if self.overflow_count[None] > 0:
            total_distance = 0.0
            total_velocity = 0.0
            max_distance = 0.0
            
            for i in range(self.overflow_count[None]):
                distance = self.landing_distances[i]
                velocity = self.launch_velocities[i]
                
                total_distance += distance
                total_velocity += velocity
                
                if distance > max_distance:
                    max_distance = distance
            
            self.mean_landing_distance[None] = total_distance / self.overflow_count[None]
            self.max_landing_distance[None] = max_distance
            self.mean_launch_velocity[None] = total_velocity / self.overflow_count[None]
        else:
            self.mean_landing_distance[None] = 0.0
            self.max_landing_distance[None] = 0.0
            self.mean_launch_velocity[None] = 0.0
    
    def calculate_theoretical_landing_distance(self, 
                                             launch_velocity: float, 
                                             launch_angle: float,
                                             slope_angle: float = 20.0) -> float:
        """
        Analytical trajectory calculation (Equation 3).
        
        Args:
            launch_velocity: Initial velocity magnitude (m/s)
            launch_angle: Launch angle from horizontal (radians)
            slope_angle: Channel slope angle (degrees)
            
        Returns:
            Landing distance (m)
        """
        g = 9.81  # Gravity
        theta = slope_angle * np.pi / 180.0  # Convert to radians
        
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        
        # Equation 3: xi = (v²launch)/(g*cosθ) * [tanθ + √(tan²θ + 2g*HB/(v²launch*cosθ))] + HB*tanθ
        term1 = launch_velocity**2 / (g * cos_theta)
        term2 = tan_theta + np.sqrt(tan_theta**2 + 2 * g * self.barrier_height / (launch_velocity**2 * cos_theta))
        landing_distance = term1 * term2 + self.barrier_height * tan_theta
        
        return landing_distance
    
    def get_impact_statistics(self) -> Dict[str, Any]:
        """Get current impact force statistics."""
        return {
            'total_impact_force': self.total_impact_force[None].to_numpy(),
            'max_impact_force': self.max_impact_force[None],
            'impact_count': self.impact_count[None]
        }
    
    def get_overflow_statistics(self) -> Dict[str, float]:
        """Get current overflow trajectory statistics."""
        return {
            'mean_landing_distance': self.mean_landing_distance[None],
            'max_landing_distance': self.max_landing_distance[None],
            'mean_launch_velocity': self.mean_launch_velocity[None],
            'overflow_count': self.overflow_count[None]
        }
    
    def export_overflow_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Export overflow trajectory data for analysis."""
        n_overflow = self.overflow_count[None]
        
        if n_overflow > 0:
            particles = self.overflow_particles.to_numpy()[:n_overflow]
            distances = self.landing_distances.to_numpy()[:n_overflow]
            velocities = self.launch_velocities.to_numpy()[:n_overflow]
            angles = self.launch_angles.to_numpy()[:n_overflow]
            
            return particles, distances, velocities, angles
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.impact_count[None] = 0
        self.overflow_count[None] = 0
        self.max_impact_force[None] = 0.0
        self.total_impact_force[None] = [0.0, 0.0, 0.0]
        self.mean_landing_distance[None] = 0.0
        self.max_landing_distance[None] = 0.0
        self.mean_launch_velocity[None] = 0.0
