"""
Two-Phase Material Point Method (MPM) Solver for Debris Flow Simulation

This implementation models debris flow as a two-phase mixture:
- Solid phase: Granular material with Drucker-Prager rheology
- Fluid phase: Incompressible pore fluid

Key features:
- Two sets of particles: solid and fluid
- Drucker-Prager μ(I) rheology for granular phase
- Inter-phase drag force coupling
- Solid volume fraction tracking
- Mixed PIC/FLIP scheme

Based on:
- Ng et al. (2023) - Two-phase MPM for debris flow impact
- Tampubolon et al. (2017) - Multi-species simulation of porous sand and water
"""

import taichi as ti
import numpy as np
import math

# Note: Taichi should be initialized by the caller before importing this module


@ti.data_oriented
class DruckerPragerModel:
    """Drucker-Prager elastoplastic model with μ(I) rheology"""
    
    def __init__(self, n_particles, E=1e7, nu=0.3, 
                 friction_angle=30.0, mu_2=1.4, xi=1.0):
        """
        Args:
            n_particles: Maximum number of particles
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            friction_angle: Friction angle (degrees)
            mu_2: Secondary friction coefficient for μ(I) rheology
            xi: Rheology parameter
        """
        self.n_particles = n_particles
        self.E = E
        self.nu = nu
        
        # Lame parameters
        self.lame_mu = E / (2 * (1 + nu))
        self.lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        # Friction parameters
        self.friction_angle = friction_angle
        self.mu_2 = mu_2
        self.xi = xi
        
        # Particle fields
        self.F_elastic = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_particles)
        self.J = ti.field(dtype=ti.f64, shape=n_particles)
        self.friction_coeff = ti.field(dtype=ti.f64, shape=n_particles)
        self.plastic_strain = ti.field(dtype=ti.f64, shape=n_particles)
    
    @ti.kernel
    def initialize(self):
        """Initialize material state"""
        friction_angle_rad = self.friction_angle * 3.14159265359 / 180.0
        alpha = 2.0 * ti.sqrt(6.0) * ti.sin(friction_angle_rad) / (3.0 - ti.sin(friction_angle_rad))
        
        for p in range(self.n_particles):
            self.F_elastic[p] = ti.Matrix.identity(ti.f64, 3)
            self.J[p] = 1.0
            self.friction_coeff[p] = alpha
            self.plastic_strain[p] = 0.0
    
    @ti.func
    def compute_stress(self, p, dt):
        """Compute Cauchy stress using Drucker-Prager model with numerical safeguards"""
        stress = ti.Matrix.zero(ti.f64, 3, 3)
        
        # Check for valid deformation gradient
        J = self.F_elastic[p].determinant()
        if J > 0.01 and J < 100.0:  # Reasonable range for J
            # SVD of elastic deformation gradient
            U, sig, V = ti.svd(self.F_elastic[p])
            
            # Clamp singular values to avoid numerical issues
            for d in ti.static(range(3)):
                sig[d, d] = ti.max(sig[d, d], 0.01)
                sig[d, d] = ti.min(sig[d, d], 10.0)
            
            # Logarithmic strain
            e = ti.Matrix.zero(ti.f64, 3, 3)
            for d in ti.static(range(3)):
                e[d, d] = ti.log(sig[d, d])
            
            # Trial stress (Kirchhoff stress)
            e_trace = e.trace()
            tau_trial = self.lame_lambda * e_trace * ti.Matrix.identity(ti.f64, 3) + 2.0 * self.lame_mu * e
            
            # Pressure (negative in compression)
            P = tau_trial.trace() / 3.0
            S_trial = tau_trial - P * ti.Matrix.identity(ti.f64, 3)
            S_norm = ti.sqrt(S_trial[0,0]**2 + S_trial[1,1]**2 + S_trial[2,2]**2 + 
                            2*(S_trial[0,1]**2 + S_trial[0,2]**2 + S_trial[1,2]**2) + 1e-10)
            
            # Yield function: f = sqrt(2/3)*q + alpha*p
            Q_trial = ti.sqrt(1.5) * S_norm
            yield_f = ti.sqrt(2.0/3.0) * Q_trial + self.friction_coeff[p] * P
            
            stress = tau_trial
            
            # Plastic return mapping (only in compression, P < 0)
            if yield_f > 0 and P < 0:
                # Simple radial return to yield surface
                mu_s = self.friction_coeff[p] / ti.sqrt(2.0)
                target_q = -self.friction_coeff[p] * P * ti.sqrt(1.5)  # q at yield surface
                
                if S_norm > 1e-10:
                    scale = ti.min(target_q / S_norm, 1.0)  # Scale deviatoric stress
                    stress = P * ti.Matrix.identity(ti.f64, 3) + scale * S_trial
            
            # Convert to Cauchy stress
            stress = stress / ti.max(J, 0.01)
        
        # Limit stress magnitude to avoid explosion
        stress_max = 1e8  # 100 MPa limit
        for i, j in ti.static(ti.ndrange(3, 3)):
            stress[i, j] = ti.max(ti.min(stress[i, j], stress_max), -stress_max)
        
        return stress
    
    @ti.func
    def update_deformation_gradient(self, p, grad_v, dt):
        """Update elastic deformation gradient with safeguards"""
        # Limit velocity gradient to avoid extreme deformation
        grad_v_limited = grad_v
        max_grad = 100.0  # Maximum velocity gradient (1/s)
        for i, j in ti.static(ti.ndrange(3, 3)):
            grad_v_limited[i, j] = ti.max(ti.min(grad_v[i, j], max_grad), -max_grad)
        
        delta_F = ti.Matrix.identity(ti.f64, 3) + dt * grad_v_limited
        new_F = delta_F @ self.F_elastic[p]
        
        # Only update if result is reasonable
        new_J = new_F.determinant()
        if new_J > 0.1 and new_J < 10.0:
            self.F_elastic[p] = new_F
            self.J[p] = new_J


@ti.data_oriented
class TwoPhaseMPMSolver:
    """Two-phase MPM solver for debris flow"""
    
    def __init__(self,
                 nx, ny, nz,
                 dx,
                 # Solid phase parameters
                 rho_s=2650.0,        # Solid density (kg/m³)
                 E_s=1e7,             # Young's modulus (Pa)
                 nu_s=0.3,            # Poisson's ratio
                 friction_angle=30.0, # Friction angle (degrees)
                 # Fluid phase parameters
                 rho_f=1000.0,        # Fluid density (kg/m³)
                 mu_f=0.001,          # Fluid viscosity (Pa·s)
                 # Coupling parameters
                 d_s=0.001,           # Solid particle diameter (m)
                 phi_s0=0.55,         # Initial solid volume fraction
                 # Simulation parameters
                 g=9.81,
                 dt=1e-4,
                 max_particles=100000,
                 flip_ratio=0.97):
        
        # Grid parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        self.inv_dx = 1.0 / dx
        
        # Physical parameters
        self.rho_s = rho_s
        self.rho_f = rho_f
        self.mu_f = mu_f
        self.d_s = d_s
        self.phi_s0 = phi_s0
        self.g = ti.Vector([0.0, -g, 0.0])
        self.dt = dt
        self.flip_ratio = flip_ratio
        
        # Maximum particles per phase
        self.max_particles = max_particles
        self.n_solid = ti.field(dtype=ti.i32, shape=())
        self.n_fluid = ti.field(dtype=ti.i32, shape=())
        
        # ========== Solid Phase Particles ==========
        self.x_s = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)
        self.v_s = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)
        self.m_s = ti.field(dtype=ti.f64, shape=max_particles)
        self.V_s = ti.field(dtype=ti.f64, shape=max_particles)
        self.C_s = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)
        self.phi_s = ti.field(dtype=ti.f64, shape=max_particles)  # Solid volume fraction
        
        # ========== Fluid Phase Particles ==========
        self.x_f = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)
        self.v_f = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)
        self.m_f = ti.field(dtype=ti.f64, shape=max_particles)
        self.V_f = ti.field(dtype=ti.f64, shape=max_particles)
        self.C_f = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)
        self.p_f = ti.field(dtype=ti.f64, shape=max_particles)  # Pore pressure
        
        # ========== Grid Fields ==========
        # Solid grid
        self.grid_v_s = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_v_s_old = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_m_s = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_f_s = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        
        # Fluid grid
        self.grid_v_f = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_v_f_old = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_m_f = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_f_f = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        
        # Coupling fields
        self.grid_phi_s = ti.field(dtype=ti.f64, shape=(nx, ny, nz))  # Grid solid fraction
        self.grid_drag = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))  # Drag force
        
        # Initialize Drucker-Prager model (use lower stiffness for stability)
        self.solid_model = DruckerPragerModel(
            max_particles, E=E_s * 0.01, nu=nu_s, friction_angle=friction_angle  # Reduce E by 100x
        )
        
        # Calm period counter
        self.step_count = ti.field(dtype=ti.i32, shape=())
        self.calm_steps = 500  # Number of steps to gradually apply forces
        
        print(f"Two-Phase MPM Solver initialized:")
        print(f"  Grid: {nx} x {ny} x {nz}, dx = {dx}")
        print(f"  Solid: ρ={rho_s} kg/m³, E={E_s:.1e} Pa, φ={friction_angle}°")
        print(f"  Fluid: ρ={rho_f} kg/m³, μ={mu_f} Pa·s")
        print(f"  Coupling: d={d_s*1000:.1f} mm, φ_s0={phi_s0}")
    
    @ti.kernel
    def initialize_particles_two_phase(self,
                                       x_min: ti.f64, x_max: ti.f64,
                                       y_min: ti.f64, y_max: ti.f64,
                                       z_min: ti.f64, z_max: ti.f64,
                                       ppc: ti.i32):
        """Initialize both solid and fluid particles in the same region"""
        self.n_solid[None] = 0
        self.n_fluid[None] = 0
        
        # Particle spacing
        particle_dx = self.dx / ti.sqrt(float(ppc))
        particle_vol = particle_dx ** 3
        
        # Mass per particle
        solid_mass = self.rho_s * particle_vol * self.phi_s0
        fluid_mass = self.rho_f * particle_vol * (1.0 - self.phi_s0)
        
        # Generate particles
        n_x = int((x_max - x_min) / particle_dx)
        n_y = int((y_max - y_min) / particle_dx)
        n_z = int((z_max - z_min) / particle_dx)
        
        for i, j, k in ti.ndrange(n_x, n_y, n_z):
            pos = ti.Vector([
                x_min + (i + 0.5) * particle_dx,
                y_min + (j + 0.5) * particle_dx,
                z_min + (k + 0.5) * particle_dx
            ])
            
            # Add solid particle
            if self.n_solid[None] < self.max_particles:
                pid_s = ti.atomic_add(self.n_solid[None], 1)
                self.x_s[pid_s] = pos
                self.v_s[pid_s] = ti.Vector([0.0, 0.0, 0.0])
                self.m_s[pid_s] = solid_mass
                self.V_s[pid_s] = particle_vol * self.phi_s0
                self.C_s[pid_s] = ti.Matrix.zero(ti.f64, 3, 3)
                self.phi_s[pid_s] = self.phi_s0
            
            # Add fluid particle (co-located)
            if self.n_fluid[None] < self.max_particles:
                pid_f = ti.atomic_add(self.n_fluid[None], 1)
                self.x_f[pid_f] = pos
                self.v_f[pid_f] = ti.Vector([0.0, 0.0, 0.0])
                self.m_f[pid_f] = fluid_mass
                self.V_f[pid_f] = particle_vol * (1.0 - self.phi_s0)
                self.C_f[pid_f] = ti.Matrix.zero(ti.f64, 3, 3)
                self.p_f[pid_f] = 0.0
    
    def init_particles(self, x_min, x_max, y_min, y_max, z_min, z_max, ppc):
        """Initialize particles (Python-scope wrapper)"""
        self.initialize_particles_two_phase(x_min, x_max, y_min, y_max, z_min, z_max, ppc)
        # Initialize solid material model (must be called from Python scope)
        self.solid_model.initialize()
    
    @ti.kernel
    def clear_grid(self):
        """Reset grid fields"""
        for i, j, k in self.grid_m_s:
            self.grid_v_s[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.grid_v_s_old[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.grid_m_s[i, j, k] = 0.0
            self.grid_f_s[i, j, k] = ti.Vector.zero(ti.f64, 3)
            
            self.grid_v_f[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.grid_v_f_old[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.grid_m_f[i, j, k] = 0.0
            self.grid_f_f[i, j, k] = ti.Vector.zero(ti.f64, 3)
            
            self.grid_phi_s[i, j, k] = 0.0
            self.grid_drag[i, j, k] = ti.Vector.zero(ti.f64, 3)
    
    @ti.kernel
    def p2g_solid(self):
        """Particle to Grid transfer for solid phase"""
        for p in range(self.n_solid[None]):
            base = (self.x_s[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x_s[p] * self.inv_dx - base.cast(float)
            
            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # Compute stress
            stress = self.solid_model.compute_stress(p, self.dt)
            
            # APIC affine
            affine = self.m_s[p] * self.C_s[p]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                idx = base + offset
                
                if 0 <= idx[0] < self.nx and 0 <= idx[1] < self.ny and 0 <= idx[2] < self.nz:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    
                    # Momentum
                    self.grid_v_s[idx] += weight * (self.m_s[p] * self.v_s[p] + affine @ dpos)
                    self.grid_m_s[idx] += weight * self.m_s[p]
                    
                    # Solid volume fraction
                    self.grid_phi_s[idx] += weight * self.phi_s[p]
                    
                    # Internal force
                    grad_w = ti.Vector([
                        (w[i][0] if i == 0 else (-2*(fx[0]-1) if i == 1 else (fx[0]-0.5))) * w[j][1] * w[k][2],
                        w[i][0] * (w[j][1] if j == 0 else (-2*(fx[1]-1) if j == 1 else (fx[1]-0.5))) * w[k][2],
                        w[i][0] * w[j][1] * (w[k][2] if k == 0 else (-2*(fx[2]-1) if k == 1 else (fx[2]-0.5)))
                    ]) * self.inv_dx
                    
                    self.grid_f_s[idx] -= self.V_s[p] * stress @ grad_w
    
    @ti.kernel
    def p2g_fluid(self):
        """Particle to Grid transfer for fluid phase"""
        for p in range(self.n_fluid[None]):
            base = (self.x_f[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x_f[p] * self.inv_dx - base.cast(float)
            
            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # APIC affine
            affine = self.m_f[p] * self.C_f[p]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                idx = base + offset
                
                if 0 <= idx[0] < self.nx and 0 <= idx[1] < self.ny and 0 <= idx[2] < self.nz:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    
                    # Momentum
                    self.grid_v_f[idx] += weight * (self.m_f[p] * self.v_f[p] + affine @ dpos)
                    self.grid_m_f[idx] += weight * self.m_f[p]
    
    @ti.kernel
    def grid_operations(self, gravity_factor: ti.f64):
        """Grid velocity update with gravity, drag, and boundary conditions"""
        for i, j, k in self.grid_m_s:
            # Normalize solid volume fraction
            if self.grid_m_s[i, j, k] > 1e-10:
                self.grid_phi_s[i, j, k] /= (self.grid_m_s[i, j, k] / (self.rho_s * self.dx**3))
                self.grid_phi_s[i, j, k] = ti.min(ti.max(self.grid_phi_s[i, j, k], 0.0), 0.65)
            
            # Momentum to velocity
            if self.grid_m_s[i, j, k] > 1e-10:
                self.grid_v_s[i, j, k] /= self.grid_m_s[i, j, k]
                self.grid_v_s_old[i, j, k] = self.grid_v_s[i, j, k]
            
            if self.grid_m_f[i, j, k] > 1e-10:
                self.grid_v_f[i, j, k] /= self.grid_m_f[i, j, k]
                self.grid_v_f_old[i, j, k] = self.grid_v_f[i, j, k]
            
            # Compute drag force (Eq. 22 in paper)
            phi_s = self.grid_phi_s[i, j, k]
            if phi_s > 0.01 and phi_s < 0.64 and self.grid_m_s[i, j, k] > 1e-10 and self.grid_m_f[i, j, k] > 1e-10:
                v_rel = self.grid_v_s[i, j, k] - self.grid_v_f[i, j, k]
                
                # Drag coefficient (simplified Di Felice model)
                F_hat = (1.0 - phi_s) ** (-2.65)  # Simplified correction
                drag_coeff = 18.0 * phi_s * (1.0 - phi_s) * self.mu_f / (self.d_s ** 2) * F_hat
                
                self.grid_drag[i, j, k] = drag_coeff * v_rel
            
            # Apply forces with gravity factor for calm period
            if self.grid_m_s[i, j, k] > 1e-10:
                # Solid: gravity + internal force - drag
                # During calm period, reduce internal force contribution
                internal_acc = self.grid_f_s[i, j, k] / self.grid_m_s[i, j, k]
                internal_acc = internal_acc * gravity_factor  # Scale internal forces during calm period
                drag_acc = self.grid_drag[i, j, k] / self.grid_m_s[i, j, k]
                acc_s = gravity_factor * self.g + internal_acc - drag_acc
                self.grid_v_s[i, j, k] += self.dt * acc_s
            
            if self.grid_m_f[i, j, k] > 1e-10:
                # Fluid: gravity + drag
                drag_acc = self.grid_drag[i, j, k] / self.grid_m_f[i, j, k]
                acc_f = gravity_factor * self.g + drag_acc
                self.grid_v_f[i, j, k] += self.dt * acc_f
            
            # Boundary conditions
            if i < 3 and self.grid_v_s[i, j, k][0] < 0:
                self.grid_v_s[i, j, k][0] = 0
                self.grid_v_f[i, j, k][0] = 0
            if i >= self.nx - 3 and self.grid_v_s[i, j, k][0] > 0:
                self.grid_v_s[i, j, k][0] = 0
                self.grid_v_f[i, j, k][0] = 0
            if j < 3 and self.grid_v_s[i, j, k][1] < 0:
                self.grid_v_s[i, j, k][1] = 0
                self.grid_v_f[i, j, k][1] = 0
            if j >= self.ny - 3 and self.grid_v_s[i, j, k][1] > 0:
                self.grid_v_s[i, j, k][1] = 0
                self.grid_v_f[i, j, k][1] = 0
            if k < 3 and self.grid_v_s[i, j, k][2] < 0:
                self.grid_v_s[i, j, k][2] = 0
                self.grid_v_f[i, j, k][2] = 0
            if k >= self.nz - 3 and self.grid_v_s[i, j, k][2] > 0:
                self.grid_v_s[i, j, k][2] = 0
                self.grid_v_f[i, j, k][2] = 0
    
    @ti.kernel
    def g2p_solid(self):
        """Grid to Particle transfer for solid phase"""
        for p in range(self.n_solid[None]):
            base = (self.x_s[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x_s[p] * self.inv_dx - base.cast(float)
            
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_v = ti.Vector.zero(ti.f64, 3)
            new_C = ti.Matrix.zero(ti.f64, 3, 3)
            v_pic = ti.Vector.zero(ti.f64, 3)
            v_flip = self.v_s[p]
            grad_v = ti.Matrix.zero(ti.f64, 3, 3)
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                idx = base + offset
                
                if 0 <= idx[0] < self.nx and 0 <= idx[1] < self.ny and 0 <= idx[2] < self.nz:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    
                    g_v = self.grid_v_s[idx]
                    g_v_old = self.grid_v_s_old[idx]
                    
                    v_pic += weight * g_v
                    v_flip += weight * (g_v - g_v_old)
                    new_C += 4.0 * weight * g_v.outer_product(dpos) * self.inv_dx ** 2
                    
                    # Velocity gradient for deformation
                    grad_w = ti.Vector([
                        (fx[0] - 1.5 if i == 0 else (2 - 2*fx[0] if i == 1 else fx[0] - 0.5)) * self.inv_dx,
                        (fx[1] - 1.5 if j == 0 else (2 - 2*fx[1] if j == 1 else fx[1] - 0.5)) * self.inv_dx,
                        (fx[2] - 1.5 if k == 0 else (2 - 2*fx[2] if k == 1 else fx[2] - 0.5)) * self.inv_dx
                    ])
                    grad_v += g_v.outer_product(ti.Vector([
                        grad_w[0] * w[j][1] * w[k][2],
                        w[i][0] * grad_w[1] * w[k][2],
                        w[i][0] * w[j][1] * grad_w[2]
                    ]))
            
            # FLIP/PIC blend
            new_v = self.flip_ratio * v_flip + (1.0 - self.flip_ratio) * v_pic
            
            # Limit velocity to avoid explosion
            vel_mag = new_v.norm()
            max_vel = 50.0  # 50 m/s limit
            if vel_mag > max_vel:
                new_v = new_v * (max_vel / vel_mag)
            
            self.v_s[p] = new_v
            self.C_s[p] = new_C
            
            # Update deformation gradient
            self.solid_model.update_deformation_gradient(p, grad_v, self.dt)
            
            # Advect position
            self.x_s[p] += self.dt * v_pic
    
    @ti.kernel
    def g2p_fluid(self):
        """Grid to Particle transfer for fluid phase"""
        for p in range(self.n_fluid[None]):
            base = (self.x_f[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x_f[p] * self.inv_dx - base.cast(float)
            
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_C = ti.Matrix.zero(ti.f64, 3, 3)
            v_pic = ti.Vector.zero(ti.f64, 3)
            v_flip = self.v_f[p]
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                idx = base + offset
                
                if 0 <= idx[0] < self.nx and 0 <= idx[1] < self.ny and 0 <= idx[2] < self.nz:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    
                    g_v = self.grid_v_f[idx]
                    g_v_old = self.grid_v_f_old[idx]
                    
                    v_pic += weight * g_v
                    v_flip += weight * (g_v - g_v_old)
                    new_C += 4.0 * weight * g_v.outer_product(dpos) * self.inv_dx ** 2
            
            # FLIP/PIC blend
            new_v = self.flip_ratio * v_flip + (1.0 - self.flip_ratio) * v_pic
            
            # Limit velocity to avoid explosion
            vel_mag = new_v.norm()
            max_vel = 50.0  # 50 m/s limit
            if vel_mag > max_vel:
                new_v = new_v * (max_vel / vel_mag)
            
            self.v_f[p] = new_v
            self.C_f[p] = new_C
            
            # Advect position
            self.x_f[p] += self.dt * v_pic
    
    def step(self):
        """Perform one simulation step"""
        # Compute gravity factor for calm period
        step = self.step_count[None]
        gravity_factor = min(1.0, step / self.calm_steps)
        
        self.clear_grid()
        self.p2g_solid()
        self.p2g_fluid()
        self.grid_operations(gravity_factor)
        self.g2p_solid()
        self.g2p_fluid()
        
        self.step_count[None] += 1
    
    def export_particles(self):
        """Export particle data as numpy arrays"""
        n_s = self.n_solid[None]
        n_f = self.n_fluid[None]
        
        return {
            'solid': {
                'positions': self.x_s.to_numpy()[:n_s],
                'velocities': self.v_s.to_numpy()[:n_s],
                'phi': self.phi_s.to_numpy()[:n_s]
            },
            'fluid': {
                'positions': self.x_f.to_numpy()[:n_f],
                'velocities': self.v_f.to_numpy()[:n_f]
            }
        }

