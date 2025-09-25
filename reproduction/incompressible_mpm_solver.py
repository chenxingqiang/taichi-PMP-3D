"""
Incompressible Material Point Method (iMPM) solver for free surface flow

This implementation reproduces the mathematical framework described in:
"Incompressible material point method for free surface flow"

Key features:
- Operator splitting method for incompressible flow
- Level set method for free surface tracking
- Pressure Poisson equation solver with PCG
- Surface tension modeling with curvature calculation
- Ghost Fluid Method for boundary conditions
- Hourglass control and mixed PIC/FLIP scheme

Authors: Implementation based on the referenced paper
"""

import taichi as ti
import numpy as np
import math
from level_set_method import LevelSetMethod
from pcg_solver import PCGSolver

# Initialize Taichi - moved to calling script for better control

@ti.data_oriented
class IncompressibleMPMSolver:
    def __init__(self,
                 nx, ny, nz,           # Grid dimensions
                 dx,                    # Grid spacing
                 rho=1000.0,           # Density (kg/m³)
                 mu=1.01e-3,           # Dynamic viscosity (Pa·s)
                 gamma=0.0,            # Surface tension coefficient (N/m)
                 g=9.81,               # Gravity magnitude (m/s²)
                 dt=1e-4,              # Time step
                 alpha_h=0.05,         # Hourglass damping coefficient
                 chi=0.03,             # FLIP blending coefficient
                 max_particles=100000  # Maximum number of particles
                 ):

        # Grid parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        self.inv_dx = 1.0 / dx

        # Physical parameters
        self.rho = rho
        self.mu = mu
        self.gamma = gamma
        self.g = ti.Vector([0.0, -g, 0.0])
        self.dt = dt
        self.alpha_h = alpha_h
        self.chi = chi

        # Particle parameters
        self.max_particles = max_particles
        self.n_particles = ti.field(dtype=int, shape=())

        # Particle fields
        self.x = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)      # Position
        self.v = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)      # Velocity
        self.v_star = ti.Vector.field(3, dtype=ti.f64, shape=max_particles) # Intermediate velocity
        self.m = ti.field(dtype=ti.f64, shape=max_particles)                # Mass
        self.V = ti.field(dtype=ti.f64, shape=max_particles)                # Volume
        self.C = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)   # APIC velocity matrix
        self.F = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)   # Deformation gradient
        self.phi_p = ti.field(dtype=ti.f64, shape=max_particles)            # Level set on particles

        # Grid fields (semi-staggered grid)
        # Velocity is stored at cell centers (semi-staggered)
        self.grid_v = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_v_star = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_v_old = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_m = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
        self.grid_f = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))
        self.div_v_star = ti.field(dtype=ti.f64, shape=(nx, ny, nz))       # Velocity divergence

        # Initialize level set method
        self.level_set_method = LevelSetMethod(nx, ny, nz, dx)

        # Initialize PCG solver
        self.pcg_solver = PCGSolver(nx, ny, nz, dx, rho, dt)
        self.pcg_solver.set_surface_tension(gamma)

        # Contact forces
        self.contact_forces = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)

        # Statistics
        self.total_kinetic_energy = ti.field(dtype=ti.f64, shape=())
        self.max_velocity = ti.field(dtype=ti.f64, shape=())

        print(f"Incompressible MPM Solver initialized:")
        print(f"  Grid: {nx} x {ny} x {nz}, dx = {dx}")
        print(f"  Density: {rho} kg/m³, Viscosity: {mu} Pa·s")
        print(f"  Surface tension: {gamma} N/m")
        print(f"  Max particles: {max_particles}")

    @ti.kernel
    def clear_grid(self):
        """Clear grid fields"""
        for i, j, k in self.grid_m:
            self.grid_m[i, j, k] = 0.0
            self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_v_star[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_f[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.div_v_star[i, j, k] = 0.0

    @ti.kernel
    def particle_to_grid(self):
        """Transfer particle data to grid (P2G)"""
        # Clear grid first
        for i, j, k in self.grid_m:
            self.grid_m[i, j, k] = 0.0
            self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_f[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

        # Transfer from particles to grid
        for p in range(self.n_particles[None]):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)

            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            # Stress calculation for viscous fluid
            # σ = -p*I + 2μ*D where D is strain rate tensor
            # For now, only consider viscous stress
            stress = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # Transfer to 3x3x3 neighboring grid points
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_pos = base + offset
                if 0 <= grid_pos[0] < self.nx and 0 <= grid_pos[1] < self.ny and 0 <= grid_pos[2] < self.nz:
                    # Mass and momentum transfer
                    self.grid_m[grid_pos] += weight * self.m[p]
                    self.grid_v[grid_pos] += weight * self.m[p] * (self.v[p] + self.C[p] @ dpos)

    @ti.kernel
    def normalize_grid_velocity(self):
        """Convert momentum to velocity on grid"""
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j, k] /= self.grid_m[i, j, k]
                self.grid_v_old[i, j, k] = self.grid_v[i, j, k]

    @ti.kernel
    def compute_intermediate_velocity(self):
        """Compute intermediate velocity ignoring pressure (Step 1 of operator splitting)"""
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0:
                # Apply gravity and viscous forces
                # v* = v^n + dt * (g + viscous_forces/ρ)
                force_per_mass = self.g

                # Add viscous forces using finite differences
                if 1 <= i < self.nx-1 and 1 <= j < self.ny-1 and 1 <= k < self.nz-1:
                    # Compute Laplacian of velocity for viscous forces
                    v_curr = self.grid_v[i, j, k]

                    # Second derivatives for viscous term: μ∇²v
                    d2v_dx2 = (self.grid_v[i+1, j, k] - 2*v_curr + self.grid_v[i-1, j, k]) / (self.dx*self.dx)
                    d2v_dy2 = (self.grid_v[i, j+1, k] - 2*v_curr + self.grid_v[i, j-1, k]) / (self.dx*self.dx)
                    d2v_dz2 = (self.grid_v[i, j, k+1] - 2*v_curr + self.grid_v[i, j, k-1]) / (self.dx*self.dx)

                    viscous_force = (self.mu / self.rho) * (d2v_dx2 + d2v_dy2 + d2v_dz2)
                    force_per_mass += viscous_force

                self.grid_v_star[i, j, k] = self.grid_v[i, j, k] + self.dt * force_per_mass

    @ti.kernel
    def compute_velocity_divergence(self):
        """Compute divergence of intermediate velocity field"""
        for i, j, k in self.div_v_star:
            if 1 <= i < self.nx-1 and 1 <= j < self.ny-1 and 1 <= k < self.nz-1:
                # Central difference for divergence
                div_x = (self.grid_v_star[i+1, j, k][0] - self.grid_v_star[i-1, j, k][0]) / (2.0 * self.dx)
                div_y = (self.grid_v_star[i, j+1, k][1] - self.grid_v_star[i, j-1, k][1]) / (2.0 * self.dx)
                div_z = (self.grid_v_star[i, j, k+1][2] - self.grid_v_star[i, j, k-1][2]) / (2.0 * self.dx)

                self.div_v_star[i, j, k] = div_x + div_y + div_z

    @ti.kernel
    def setup_pressure_system(self):
        """Setup the pressure Poisson equation: ∇²p = ρ/dt * ∇·v*"""
        for i, j, k in self.rhs:
            self.rhs[i, j, k] = (self.rho / self.dt) * self.div_v_star[i, j, k]

    @ti.kernel
    def apply_laplacian(self):
        """Apply 7-point Laplacian operator: Ap = ∇²p"""
        for i, j, k in self.Ap:
            self.Ap[i, j, k] = 0.0
            if 1 <= i < self.nx-1 and 1 <= j < self.ny-1 and 1 <= k < self.nz-1:
                # 7-point stencil for 3D Laplacian
                center = -6.0 * self.pressure[i, j, k]
                neighbors = (self.pressure[i+1, j, k] + self.pressure[i-1, j, k] +
                           self.pressure[i, j+1, k] + self.pressure[i, j-1, k] +
                           self.pressure[i, j, k+1] + self.pressure[i, j, k-1])

                self.Ap[i, j, k] = (center + neighbors) / (self.dx * self.dx)

    def solve_pressure_pcg(self, max_iter=100, tol=1e-6):
        """Solve pressure system using Preconditioned Conjugate Gradient"""
        # Update level set and classify cells
        self.pcg_solver.update_level_set(self.level_set_method.phi)
        self.pcg_solver.classify_cells(self.level_set_method.phi)
        self.pcg_solver.update_curvature(self.level_set_method.curvature)

        # Solve using PCG
        iterations = self.pcg_solver.solve_pcg(self.div_v_star, max_iter, tol)
        return iterations

    @ti.kernel
    def pressure_correction(self):
        """Apply pressure correction to get incompressible velocity field"""
        for i, j, k in self.grid_v:
            if self.grid_m[i, j, k] > 0:
                # Compute pressure gradient using weighted averaging (Eq. 25 in paper)
                if 1 <= i < self.nx-1 and 1 <= j < self.ny-1 and 1 <= k < self.nz-1:
                    # Compute pressure gradient directly from PCG solver pressure field
                    grad_p_x = (self.pcg_solver.pressure[i+1, j, k] - self.pcg_solver.pressure[i-1, j, k]) / (2.0 * self.dx)
                    grad_p_y = (self.pcg_solver.pressure[i, j+1, k] - self.pcg_solver.pressure[i, j-1, k]) / (2.0 * self.dx)
                    grad_p_z = (self.pcg_solver.pressure[i, j, k+1] - self.pcg_solver.pressure[i, j, k-1]) / (2.0 * self.dx)

                    grad_p = ti.Vector([grad_p_x, grad_p_y, grad_p_z])

                    # Apply weighted pressure gradient calculation near interfaces
                    # β weight based on level set function for stability
                    phi_curr = self.level_set_method.phi[i, j, k]
                    beta_weight = 1.0 if phi_curr < 0 else 0.1  # Reduce gradient in air regions

                    # v^{n+1} = v* - dt/ρ * β * ∇p
                    self.grid_v[i, j, k] = self.grid_v_star[i, j, k] - (self.dt / self.rho) * beta_weight * grad_p

    @ti.kernel
    def apply_boundary_conditions(self):
        """Apply boundary conditions and hourglass control"""
        # Apply domain boundary conditions (no-slip)
        for i, j, k in self.grid_v:
            if i == 0 or i == self.nx-1:
                self.grid_v[i, j, k][0] = 0.0
            if j == 0:  # Bottom boundary - no penetration
                self.grid_v[i, j, k][1] = 0.0
            if j == self.ny-1:  # Top boundary - free slip for dam break
                pass  # Keep velocity as is
            if k == 0 or k == self.nz-1:
                self.grid_v[i, j, k][2] = 0.0

        # Note: Hourglass control is applied separately after this kernel

    @ti.kernel
    def apply_hourglass_damping(self):
        """Apply hourglass damping to suppress spurious modes (Eq. 53 in paper)"""
        # Simplified hourglass control for semi-staggered grid
        for i, j, k in self.grid_v:
            if (self.grid_m[i, j, k] > 0 and 1 <= i < self.nx-1 and
                1 <= j < self.ny-1 and 1 <= k < self.nz-1):

                # Compute velocity differences for hourglass modes
                v_center = self.grid_v[i, j, k]
                v_neighbors = (self.grid_v[i+1, j, k] + self.grid_v[i-1, j, k] +
                              self.grid_v[i, j+1, k] + self.grid_v[i, j-1, k] +
                              self.grid_v[i, j, k+1] + self.grid_v[i, j, k-1]) / 6.0

                # Hourglass correction
                hourglass_correction = self.alpha_h * (v_neighbors - v_center)
                self.grid_v[i, j, k] += hourglass_correction

    @ti.kernel
    def grid_to_particle(self):
        """Transfer data from grid back to particles (G2P) with mixed PIC/FLIP"""
        for p in range(self.n_particles[None]):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)

            # Quadratic B-spline weights and gradients
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            dw = [(fx - 1.5) * self.inv_dx, (2 - 2*fx) * self.inv_dx, (fx - 0.5) * self.inv_dx]

            new_v_PIC = ti.Vector([0.0, 0.0, 0.0])
            new_v_FLIP = self.v[p]  # Start with old velocity
            new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # Interpolate velocity and compute affine velocity matrix
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_pos = base + offset
                if 0 <= grid_pos[0] < self.nx and 0 <= grid_pos[1] < self.ny and 0 <= grid_pos[2] < self.nz:
                    g_v = self.grid_v[grid_pos]
                    g_v_old = self.grid_v_old[grid_pos]

                    # PIC component
                    new_v_PIC += weight * g_v

                    # FLIP component (add velocity change)
                    new_v_FLIP += weight * (g_v - g_v_old)

                    # APIC affine velocity matrix
                    new_C += 4 * self.inv_dx**2 * weight * g_v.outer_product(dpos)

            # Mixed PIC/FLIP update (Eq. 54 in paper)
            self.v[p] = self.chi * new_v_PIC + (1.0 - self.chi) * new_v_FLIP
            self.C[p] = new_C

            # Update particle position using RK3 (simplified to Euler for now)
            self.x[p] += self.dt * self.v[p]

    @ti.kernel
    def initialize_particles_dam_break(self,
                                      x_min: ti.f64, x_max: ti.f64,
                                      y_min: ti.f64, y_max: ti.f64,
                                      z_min: ti.f64, z_max: ti.f64,
                                      ppc: int):
        """Initialize particles for dam break scenario"""
        self.n_particles[None] = 0

        # Calculate particle spacing
        particle_dx = self.dx / ti.sqrt(ppc)
        particle_volume = particle_dx**3
        particle_mass = self.rho * particle_volume

        # Generate particles in the specified region
        for i in range(int((x_max - x_min) / particle_dx)):
            for j in range(int((y_max - y_min) / particle_dx)):
                for k in range(int((z_max - z_min) / particle_dx)):
                    if self.n_particles[None] < self.max_particles:
                        pid = ti.atomic_add(self.n_particles[None], 1)

                        self.x[pid] = ti.Vector([
                            x_min + (i + 0.5) * particle_dx,
                            y_min + (j + 0.5) * particle_dx,
                            z_min + (k + 0.5) * particle_dx
                        ])

                        self.v[pid] = ti.Vector([0.0, 0.0, 0.0])
                        self.m[pid] = particle_mass
                        self.V[pid] = particle_volume
                        self.C[pid] = ti.Matrix.zero(ti.f64, 3, 3)
                        self.F[pid] = ti.Matrix.identity(ti.f64, 3)

    def step(self):
        """Perform one simulation step following iMPM algorithm"""
        # Step 1: Particle to Grid transfer (P2G)
        self.particle_to_grid()
        self.normalize_grid_velocity()

        # Step 2: Compute intermediate velocity v* (ignore pressure)
        self.compute_intermediate_velocity()

        # Step 3: Update level set function
        self.level_set_method.step(self.dt, self.grid_v_star)

        # Step 4: Compute velocity divergence ∇·v*
        self.compute_velocity_divergence()

        # Step 5: Solve pressure Poisson equation ∇²p = (ρ/Δt)∇·v*
        iterations = self.solve_pressure_pcg()

        # Step 6: Apply pressure correction v^{n+1} = v* - (Δt/ρ)∇p
        self.pressure_correction()

        # Step 7: Apply boundary conditions and hourglass control
        self.apply_boundary_conditions()
        self.apply_hourglass_damping()

        # Step 8: Grid to Particle transfer (G2P) with mixed PIC/FLIP
        self.grid_to_particle()

        # Reinitialize level set periodically
        if hasattr(self, 'step_count'):
            self.step_count += 1
            if self.step_count % 5 == 0:
                self.level_set_method.reinitialize()
        else:
            self.step_count = 1

        return iterations

    def export_particles_to_numpy(self):
        """Export particle data for visualization"""
        positions = self.x.to_numpy()[:self.n_particles[None]]
        velocities = self.v.to_numpy()[:self.n_particles[None]]
        return positions, velocities

    def export_vtk(self, filename: str):
        """Export particle data to VTK file for visualization"""
        positions, velocities = self.export_particles_to_numpy()

        if len(positions) == 0:
            print(f"Warning: No particles to export to {filename}")
            return

        # Create a simple VTK file
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("MPM Particle Data\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {len(positions)} float\n")

            for pos in positions:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

            f.write(f"POINT_DATA {len(positions)}\n")
            f.write("VECTORS velocity float\n")

            for vel in velocities:
                f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")

        print(f"Exported {len(positions)} particles to {filename}")

    @ti.kernel
    def compute_statistics(self):
        """Compute simulation statistics"""
        total_ke = 0.0
        max_vel = 0.0

        for p in range(self.n_particles[None]):
            vel_mag = self.v[p].norm()
            total_ke += 0.5 * self.m[p] * vel_mag**2
            max_vel = max(max_vel, vel_mag)

        self.total_kinetic_energy[None] = total_ke
        self.max_velocity[None] = max_vel

# Test function for basic functionality
def test_dam_break():
    """Test dam break scenario"""
    # Domain setup
    nx, ny, nz = 64, 32, 32
    dx = 0.02

    # Create solver
    solver = IncompressibleMPMSolver(nx, ny, nz, dx)

    # Initialize dam break
    solver.initialize_particles_dam_break(
        x_min=0.0, x_max=0.6,
        y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=0.6,
        ppc=4
    )

    print(f"Initialized {solver.n_particles[None]} particles")

    # Run simulation
    for frame in range(100):
        solver.step()
        solver.compute_statistics()

        if frame % 10 == 0:
            print(f"Frame {frame}: KE = {solver.total_kinetic_energy[None]:.6f}, Max vel = {solver.max_velocity[None]:.3f}")

if __name__ == "__main__":
    test_dam_break()