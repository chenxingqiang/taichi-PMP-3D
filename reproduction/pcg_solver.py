"""
Preconditioned Conjugate Gradient (PCG) Solver for Pressure Poisson Equation

This module implements the PCG solver for the pressure system in iMPM:
∇²p^{n+1} = (ρ/Δt) ∇·v*

Key features:
- 7-point finite difference Laplacian in 3D
- Ghost Fluid Method (GFM) for free surface boundary conditions
- Jacobi preconditioner for improved convergence
- Support for Neumann boundary conditions at solid walls
- Semi-staggered grid layout (pressure at cell centers)

Mathematical framework:
- Linear system: Ap = b where A is the discrete Laplacian
- Ghost cells: p^G = (p^fs + (θ-1)p^f)/θ for free surface BCs
- Solid wall BCs: ∇p·n = 0 (no penetration condition)
"""

import taichi as ti
import numpy as np

@ti.data_oriented
class PCGSolver:
    def __init__(self, nx, ny, nz, dx, rho, dt):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        self.rho = rho
        self.dt = dt
        self.inv_dx2 = 1.0 / (dx * dx)
        
        # Pressure and solver fields
        self.pressure = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
        self.rhs = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
        
        # PCG solver fields
        self.r = ti.field(dtype=ti.f64, shape=(nx, ny, nz))      # Residual
        self.z = ti.field(dtype=ti.f64, shape=(nx, ny, nz))      # Preconditioned residual
        self.p = ti.field(dtype=ti.f64, shape=(nx, ny, nz))      # Search direction
        self.Ap = ti.field(dtype=ti.f64, shape=(nx, ny, nz))     # A times p
        
        # Boundary condition fields
        self.cell_type = ti.field(dtype=int, shape=(nx, ny, nz))  # 0: fluid, 1: solid, 2: air
        self.level_set = ti.field(dtype=ti.f64, shape=(nx+1, ny+1, nz+1))  # Level set function
        
        # Surface tension parameters
        self.surface_tension = ti.field(dtype=ti.f64, shape=())
        self.curvature = ti.field(dtype=ti.f64, shape=(nx+1, ny+1, nz+1))
        
        # Scalar reduction fields
        self.dot_result = ti.field(dtype=ti.f64, shape=())
        self.alpha_denom = ti.field(dtype=ti.f64, shape=())
        
        # Ghost pressure values
        self.p_air = 0.0  # Atmospheric pressure (gauge pressure = 0)
        
        print(f"PCG Solver initialized for {nx}x{ny}x{nz} grid")
    
    @ti.kernel
    def setup_rhs(self, div_v_star: ti.template()):
        """Setup right-hand side: b = (ρ/Δt) ∇·v*"""
        for i, j, k in self.rhs:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                self.rhs[i, j, k] = (self.rho / self.dt) * div_v_star[i, j, k]
            else:
                self.rhs[i, j, k] = 0.0
    
    @ti.func
    def get_ghost_pressure(self, i, j, k, direction):
        """Compute ghost pressure using Ghost Fluid Method"""
        # direction: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
        ghost_pressure = 0.0
        
        # Check if we're at a free surface boundary
        if direction == 0 and i < self.nx-1:  # +x direction
            if self.cell_type[i+1, j, k] == 2:  # Air cell
                # Compute θ - normalized distance from interface to fluid cell center
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i+1, j, k]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    # Surface pressure with curvature
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i+1, j, k]
        
        elif direction == 1 and i > 0:  # -x direction
            if self.cell_type[i-1, j, k] == 2:  # Air cell
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i-1, j, k]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i-1, j, k]
        
        # Similar logic for y and z directions
        elif direction == 2 and j < self.ny-1:  # +y direction
            if self.cell_type[i, j+1, k] == 2:
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i, j+1, k]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i, j+1, k]
                
        elif direction == 3 and j > 0:  # -y direction
            if self.cell_type[i, j-1, k] == 2:
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i, j-1, k]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i, j-1, k]
        
        elif direction == 4 and k < self.nz-1:  # +z direction
            if self.cell_type[i, j, k+1] == 2:
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i, j, k+1]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i, j, k+1]
                
        elif direction == 5 and k > 0:  # -z direction
            if self.cell_type[i, j, k-1] == 2:
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i, j, k-1]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i, j, k-1]
        
        return ghost_pressure
    
    @ti.kernel
    def apply_laplacian(self, input_field: ti.template(), output_field: ti.template()):
        """Apply 7-point Laplacian with Ghost Fluid Method"""
        for i, j, k in output_field:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                # Get neighboring pressure values (including ghost values)
                p_xp = self.get_ghost_pressure(i, j, k, 0) if i == self.nx-1 or self.cell_type[i+1, j, k] != 0 else input_field[i+1, j, k]
                p_xm = self.get_ghost_pressure(i, j, k, 1) if i == 0 or self.cell_type[i-1, j, k] != 0 else input_field[i-1, j, k]
                p_yp = self.get_ghost_pressure(i, j, k, 2) if j == self.ny-1 or self.cell_type[i, j+1, k] != 0 else input_field[i, j+1, k]
                p_ym = self.get_ghost_pressure(i, j, k, 3) if j == 0 or self.cell_type[i, j-1, k] != 0 else input_field[i, j-1, k]
                p_zp = self.get_ghost_pressure(i, j, k, 4) if k == self.nz-1 or self.cell_type[i, j, k+1] != 0 else input_field[i, j, k+1]
                p_zm = self.get_ghost_pressure(i, j, k, 5) if k == 0 or self.cell_type[i, j, k-1] != 0 else input_field[i, j, k-1]
                
                # 7-point Laplacian stencil
                center = -6.0 * input_field[i, j, k]
                neighbors = p_xp + p_xm + p_yp + p_ym + p_zp + p_zm
                
                output_field[i, j, k] = self.inv_dx2 * (center + neighbors)
            else:
                output_field[i, j, k] = 0.0
    
    @ti.kernel
    def apply_preconditioner(self, input_field: ti.template(), output_field: ti.template()):
        """Apply Jacobi preconditioner: z = M^{-1} * r"""
        for i, j, k in output_field:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                # Jacobi preconditioner: diagonal scaling
                # For 7-point Laplacian, diagonal entry is -6/dx²
                diagonal = -6.0 * self.inv_dx2
                if abs(diagonal) > 1e-12:
                    output_field[i, j, k] = input_field[i, j, k] / diagonal
                else:
                    output_field[i, j, k] = input_field[i, j, k]
            else:
                output_field[i, j, k] = 0.0
    
    @ti.kernel
    def compute_dot_product(self, field_a: ti.template(), field_b: ti.template()) -> ti.f64:
        """Compute dot product of two fields over fluid cells only"""
        result = 0.0
        for i, j, k in field_a:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                result += field_a[i, j, k] * field_b[i, j, k]
        return result
    
    @ti.kernel
    def vector_axpy(self, alpha: ti.f64, x: ti.template(), y: ti.template()):
        """Compute y = alpha * x + y for fluid cells only"""
        for i, j, k in y:
            if self.cell_type[i, j, k] == 0:
                y[i, j, k] += alpha * x[i, j, k]
    
    @ti.kernel
    def vector_copy(self, src: ti.template(), dst: ti.template()):
        """Copy src to dst for fluid cells only"""
        for i, j, k in dst:
            if self.cell_type[i, j, k] == 0:
                dst[i, j, k] = src[i, j, k]
            else:
                dst[i, j, k] = 0.0
    
    @ti.kernel
    def vector_scale(self, alpha: ti.f64, x: ti.template()):
        """Scale vector x by alpha for fluid cells only"""
        for i, j, k in x:
            if self.cell_type[i, j, k] == 0:
                x[i, j, k] *= alpha
    
    @ti.kernel
    def clear_field(self, field: ti.template()):
        """Clear field"""
        for i, j, k in field:
            field[i, j, k] = 0.0
    
    @ti.kernel
    def compute_initial_residual(self):
        """Compute initial residual: r = b - Ap"""
        for i, j, k in self.r:
            if self.cell_type[i, j, k] == 0:
                self.r[i, j, k] = self.rhs[i, j, k] - self.Ap[i, j, k]
            else:
                self.r[i, j, k] = 0.0
    
    def solve_pcg(self, div_v_star, max_iter=100, tol=1e-6):
        """Solve pressure Poisson equation using PCG"""
        # Setup RHS
        self.setup_rhs(div_v_star)
        
        # Initial guess (use previous pressure or zero)
        # self.clear_field(self.pressure)
        
        # Compute initial residual: r = b - Ap
        self.apply_laplacian(self.pressure, self.Ap)
        self.compute_initial_residual()
        
        # Apply preconditioner: z = M^{-1} * r
        self.apply_preconditioner(self.r, self.z)
        
        # Initial search direction: p = z
        self.vector_copy(self.z, self.p)
        
        # Initial dot product: rz_old = r · z
        rz_old = self.compute_dot_product(self.r, self.z)
        
        # Check for immediate convergence
        initial_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        if initial_residual < tol:
            print(f"PCG converged immediately, residual = {initial_residual:.2e}")
            return 0
        
        # PCG iteration
        for iteration in range(max_iter):
            # Compute Ap
            self.apply_laplacian(self.p, self.Ap)
            
            # Compute alpha = rz_old / (p · Ap)
            pAp = self.compute_dot_product(self.p, self.Ap)
            if abs(pAp) < 1e-14:
                print(f"PCG breakdown: pAp = {pAp}")
                break
            
            alpha = rz_old / pAp
            
            # Update solution: x = x + alpha * p
            self.vector_axpy(alpha, self.p, self.pressure)
            
            # Update residual: r = r - alpha * Ap
            self.vector_axpy(-alpha, self.Ap, self.r)
            
            # Check convergence
            residual_norm = ti.sqrt(self.compute_dot_product(self.r, self.r))
            if residual_norm < tol:
                print(f"PCG converged in {iteration+1} iterations, residual = {residual_norm:.2e}")
                return iteration + 1
            
            # Apply preconditioner: z = M^{-1} * r
            self.apply_preconditioner(self.r, self.z)
            
            # Compute beta = rz_new / rz_old
            rz_new = self.compute_dot_product(self.r, self.z)
            if abs(rz_old) < 1e-14:
                print(f"PCG breakdown: rz_old = {rz_old}")
                break
            
            beta = rz_new / rz_old
            
            # Update search direction: p = z + beta * p
            self.vector_scale(beta, self.p)
            self.vector_axpy(1.0, self.z, self.p)
            
            # Update rz_old for next iteration
            rz_old = rz_new
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"  PCG iteration {iteration+1}: residual = {residual_norm:.2e}")
        
        final_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        print(f"PCG did not converge in {max_iter} iterations, final residual = {final_residual:.2e}")
        return max_iter
    
    @ti.kernel
    def classify_cells(self, phi: ti.template()):
        """Classify cells based on level set function"""
        for i, j, k in self.cell_type:
            # Sample level set at cell center
            phi_center = 0.125 * (
                phi[i, j, k] + phi[i+1, j, k] + phi[i, j+1, k] + phi[i+1, j+1, k] +
                phi[i, j, k+1] + phi[i+1, j, k+1] + phi[i, j+1, k+1] + phi[i+1, j+1, k+1]
            )
            
            if phi_center < 0:
                self.cell_type[i, j, k] = 0  # Fluid
            else:
                self.cell_type[i, j, k] = 2  # Air
    
    @ti.kernel
    def update_level_set(self, phi: ti.template()):
        """Update level set function from external source"""
        for i, j, k in self.level_set:
            if i < phi.shape[0] and j < phi.shape[1] and k < phi.shape[2]:
                self.level_set[i, j, k] = phi[i, j, k]
    
    @ti.kernel
    def update_curvature(self, kappa: ti.template()):
        """Update curvature field from external source"""
        for i, j, k in self.curvature:
            if i < kappa.shape[0] and j < kappa.shape[1] and k < kappa.shape[2]:
                self.curvature[i, j, k] = kappa[i, j, k]
    
    @ti.kernel
    def set_surface_tension(self, gamma: ti.f64):
        """Set surface tension coefficient"""
        self.surface_tension[None] = gamma
    
    def get_pressure_numpy(self):
        """Export pressure field as numpy array"""
        return self.pressure.to_numpy()
    
    @ti.kernel
    def compute_pressure_statistics(self) -> ti.f64:
        """Compute pressure statistics for monitoring"""
        max_pressure = 0.0
        min_pressure = 0.0
        for i, j, k in self.pressure:
            if self.cell_type[i, j, k] == 0:
                max_pressure = max(max_pressure, self.pressure[i, j, k])
                min_pressure = min(min_pressure, self.pressure[i, j, k])
        return max_pressure - min_pressure