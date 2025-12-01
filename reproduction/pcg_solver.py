"""
Preconditioned Conjugate Gradient (PCG) Solver for Pressure Poisson Equation

This module implements the PCG solver for the pressure system in iMPM:
∇²p^{n+1} = (ρ/Δt) ∇·v*

Key features:
- 7-point finite difference Laplacian in 3D
- Ghost Fluid Method (GFM) for free surface boundary conditions
- Multiple preconditioners: Jacobi, MIC (Modified Incomplete Cholesky), SSOR
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
    def __init__(self, nx, ny, nz, dx, preconditioner='jacobi'):
        """
        Initialize PCG solver.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx: Grid spacing
            preconditioner: 'jacobi' (default), 'mic', or 'ssor'
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        self.inv_dx2 = 1.0 / (dx * dx)
        
        # Preconditioner selection
        self.preconditioner_type = preconditioner

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
        
        # Preconditioner storage (for MIC and SSOR)
        self.diag = ti.field(dtype=ti.f64, shape=(nx, ny, nz))      # Diagonal elements
        self.mic_diag = ti.field(dtype=ti.f64, shape=(nx, ny, nz))  # MIC modified diagonal
        self.z_temp = ti.field(dtype=ti.f64, shape=(nx, ny, nz))    # Temporary for SSOR/MIC
        
        # SSOR relaxation parameter
        self.omega = 1.7  # Optimal typically between 1.5-1.9

        print(f"PCG Solver initialized for {nx}x{ny}x{nz} grid")
        print(f"  Preconditioner: {preconditioner.upper()}")

    # ==================== Preconditioner Methods ====================

    @ti.kernel
    def compute_diagonal(self):
        """Compute diagonal entries of the negative Laplacian matrix (-∇²)
        
        We use -∇² instead of ∇² to make the matrix positive semi-definite,
        which is required for PCG convergence.
        
        For GFM (Ghost Fluid Method), we count all non-solid neighbors (fluid + air).
        The diagonal is +n_neighbors * inv_dx² (positive, for -∇²).
        """
        for i, j, k in self.diag:
            if self.cell_type[i, j, k] == 0:  # Fluid cells
                # Count all non-solid neighbors (fluid + air)
                n_neighbors = 0.0
                
                # Check -x direction
                if i > 0 and self.cell_type[i-1, j, k] != 1:  # Not solid
                    n_neighbors += 1.0
                # Check +x direction
                if i < self.nx-1 and self.cell_type[i+1, j, k] != 1:  # Not solid
                    n_neighbors += 1.0
                # Check -y direction
                if j > 0 and self.cell_type[i, j-1, k] != 1:  # Not solid
                    n_neighbors += 1.0
                # Check +y direction
                if j < self.ny-1 and self.cell_type[i, j+1, k] != 1:  # Not solid
                    n_neighbors += 1.0
                # Check -z direction
                if k > 0 and self.cell_type[i, j, k-1] != 1:  # Not solid
                    n_neighbors += 1.0
                # Check +z direction
                if k < self.nz-1 and self.cell_type[i, j, k+1] != 1:  # Not solid
                    n_neighbors += 1.0
                
                # Diagonal for negative Laplacian -∇² (positive definite)
                self.diag[i, j, k] = n_neighbors * self.inv_dx2  # Positive!
                
                # Ensure non-zero diagonal for numerical stability
                if self.diag[i, j, k] < 1e-10:
                    self.diag[i, j, k] = self.inv_dx2  # Minimum value
            else:
                self.diag[i, j, k] = 1.0  # Non-fluid cells (positive)

    @ti.kernel
    def compute_mic_factorization(self):
        """Compute Modified Incomplete Cholesky factorization"""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                diag_val = self.diag[i, j, k]
                a_off = self.inv_dx2
                
                # Subtract contributions from previous cells
                if i > 0 and self.cell_type[i-1, j, k] == 0:
                    d_prev = self.mic_diag[i-1, j, k]
                    if ti.abs(d_prev) > 1e-14:
                        diag_val -= a_off * a_off / d_prev
                
                if j > 0 and self.cell_type[i, j-1, k] == 0:
                    d_prev = self.mic_diag[i, j-1, k]
                    if ti.abs(d_prev) > 1e-14:
                        diag_val -= a_off * a_off / d_prev
                
                if k > 0 and self.cell_type[i, j, k-1] == 0:
                    d_prev = self.mic_diag[i, j, k-1]
                    if ti.abs(d_prev) > 1e-14:
                        diag_val -= a_off * a_off / d_prev
                
                # Store modified diagonal (ensure it's negative)
                self.mic_diag[i, j, k] = ti.min(diag_val, -1e-14)
            else:
                self.mic_diag[i, j, k] = -1.0

    @ti.kernel
    def apply_jacobi_preconditioner(self, input_field: ti.template(), output_field: ti.template()):
        """Apply Jacobi preconditioner: z = D^{-1} * r
        
        Uses the actual diagonal values computed from the Laplacian matrix,
        which varies based on the number of non-solid neighbors at each cell.
        """
        for i, j, k in output_field:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                diagonal = self.diag[i, j, k]
                if ti.abs(diagonal) > 1e-12:
                    output_field[i, j, k] = input_field[i, j, k] / diagonal
                else:
                    output_field[i, j, k] = input_field[i, j, k]
            else:
                output_field[i, j, k] = 0.0

    @ti.kernel
    def apply_mic_preconditioner(self, input_field: ti.template(), output_field: ti.template()):
        """Apply MIC preconditioner: solve (LD^{-1}L^T)z = r"""
        a_off = self.inv_dx2
        
        # Forward substitution: Ly* = r
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 0:
                val = input_field[i, j, k]
                
                if i > 0 and self.cell_type[i-1, j, k] == 0:
                    val -= a_off * self.z_temp[i-1, j, k] / self.mic_diag[i-1, j, k]
                if j > 0 and self.cell_type[i, j-1, k] == 0:
                    val -= a_off * self.z_temp[i, j-1, k] / self.mic_diag[i, j-1, k]
                if k > 0 and self.cell_type[i, j, k-1] == 0:
                    val -= a_off * self.z_temp[i, j, k-1] / self.mic_diag[i, j, k-1]
                
                self.z_temp[i, j, k] = val
            else:
                self.z_temp[i, j, k] = 0.0
        
        # Backward substitution: L^T z = D^{-1} y*
        # Note: Using forward loop with reverse index calculation for Taichi compatibility
        for idx in range(self.nx * self.ny * self.nz):
            # Convert linear index to reverse (i, j, k)
            i = self.nx - 1 - (idx % self.nx)
            j = self.ny - 1 - ((idx // self.nx) % self.ny)
            k = self.nz - 1 - (idx // (self.nx * self.ny))
            
            if self.cell_type[i, j, k] == 0:
                val = self.z_temp[i, j, k] / self.mic_diag[i, j, k]
                
                if i < self.nx-1 and self.cell_type[i+1, j, k] == 0:
                    val -= a_off * output_field[i+1, j, k] / self.mic_diag[i, j, k]
                if j < self.ny-1 and self.cell_type[i, j+1, k] == 0:
                    val -= a_off * output_field[i, j+1, k] / self.mic_diag[i, j, k]
                if k < self.nz-1 and self.cell_type[i, j, k+1] == 0:
                    val -= a_off * output_field[i, j, k+1] / self.mic_diag[i, j, k]
                
                output_field[i, j, k] = val
            else:
                output_field[i, j, k] = 0.0

    @ti.kernel
    def apply_ssor_preconditioner(self, input_field: ti.template(), output_field: ti.template()):
        """Apply SSOR preconditioner"""
        omega = self.omega
        a_off = self.inv_dx2
        
        # Forward sweep: (D/ω + L)z* = r
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 0:
                val = input_field[i, j, k]
                
                if i > 0 and self.cell_type[i-1, j, k] == 0:
                    val -= a_off * self.z_temp[i-1, j, k]
                if j > 0 and self.cell_type[i, j-1, k] == 0:
                    val -= a_off * self.z_temp[i, j-1, k]
                if k > 0 and self.cell_type[i, j, k-1] == 0:
                    val -= a_off * self.z_temp[i, j, k-1]
                
                diag_omega = self.diag[i, j, k] / omega
                if ti.abs(diag_omega) > 1e-14:
                    self.z_temp[i, j, k] = val / diag_omega
                else:
                    self.z_temp[i, j, k] = 0.0
            else:
                self.z_temp[i, j, k] = 0.0
        
        # Backward sweep: (D/ω + U)z = ωD z*
        # Note: Using forward loop with reverse index calculation for Taichi compatibility
        for idx in range(self.nx * self.ny * self.nz):
            # Convert linear index to reverse (i, j, k)
            i = self.nx - 1 - (idx % self.nx)
            j = self.ny - 1 - ((idx // self.nx) % self.ny)
            k = self.nz - 1 - (idx // (self.nx * self.ny))
            
            if self.cell_type[i, j, k] == 0:
                val = omega * self.diag[i, j, k] * self.z_temp[i, j, k]
                
                if i < self.nx-1 and self.cell_type[i+1, j, k] == 0:
                    val -= a_off * output_field[i+1, j, k]
                if j < self.ny-1 and self.cell_type[i, j+1, k] == 0:
                    val -= a_off * output_field[i, j+1, k]
                if k < self.nz-1 and self.cell_type[i, j, k+1] == 0:
                    val -= a_off * output_field[i, j, k+1]
                
                diag_omega = self.diag[i, j, k] / omega
                if ti.abs(diag_omega) > 1e-14:
                    output_field[i, j, k] = val / diag_omega
                else:
                    output_field[i, j, k] = 0.0
            else:
                output_field[i, j, k] = 0.0

    def apply_preconditioner(self, input_field, output_field):
        """Apply selected preconditioner"""
        if self.preconditioner_type == 'mic':
            self.apply_mic_preconditioner(input_field, output_field)
        elif self.preconditioner_type == 'ssor':
            self.apply_ssor_preconditioner(input_field, output_field)
        else:  # jacobi (default)
            self.apply_jacobi_preconditioner(input_field, output_field)

    def setup_preconditioner(self):
        """Setup preconditioner (compute factorizations if needed)"""
        # All preconditioners need the diagonal matrix
        self.compute_diagonal()
        
        # Additional setup for specific preconditioners
        if self.preconditioner_type == 'mic':
            self.compute_mic_factorization()

    # ==================== RHS and Laplacian Methods ====================

    @ti.kernel
    def setup_rhs(self, div_v_star: ti.template(), rho: ti.f64, dt: ti.f64):
        """Setup right-hand side: b = (ρ/Δt)∇·v* for pressure Poisson equation
        
        Note: We solve Ap = b where A = -∇² (negative Laplacian) is positive semi-definite.
        This requires the RHS to also be negated: b = -(ρ/Δt)∇·v*
        so we effectively solve: -∇²p = -(ρ/Δt)∇·v*
        which is equivalent to the original: ∇²p = (ρ/Δt)∇·v*
        """
        neg_rho_over_dt = -rho / dt  # Note the negative sign
        for i, j, k in self.rhs:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                self.rhs[i, j, k] = neg_rho_over_dt * div_v_star[i, j, k]
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
                phi_fluid = self.level_set[i, j, k]
                phi_air = self.level_set[i+1, j, k]
                if abs(phi_fluid - phi_air) > 1e-10:
                    theta = abs(phi_fluid) / (abs(phi_fluid) + abs(phi_air))
                    p_fs = self.p_air + self.surface_tension[None] * self.curvature[i, j, k]
                    ghost_pressure = (p_fs + (theta - 1.0) * self.pressure[i, j, k]) / theta
                else:
                    ghost_pressure = self.p_air
            else:
                ghost_pressure = self.pressure[i+1, j, k]

        elif direction == 1 and i > 0:  # -x direction
            if self.cell_type[i-1, j, k] == 2:
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
        """Apply negative Laplacian (-∇²) with Ghost Fluid Method
        
        We compute -∇² instead of ∇² to make the matrix positive semi-definite.
        -∇²p = n_neighbors * p_center - sum(p_neighbors)
        
        This is consistent with the positive diagonal computed in compute_diagonal().
        """
        for i, j, k in output_field:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                # Count actual non-solid neighbors and sum their pressure contributions
                neighbors_sum = 0.0
                n_neighbors = 0
                
                # -x direction
                if i > 0 and self.cell_type[i-1, j, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i-1, j, k] == 0:  # Fluid
                        neighbors_sum += input_field[i-1, j, k]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 1)
                
                # +x direction
                if i < self.nx-1 and self.cell_type[i+1, j, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i+1, j, k] == 0:  # Fluid
                        neighbors_sum += input_field[i+1, j, k]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 0)
                
                # -y direction
                if j > 0 and self.cell_type[i, j-1, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j-1, k] == 0:  # Fluid
                        neighbors_sum += input_field[i, j-1, k]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 3)
                
                # +y direction
                if j < self.ny-1 and self.cell_type[i, j+1, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j+1, k] == 0:  # Fluid
                        neighbors_sum += input_field[i, j+1, k]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 2)
                
                # -z direction
                if k > 0 and self.cell_type[i, j, k-1] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j, k-1] == 0:  # Fluid
                        neighbors_sum += input_field[i, j, k-1]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 5)
                
                # +z direction
                if k < self.nz-1 and self.cell_type[i, j, k+1] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j, k+1] == 0:  # Fluid
                        neighbors_sum += input_field[i, j, k+1]
                    else:  # Air - use ghost pressure
                        neighbors_sum += self.get_ghost_pressure(i, j, k, 4)
                
                # Negative Laplacian: +n_neighbors * p_center - sum(p_neighbors)
                # This is consistent with self.diag[i,j,k] = +n_neighbors * inv_dx²
                center = float(n_neighbors) * input_field[i, j, k]
                output_field[i, j, k] = self.inv_dx2 * (center - neighbors_sum)
            else:
                output_field[i, j, k] = 0.0

    # ==================== Vector Operations ====================

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

    @ti.kernel
    def compute_field_sum(self, field: ti.template()) -> ti.f64:
        """Compute sum of field values in fluid cells"""
        total = 0.0
        for i, j, k in field:
            if self.cell_type[i, j, k] == 0:
                total += field[i, j, k]
        return total
    
    @ti.kernel
    def count_fluid_cells(self) -> ti.i32:
        """Count number of fluid cells"""
        count = 0
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == 0:
                count += 1
        return count
    
    @ti.kernel
    def subtract_mean(self, field: ti.template(), mean: ti.f64):
        """Subtract mean from field in fluid cells"""
        for i, j, k in field:
            if self.cell_type[i, j, k] == 0:
                field[i, j, k] -= mean
    
    def remove_null_space(self, field):
        """Remove constant null space component (for Neumann BC stability)
        
        For pure Neumann boundary conditions, the pressure is only determined
        up to a constant. We remove the mean to ensure a unique solution and
        improve PCG convergence.
        """
        count = self.count_fluid_cells()
        if count > 0:
            total = self.compute_field_sum(field)
            mean = total / count
            self.subtract_mean(field, mean)

    # ==================== Main Solve Method ====================

    def solve_pcg(self, div_v_star, max_iter=200, tol=1e-4, rho=1000.0, dt=1e-4):
        """Solve pressure Poisson equation using PCG
        
        Args:
            div_v_star: Velocity divergence field
            max_iter: Maximum iterations
            tol: Convergence tolerance
            rho: Fluid density (kg/m³)
            dt: Time step (s)
        """
        # Setup RHS: b = -(ρ/Δt)∇·v*
        self.setup_rhs(div_v_star, rho, dt)
        
        # Remove null space from RHS for Neumann BC stability
        self.remove_null_space(self.rhs)
        
        # Setup preconditioner (for MIC/SSOR)
        self.setup_preconditioner()

        # Compute initial residual: r = b - Ap
        self.apply_laplacian(self.pressure, self.Ap)
        self.compute_initial_residual()

        # Apply preconditioner: z = M^{-1} * r
        self.apply_preconditioner(self.r, self.z)

        # Initial search direction: p = z
        self.vector_copy(self.z, self.p)

        # Initial dot product: rz_old = r · z
        rz_old = self.compute_dot_product(self.r, self.z)

        # Check for immediate convergence (use relative residual)
        initial_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        if initial_residual < 1e-14:
            print(f"PCG converged immediately, residual = {initial_residual:.2e}")
            return 0

        # PCG iteration with relative residual tolerance
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
            
            # Remove null space from pressure to prevent drift
            self.remove_null_space(self.pressure)

            # Update residual: r = r - alpha * Ap
            self.vector_axpy(-alpha, self.Ap, self.r)

            # Check convergence using RELATIVE residual
            residual_norm = ti.sqrt(self.compute_dot_product(self.r, self.r))
            relative_residual = residual_norm / initial_residual
            if relative_residual < tol:
                print(f"PCG converged in {iteration+1} iterations, rel_residual = {relative_residual:.2e}")
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
                print(f"  PCG iteration {iteration+1}: rel_residual = {relative_residual:.2e}")

        final_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        final_relative = final_residual / initial_residual
        print(f"PCG did not converge in {max_iter} iterations, rel_residual = {final_relative:.2e}")
        return max_iter

    # ==================== Utility Methods ====================

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
