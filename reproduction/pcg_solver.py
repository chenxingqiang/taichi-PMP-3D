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
- Two-phase flow support with porosity-weighted pressure gradient

Mathematical framework:
- Linear system: Ap = b where A is the discrete Laplacian
- Ghost cells: p^G = (p^fs + (θ-1)p^f)/θ for free surface BCs
- Solid wall BCs: ∇p·n = 0 (no penetration condition)

Two-phase flow equations (from paper):
- Solid momentum: ρ̄s(Dvs/Dt) = ρ̄sg + ∇·σ' - fd - φ∇pf     [Eq. 5]
- Fluid momentum: ρ̄f(Dvf/Dt) = ρ̄fg + ∇·Tf + fd - (1-φ)∇pf  [Eq. 6]
- The pore pressure gradient is shared between phases with porosity weighting
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
        
        # Two-phase flow support: porosity field
        self.porosity = ti.field(dtype=ti.f64, shape=(nx, ny, nz))  # φ: solid volume fraction
        self.pressure_gradient = ti.Vector.field(3, dtype=ti.f64, shape=(nx, ny, nz))  # ∇p
        self.two_phase_mode = False  # Flag for two-phase flow mode

        print(f"PCG Solver initialized for {nx}x{ny}x{nz} grid")
        print(f"  Preconditioner: {preconditioner.upper()}")

    # ==================== Preconditioner Methods ====================

    @ti.kernel
    def compute_diagonal(self):
        """Compute diagonal entries of the negative Laplacian matrix (-∇²)
        
        Following Bridson's approach (Figure 5.4):
        - For SOLID cell boundary: delete mention of that p AND reduce the coefficient
          in front of p_{i,j} by one
        - The coefficient in front of p_{i,j} = number of NON-SOLID grid cell neighbors
        
        We use -∇² instead of ∇² to make the matrix positive semi-definite,
        which is required for PCG convergence.
        
        For GFM (Ghost Fluid Method), we count all non-solid neighbors (fluid + air).
        The diagonal is +n_neighbors * inv_dx² (positive, for -∇²).
        """
        for i, j, k in self.diag:
            if self.cell_type[i, j, k] == 0:  # Fluid cells
                # Count all non-solid neighbors (fluid + air)
                # This implements Bridson's rule: diagonal = number of non-solid neighbors
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
                # coefficient = n_non_solid_neighbors / dx²
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

    @ti.kernel
    def setup_rhs_with_solid_velocity(self, div_v_star: ti.template(), 
                                       v_star: ti.template(),
                                       v_solid: ti.template(),
                                       rho: ti.f64, dt: ti.f64):
        """Setup RHS with solid boundary velocity contribution (Bridson Figure 5.4)
        
        For solid boundaries, we need to modify RHS to account for solid velocities:
        - When neighbor is SOLID, add: scale * (u_fluid - u_solid) to RHS
        
        This ensures the pressure solve produces correct velocity at solid boundaries:
        ∂p/∂n = ρ/dt * (u_fluid - u_solid) · n
        
        Args:
            div_v_star: Velocity divergence field
            v_star: Intermediate velocity field (3D vector field)
            v_solid: Solid velocity field (3D vector field, can be zero for static solids)
            rho: Fluid density
            dt: Time step
        """
        scale = 1.0 / self.dx
        neg_rho_over_dt = -rho / dt
        
        for i, j, k in self.rhs:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                # Start with divergence term
                rhs_val = neg_rho_over_dt * div_v_star[i, j, k]
                
                # Add solid velocity contributions (from Bridson Figure 5.4)
                # -x direction: if label(i-1,j,k)==SOLID
                if i > 0 and self.cell_type[i-1, j, k] == 1:  # Solid
                    # rhs -= scale * (u(i,j,k) - u_solid(i,j,k))
                    # For negative Laplacian formulation, we add instead
                    rhs_val += scale * (v_star[i, j, k][0] - v_solid[i-1, j, k][0])
                
                # +x direction: if label(i+1,j,k)==SOLID
                if i < self.nx-1 and self.cell_type[i+1, j, k] == 1:  # Solid
                    # rhs += scale * (u(i+1,j,k) - u_solid(i+1,j,k))
                    rhs_val -= scale * (v_star[i+1, j, k][0] - v_solid[i+1, j, k][0])
                
                # -y direction: if label(i,j-1,k)==SOLID
                if j > 0 and self.cell_type[i, j-1, k] == 1:  # Solid
                    rhs_val += scale * (v_star[i, j, k][1] - v_solid[i, j-1, k][1])
                
                # +y direction: if label(i,j+1,k)==SOLID
                if j < self.ny-1 and self.cell_type[i, j+1, k] == 1:  # Solid
                    rhs_val -= scale * (v_star[i, j+1, k][1] - v_solid[i, j+1, k][1])
                
                # -z direction: if label(i,j,k-1)==SOLID
                if k > 0 and self.cell_type[i, j, k-1] == 1:  # Solid
                    rhs_val += scale * (v_star[i, j, k][2] - v_solid[i, j, k-1][2])
                
                # +z direction: if label(i,j,k+1)==SOLID
                if k < self.nz-1 and self.cell_type[i, j, k+1] == 1:  # Solid
                    rhs_val -= scale * (v_star[i, j, k+1][2] - v_solid[i, j, k+1][2])
                
                self.rhs[i, j, k] = rhs_val
            else:
                self.rhs[i, j, k] = 0.0

    @ti.kernel
    def setup_rhs_with_static_solid(self, div_v_star: ti.template(), 
                                     v_star: ti.template(),
                                     rho: ti.f64, dt: ti.f64):
        """Setup RHS for static solid boundaries (u_solid = 0)
        
        Simplified version of setup_rhs_with_solid_velocity for static solids.
        This is the most common case in MPM simulations where walls don't move.
        
        The key modification from Bridson Figure 5.4:
        - When neighbor is SOLID: rhs ± scale * u_fluid
        """
        scale = 1.0 / self.dx
        neg_rho_over_dt = -rho / dt
        
        for i, j, k in self.rhs:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                # Start with divergence term
                rhs_val = neg_rho_over_dt * div_v_star[i, j, k]
                
                # Add static solid velocity contributions (u_solid = 0)
                # -x direction
                if i > 0 and self.cell_type[i-1, j, k] == 1:
                    rhs_val += scale * v_star[i, j, k][0]
                
                # +x direction
                if i < self.nx-1 and self.cell_type[i+1, j, k] == 1:
                    rhs_val -= scale * v_star[i+1, j, k][0]
                
                # -y direction
                if j > 0 and self.cell_type[i, j-1, k] == 1:
                    rhs_val += scale * v_star[i, j, k][1]
                
                # +y direction
                if j < self.ny-1 and self.cell_type[i, j+1, k] == 1:
                    rhs_val -= scale * v_star[i, j+1, k][1]
                
                # -z direction
                if k > 0 and self.cell_type[i, j, k-1] == 1:
                    rhs_val += scale * v_star[i, j, k][2]
                
                # +z direction
                if k < self.nz-1 and self.cell_type[i, j, k+1] == 1:
                    rhs_val -= scale * v_star[i, j, k+1][2]
                
                self.rhs[i, j, k] = rhs_val
            else:
                self.rhs[i, j, k] = 0.0

    @ti.func
    def get_ghost_pressure(self, i, j, k, direction, input_field: ti.template()):
        """Compute ghost pressure using simplified Ghost Fluid Method
        
        For free surface flows, we use p = p_air = 0 at the interface.
        This provides a simple but effective Dirichlet BC that keeps
        the linear system well-conditioned for PCG.
        
        The matrix remains linear and symmetric positive semi-definite.
        """
        # direction: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
        ghost_pressure = 0.0

        # Neighbor indices based on direction
        ni, nj, nk = i, j, k
        if direction == 0:
            ni = i + 1
        elif direction == 1:
            ni = i - 1
        elif direction == 2:
            nj = j + 1
        elif direction == 3:
            nj = j - 1
        elif direction == 4:
            nk = k + 1
        elif direction == 5:
            nk = k - 1
        
        # Check bounds
        if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
            if self.cell_type[ni, nj, nk] == 2:  # Air cell
                # Simple GFM: use atmospheric pressure (Dirichlet BC)
                # p_air = 0 (gauge pressure)
                ghost_pressure = self.p_air
            elif self.cell_type[ni, nj, nk] == 0:  # Fluid cell
                ghost_pressure = input_field[ni, nj, nk]
            # Solid cells: use Neumann BC (∂p/∂n = 0), handled separately

        return ghost_pressure

    @ti.kernel
    def apply_laplacian(self, input_field: ti.template(), output_field: ti.template()):
        """Apply negative Laplacian (-∇²) with simplified Ghost Fluid Method
        
        We compute -∇² instead of ∇² to make the matrix positive semi-definite.
        -∇²p = n_neighbors * p_center - sum(p_neighbors)
        
        At free surface boundaries (fluid-air), we use p_air = 0 (Dirichlet BC).
        This keeps the matrix linear and symmetric.
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
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
                # +x direction
                if i < self.nx-1 and self.cell_type[i+1, j, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i+1, j, k] == 0:  # Fluid
                        neighbors_sum += input_field[i+1, j, k]
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
                # -y direction
                if j > 0 and self.cell_type[i, j-1, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j-1, k] == 0:  # Fluid
                        neighbors_sum += input_field[i, j-1, k]
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
                # +y direction
                if j < self.ny-1 and self.cell_type[i, j+1, k] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j+1, k] == 0:  # Fluid
                        neighbors_sum += input_field[i, j+1, k]
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
                # -z direction
                if k > 0 and self.cell_type[i, j, k-1] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j, k-1] == 0:  # Fluid
                        neighbors_sum += input_field[i, j, k-1]
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
                # +z direction
                if k < self.nz-1 and self.cell_type[i, j, k+1] != 1:  # Not solid
                    n_neighbors += 1
                    if self.cell_type[i, j, k+1] == 0:  # Fluid
                        neighbors_sum += input_field[i, j, k+1]
                    else:  # Air - use p_air = 0
                        neighbors_sum += self.p_air
                
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
    def _count_air_boundaries(self) -> ti.i32:
        """Count fluid cells that have at least one air neighbor (Dirichlet BC)"""
        count = 0
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == 0:  # Fluid cell
                has_air = False
                if i > 0 and self.cell_type[i-1, j, k] == 2:
                    has_air = True
                if i < self.nx-1 and self.cell_type[i+1, j, k] == 2:
                    has_air = True
                if j > 0 and self.cell_type[i, j-1, k] == 2:
                    has_air = True
                if j < self.ny-1 and self.cell_type[i, j+1, k] == 2:
                    has_air = True
                if k > 0 and self.cell_type[i, j, k-1] == 2:
                    has_air = True
                if k < self.nz-1 and self.cell_type[i, j, k+1] == 2:
                    has_air = True
                if has_air:
                    count += 1
        return count
    
    def has_air_boundary(self) -> bool:
        """Check if there are any fluid cells adjacent to air (Dirichlet BC)"""
        return self._count_air_boundaries() > 0
    
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

    def solve_pcg(self, div_v_star, max_iter=200, tol=1e-4, rho=1000.0, dt=1e-4,
                  v_star=None, v_solid=None, use_solid_bc=False):
        """Solve pressure Poisson equation using PCG
        
        Args:
            div_v_star: Velocity divergence field
            max_iter: Maximum iterations
            tol: Convergence tolerance
            rho: Fluid density (kg/m³)
            dt: Time step (s)
            v_star: Intermediate velocity field (required if use_solid_bc=True)
            v_solid: Solid velocity field (optional, defaults to zero/static)
            use_solid_bc: Whether to use solid boundary velocity correction (Bridson 5.4)
        """
        # Setup RHS based on boundary condition mode
        if use_solid_bc and v_star is not None:
            if v_solid is not None:
                # Full solid velocity treatment
                self.setup_rhs_with_solid_velocity(div_v_star, v_star, v_solid, rho, dt)
            else:
                # Static solid (u_solid = 0)
                self.setup_rhs_with_static_solid(div_v_star, v_star, rho, dt)
        else:
            # Standard RHS setup
            self.setup_rhs(div_v_star, rho, dt)
        
        # Check if we have Dirichlet BC (air neighbors)
        # If so, the matrix is not singular and we shouldn't remove null space
        has_dirichlet = self.has_air_boundary()
        
        if not has_dirichlet:
            # Only remove null space for pure Neumann BC (enclosed domain)
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
            
            # Note: Don't remove null space here - only needed for pure Neumann BC
            # and it was already handled in the RHS setup

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

    # ==================== Two-Phase Flow Methods ====================
    
    def set_two_phase_mode(self, enabled: bool):
        """Enable or disable two-phase flow mode
        
        When enabled, pressure gradient calculations consider porosity weighting:
        - Solid phase receives: -φ∇pf
        - Fluid phase receives: -(1-φ)∇pf
        """
        self.two_phase_mode = enabled
        if enabled:
            print("  Two-phase flow mode: ENABLED")
    
    @ti.kernel
    def update_porosity(self, porosity_field: ti.template()):
        """Update porosity field from external source (e.g., from two-phase solver)
        
        Args:
            porosity_field: External field containing solid volume fraction φ
        """
        for i, j, k in self.porosity:
            if i < porosity_field.shape[0] and j < porosity_field.shape[1] and k < porosity_field.shape[2]:
                self.porosity[i, j, k] = porosity_field[i, j, k]
            else:
                self.porosity[i, j, k] = 0.0

    @ti.kernel
    def compute_pressure_gradient(self):
        """Compute pressure gradient ∇p at each cell center
        
        Uses central differences for interior cells and one-sided differences at boundaries.
        The gradient is stored for later use in computing porosity-weighted contributions.
        """
        for i, j, k in self.pressure_gradient:
            grad_p_x = 0.0
            grad_p_y = 0.0
            grad_p_z = 0.0
            
            # X-direction gradient
            if i == 0:
                grad_p_x = (self.pressure[i+1, j, k] - self.pressure[i, j, k]) / self.dx
            elif i == self.nx - 1:
                grad_p_x = (self.pressure[i, j, k] - self.pressure[i-1, j, k]) / self.dx
            else:
                grad_p_x = (self.pressure[i+1, j, k] - self.pressure[i-1, j, k]) / (2.0 * self.dx)
            
            # Y-direction gradient
            if j == 0:
                grad_p_y = (self.pressure[i, j+1, k] - self.pressure[i, j, k]) / self.dx
            elif j == self.ny - 1:
                grad_p_y = (self.pressure[i, j, k] - self.pressure[i, j-1, k]) / self.dx
            else:
                grad_p_y = (self.pressure[i, j+1, k] - self.pressure[i, j-1, k]) / (2.0 * self.dx)
            
            # Z-direction gradient
            if k == 0:
                grad_p_z = (self.pressure[i, j, k+1] - self.pressure[i, j, k]) / self.dx
            elif k == self.nz - 1:
                grad_p_z = (self.pressure[i, j, k] - self.pressure[i, j, k-1]) / self.dx
            else:
                grad_p_z = (self.pressure[i, j, k+1] - self.pressure[i, j, k-1]) / (2.0 * self.dx)
            
            self.pressure_gradient[i, j, k] = ti.Vector([grad_p_x, grad_p_y, grad_p_z])

    @ti.kernel
    def get_solid_phase_pressure_gradient(self, output_field: ti.template()):
        """Get porosity-weighted pressure gradient for solid phase: -φ∇pf
        
        This implements the pressure gradient term in the solid momentum equation [Eq. 5]:
        ρ̄s(Dvs/Dt) = ρ̄sg + ∇·σ' - fd - φ∇pf
        
        Args:
            output_field: Vector field to store the weighted pressure gradient
        """
        for i, j, k in output_field:
            phi = self.porosity[i, j, k]  # solid volume fraction
            # Solid phase receives -φ∇pf
            output_field[i, j, k] = -phi * self.pressure_gradient[i, j, k]

    @ti.kernel
    def get_fluid_phase_pressure_gradient(self, output_field: ti.template()):
        """Get porosity-weighted pressure gradient for fluid phase: -(1-φ)∇pf
        
        This implements the pressure gradient term in the fluid momentum equation [Eq. 6]:
        ρ̄f(Dvf/Dt) = ρ̄fg + ∇·Tf + fd - (1-φ)∇pf
        
        Args:
            output_field: Vector field to store the weighted pressure gradient
        """
        for i, j, k in output_field:
            phi = self.porosity[i, j, k]  # solid volume fraction
            phi_l = 1.0 - phi  # liquid volume fraction
            # Fluid phase receives -(1-φ)∇pf
            output_field[i, j, k] = -phi_l * self.pressure_gradient[i, j, k]

    @ti.kernel
    def apply_two_phase_laplacian(self, input_field: ti.template(), output_field: ti.template()):
        """Apply porosity-modified Laplacian for two-phase flow
        
        For two-phase flow, the pressure equation becomes:
        ∇·[(1-φ)∇p] = RHS
        
        This modifies the standard 7-point stencil to account for variable porosity.
        The off-diagonal coefficients are weighted by the harmonic mean of
        (1-φ) at the cell faces.
        """
        for i, j, k in output_field:
            if self.cell_type[i, j, k] == 0:  # Fluid cells only
                neighbors_sum = 0.0
                center_coeff = 0.0
                
                phi_c = self.porosity[i, j, k]
                perm_c = 1.0 - phi_c  # liquid fraction at center
                
                # -x direction
                if i > 0 and self.cell_type[i-1, j, k] != 1:
                    phi_n = self.porosity[i-1, j, k]
                    perm_n = 1.0 - phi_n
                    # Harmonic mean for interface permeability
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i-1, j, k] == 0:  # Fluid
                        neighbors_sum += perm_face * input_field[i-1, j, k]
                    else:  # Air - use p_air = 0
                        neighbors_sum += perm_face * self.p_air
                
                # +x direction
                if i < self.nx-1 and self.cell_type[i+1, j, k] != 1:
                    phi_n = self.porosity[i+1, j, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i+1, j, k] == 0:
                        neighbors_sum += perm_face * input_field[i+1, j, k]
                    else:
                        neighbors_sum += perm_face * self.p_air
                
                # -y direction
                if j > 0 and self.cell_type[i, j-1, k] != 1:
                    phi_n = self.porosity[i, j-1, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i, j-1, k] == 0:
                        neighbors_sum += perm_face * input_field[i, j-1, k]
                    else:
                        neighbors_sum += perm_face * self.p_air
                
                # +y direction
                if j < self.ny-1 and self.cell_type[i, j+1, k] != 1:
                    phi_n = self.porosity[i, j+1, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i, j+1, k] == 0:
                        neighbors_sum += perm_face * input_field[i, j+1, k]
                    else:
                        neighbors_sum += perm_face * self.p_air
                
                # -z direction
                if k > 0 and self.cell_type[i, j, k-1] != 1:
                    phi_n = self.porosity[i, j, k-1]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i, j, k-1] == 0:
                        neighbors_sum += perm_face * input_field[i, j, k-1]
                    else:
                        neighbors_sum += perm_face * self.p_air
                
                # +z direction
                if k < self.nz-1 and self.cell_type[i, j, k+1] != 1:
                    phi_n = self.porosity[i, j, k+1]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    if self.cell_type[i, j, k+1] == 0:
                        neighbors_sum += perm_face * input_field[i, j, k+1]
                    else:
                        neighbors_sum += perm_face * self.p_air
                
                # Negative Laplacian with porosity weighting
                center = center_coeff * input_field[i, j, k]
                output_field[i, j, k] = self.inv_dx2 * (center - neighbors_sum)
            else:
                output_field[i, j, k] = 0.0

    @ti.kernel
    def compute_two_phase_diagonal(self):
        """Compute diagonal entries for two-phase Laplacian matrix
        
        The diagonal is modified to account for variable porosity across cell faces.
        """
        for i, j, k in self.diag:
            if self.cell_type[i, j, k] == 0:  # Fluid cells
                phi_c = self.porosity[i, j, k]
                perm_c = 1.0 - phi_c  # liquid fraction at center
                center_coeff = 0.0
                
                # Sum contributions from all non-solid neighbors
                if i > 0 and self.cell_type[i-1, j, k] != 1:
                    phi_n = self.porosity[i-1, j, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    
                if i < self.nx-1 and self.cell_type[i+1, j, k] != 1:
                    phi_n = self.porosity[i+1, j, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    
                if j > 0 and self.cell_type[i, j-1, k] != 1:
                    phi_n = self.porosity[i, j-1, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    
                if j < self.ny-1 and self.cell_type[i, j+1, k] != 1:
                    phi_n = self.porosity[i, j+1, k]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    
                if k > 0 and self.cell_type[i, j, k-1] != 1:
                    phi_n = self.porosity[i, j, k-1]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                    
                if k < self.nz-1 and self.cell_type[i, j, k+1] != 1:
                    phi_n = self.porosity[i, j, k+1]
                    perm_n = 1.0 - phi_n
                    perm_face = 2.0 * perm_c * perm_n / (perm_c + perm_n + 1e-14)
                    center_coeff += perm_face
                
                # Diagonal for negative Laplacian (positive definite)
                self.diag[i, j, k] = center_coeff * self.inv_dx2
                
                # Ensure non-zero diagonal
                if self.diag[i, j, k] < 1e-10:
                    self.diag[i, j, k] = self.inv_dx2
            else:
                self.diag[i, j, k] = 1.0  # Non-fluid cells

    def solve_two_phase_pcg(self, div_v_star, max_iter=200, tol=1e-4, rho=1000.0, dt=1e-4):
        """Solve pressure Poisson equation for two-phase flow using PCG
        
        The pressure equation for two-phase flow is:
        ∇·[(1-φ)∇p] = (ρ/Δt)∇·[(1-φ)v*]
        
        Args:
            div_v_star: Velocity divergence field (weighted by liquid fraction if needed)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            rho: Fluid density (kg/m³)
            dt: Time step (s)
        """
        # Setup RHS
        self.setup_rhs(div_v_star, rho, dt)
        
        # Check for Dirichlet BC
        has_dirichlet = self.has_air_boundary()
        if not has_dirichlet:
            self.remove_null_space(self.rhs)
        
        # Setup preconditioner with two-phase diagonal
        self.compute_two_phase_diagonal()
        
        # Compute initial residual using two-phase Laplacian
        self.apply_two_phase_laplacian(self.pressure, self.Ap)
        self.compute_initial_residual()
        
        # Apply preconditioner
        self.apply_preconditioner(self.r, self.z)
        
        # Initial search direction
        self.vector_copy(self.z, self.p)
        
        # Initial dot product
        rz_old = self.compute_dot_product(self.r, self.z)
        
        # Check immediate convergence
        initial_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        if initial_residual < 1e-14:
            print(f"Two-phase PCG converged immediately, residual = {initial_residual:.2e}")
            # Compute and store pressure gradient
            self.compute_pressure_gradient()
            return 0
        
        # PCG iteration
        for iteration in range(max_iter):
            # Use two-phase Laplacian
            self.apply_two_phase_laplacian(self.p, self.Ap)
            
            pAp = self.compute_dot_product(self.p, self.Ap)
            if abs(pAp) < 1e-14:
                print(f"Two-phase PCG breakdown: pAp = {pAp}")
                break
            
            alpha = rz_old / pAp
            self.vector_axpy(alpha, self.p, self.pressure)
            self.vector_axpy(-alpha, self.Ap, self.r)
            
            residual_norm = ti.sqrt(self.compute_dot_product(self.r, self.r))
            relative_residual = residual_norm / initial_residual
            if relative_residual < tol:
                print(f"Two-phase PCG converged in {iteration+1} iters, rel_res = {relative_residual:.2e}")
                # Compute and store pressure gradient for phase-weighted contributions
                self.compute_pressure_gradient()
                return iteration + 1
            
            self.apply_preconditioner(self.r, self.z)
            rz_new = self.compute_dot_product(self.r, self.z)
            if abs(rz_old) < 1e-14:
                print(f"Two-phase PCG breakdown: rz_old = {rz_old}")
                break
            
            beta = rz_new / rz_old
            self.vector_scale(beta, self.p)
            self.vector_axpy(1.0, self.z, self.p)
            rz_old = rz_new
            
            if (iteration + 1) % 10 == 0:
                print(f"  Two-phase PCG iter {iteration+1}: rel_res = {relative_residual:.2e}")
        
        final_residual = ti.sqrt(self.compute_dot_product(self.r, self.r))
        final_relative = final_residual / initial_residual
        print(f"Two-phase PCG did not converge in {max_iter} iters, rel_res = {final_relative:.2e}")
        
        # Still compute pressure gradient for use
        self.compute_pressure_gradient()
        return max_iter
    
    def get_pressure_gradient_numpy(self):
        """Export pressure gradient field as numpy array"""
        return self.pressure_gradient.to_numpy()
    
    def get_porosity_numpy(self):
        """Export porosity field as numpy array"""
        return self.porosity.to_numpy()
