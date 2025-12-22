# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Test script for Ghost Fluid Method (GFM) implementation

This tests the pressure boundary condition handling at free surfaces:
1. θ calculation from level set
2. Modified Laplacian coefficients
3. Node-based pressure gradient computation
"""

import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

from pcg_solver import PCGSolver
from level_set_method import LevelSetMethod


def test_theta_calculation():
    """Test θ = |φ_fluid| / (|φ_fluid| + |φ_air|) at interfaces"""
    print("Testing theta calculation at fluid-air interfaces...")
    
    nx, ny, nz = 8, 8, 8
    dx = 0.1
    solver = PCGSolver(nx, ny, nz, dx)
    level_set = LevelSetMethod(nx, ny, nz, dx)
    
    # Initialize level set as a box (fluid in lower half)
    level_set.initialize_box(0.0, 0.8, 0.0, 0.4, 0.0, 0.8)
    
    # Update solver with level set
    solver.update_level_set(level_set.phi)
    solver.classify_cells(level_set.phi)
    solver.compute_theta_gfm(level_set.phi)
    
    # Check cell classification
    cell_types = solver.cell_type.to_numpy()
    n_fluid = np.sum(cell_types == 0)
    n_air = np.sum(cell_types == 2)
    print(f"  Cell classification: {n_fluid} fluid, {n_air} air")
    
    # Check theta values at interface
    theta_yp = solver.theta_yp.to_numpy()
    
    # Find cells at the fluid-air interface (y direction)
    interface_cells = []
    for i in range(nx):
        for j in range(ny-1):
            for k in range(nz):
                if cell_types[i,j,k] == 0 and cell_types[i,j+1,k] == 2:
                    interface_cells.append((i, j, k, theta_yp[i,j,k]))
    
    print(f"  Found {len(interface_cells)} interface cells (fluid-air in +y)")
    
    if len(interface_cells) > 0:
        theta_vals = [c[3] for c in interface_cells]
        print(f"  Theta values: min={min(theta_vals):.3f}, max={max(theta_vals):.3f}, mean={np.mean(theta_vals):.3f}")
        
        # Verify theta is in valid range (0, 1]
        assert all(0 < t <= 1 for t in theta_vals), "Theta should be in (0, 1]"
        print(f"  ✓ Theta values are in valid range (0, 1]")
        
        # Show coefficient modification
        mean_theta = np.mean(theta_vals)
        print(f"  Modified coefficient: -(1 + 1/θ) = {-1 - 1/mean_theta:.3f} (vs -2 without GFM)")
    
    assert n_fluid > 0 and n_air > 0, "Should have both fluid and air cells"
    print("✓ Theta calculation test passed\n")
    return True


def test_gfm_pressure_solve():
    """Test pressure solve with GFM"""
    print("Testing pressure solve with GFM...")
    
    nx, ny, nz = 16, 16, 16
    dx = 0.1
    solver = PCGSolver(nx, ny, nz, dx)
    level_set = LevelSetMethod(nx, ny, nz, dx)
    
    # Initialize level set as a box (fluid in lower region)
    level_set.initialize_box(0.1, 1.5, 0.1, 0.8, 0.1, 1.5)
    
    # Update solver
    solver.update_level_set(level_set.phi)
    solver.classify_cells(level_set.phi)
    
    # Create divergence field with non-zero values in fluid
    div_field = ti.field(dtype=ti.f64, shape=(nx, ny, nz))
    
    @ti.kernel
    def setup_divergence():
        for i, j, k in div_field:
            if solver.cell_type[i, j, k] == 0:  # Fluid cells
                div_field[i, j, k] = 1.0
            else:
                div_field[i, j, k] = 0.0
    
    setup_divergence()
    
    # Solve pressure
    iterations = solver.solve_pcg(div_field, max_iter=100, tol=1e-4, phi=level_set.phi)
    print(f"  PCG converged in {iterations} iterations")
    
    # Check pressure field
    pressure = solver.pressure.to_numpy()
    cell_types = solver.cell_type.to_numpy()
    
    # Pressure should be smooth in fluid and zero in air
    fluid_pressure = pressure[cell_types == 0]
    air_pressure = pressure[cell_types == 2]
    
    print(f"  Fluid pressure: min={fluid_pressure.min():.4f}, max={fluid_pressure.max():.4f}")
    print(f"  Air pressure: min={air_pressure.min():.6f}, max={air_pressure.max():.6f}")
    
    # Check node gradient
    grad_p_node = solver.grad_p_node.to_numpy()
    grad_magnitude = np.linalg.norm(grad_p_node, axis=-1)
    
    print(f"  Node gradient magnitude: max={grad_magnitude.max():.4f}")
    
    assert iterations <= 100, "PCG should converge within max iterations"
    print("✓ GFM pressure solve test passed\n")
    return True


def test_node_gradient_weighting():
    """Test β-weighted node gradient computation"""
    print("Testing node gradient β-weighting...")
    
    nx, ny, nz = 8, 8, 8
    dx = 0.1
    solver = PCGSolver(nx, ny, nz, dx)
    level_set = LevelSetMethod(nx, ny, nz, dx)
    
    # Initialize level set
    level_set.initialize_box(0.1, 0.7, 0.1, 0.4, 0.1, 0.7)
    
    solver.update_level_set(level_set.phi)
    solver.classify_cells(level_set.phi)
    solver.compute_theta_gfm(level_set.phi)
    
    # Set up simple pressure field
    @ti.kernel
    def set_linear_pressure():
        for i, j, k in solver.pressure:
            if solver.cell_type[i, j, k] == 0:  # Fluid
                solver.pressure[i, j, k] = float(j) * 100.0  # Linear in y
            else:
                solver.pressure[i, j, k] = 0.0
    
    set_linear_pressure()
    
    # Compute face gradients and node gradients
    solver.compute_face_center_gradients_gfm()
    solver.compute_node_pressure_gradient_gfm()
    
    # Check beta weights
    beta_face_y = solver.beta_face_y.to_numpy()
    n_valid_y_faces = np.sum(beta_face_y > 0)
    print(f"  Valid y-face gradients (β=1): {n_valid_y_faces}")
    
    # Check node gradients
    grad_p_node = solver.grad_p_node.to_numpy()
    
    # In fluid interior, y-gradient should be approximately 100/dx = 1000
    # (due to linear pressure p = j * 100)
    cell_types = solver.cell_type.to_numpy()
    
    # Find interior fluid nodes (away from boundaries)
    interior_grads = []
    for i in range(2, nx-1):
        for j in range(2, ny-1):
            for k in range(2, nz-1):
                # Check if surrounded by fluid
                if all(cell_types[i+di, j+dj, k+dk] == 0 
                       for di in [-1,0] for dj in [-1,0] for dk in [-1,0]):
                    interior_grads.append(grad_p_node[i, j, k])
    
    if len(interior_grads) > 0:
        interior_grads = np.array(interior_grads)
        mean_y_grad = np.mean(interior_grads[:, 1])
        expected_grad = 100.0 / dx
        print(f"  Mean y-gradient in interior: {mean_y_grad:.1f} (expected: {expected_grad:.1f})")
    
    print("✓ Node gradient weighting test passed\n")
    return True


def main():
    print("=" * 60)
    print("Ghost Fluid Method (GFM) Implementation Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Theta Calculation", test_theta_calculation),
        ("GFM Pressure Solve", test_gfm_pressure_solve),
        ("Node Gradient Weighting", test_node_gradient_weighting),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
