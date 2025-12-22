# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Simple test runner for the iMPM implementation

This script performs basic tests to verify the implementation:
1. Initialization tests
2. Simple dam break with reduced parameters
3. Pressure solver convergence test
4. Level set method test

Usage: python test_impm.py
"""

import sys
import os
import numpy as np
import taichi as ti

# Initialize Taichi with CPU backend for testing
ti.init(arch=ti.cpu, default_fp=ti.f64)

from incompressible_mpm_solver import IncompressibleMPMSolver
from level_set_method import LevelSetMethod
from pcg_solver import PCGSolver

def test_initialization():
    """Test basic initialization"""
    print("Testing solver initialization...")

    try:
        # Small test case
        solver = IncompressibleMPMSolver(
            nx=16, ny=16, nz=16,
            dx=0.1,
            rho=1000.0,
            mu=1e-3,
            gamma=0.0,
            g=9.8,
            dt=1e-3,
            max_particles=1000
        )

        print("✓ Solver initialization successful")

        # Test particle initialization
        solver.initialize_particles_dam_break(
            x_min=0.0, x_max=0.3,
            y_min=0.0, y_max=0.5,
            z_min=0.0, z_max=0.3,
            ppc=8
        )

        n_particles = solver.n_particles[None]
        print(f"✓ Initialized {n_particles} particles")

        if n_particles > 0:
            positions, velocities = solver.export_particles_to_numpy()
            print(f"✓ Particle data export works: {len(positions)} particles")

        return True

    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        return False

def test_level_set():
    """Test level set method"""
    print("Testing level set method...")

    try:
        level_set = LevelSetMethod(16, 16, 16, 0.1)

        # Initialize as sphere
        level_set.initialize_sphere(0.8, 0.8, 0.8, 0.3)

        # Compute gradients
        level_set.compute_gradient()
        level_set.compute_curvature_least_squares()

        print("✓ Level set initialization and gradient computation successful")

        # Test evolution (dummy velocity field)
        dummy_velocity = ti.Vector.field(3, dtype=ti.f64, shape=(16, 16, 16))
        level_set.step(0.01, dummy_velocity)

        print("✓ Level set evolution step successful")
        return True

    except Exception as e:
        print(f"✗ Level set test failed: {e}")
        return False

def test_pcg_solver():
    """Test PCG solver"""
    print("Testing PCG solver...")

    try:
        # PCGSolver only takes grid dimensions and spacing
        pcg = PCGSolver(16, 16, 16, 0.1)

        # Create dummy divergence field
        div_field = ti.field(dtype=ti.f64, shape=(16, 16, 16))

        @ti.kernel
        def setup_dummy_divergence():
            for i, j, k in div_field:
                if 4 <= i < 12 and 4 <= j < 12 and 4 <= k < 12:
                    div_field[i, j, k] = 1.0
                    pcg.cell_type[i, j, k] = 0  # Fluid
                else:
                    div_field[i, j, k] = 0.0
                    pcg.cell_type[i, j, k] = 2  # Air

        setup_dummy_divergence()

        # Test PCG solve
        iterations = pcg.solve_pcg(div_field, max_iter=10, tol=1e-4)

        print(f"✓ PCG solver converged in {iterations} iterations")
        return True

    except Exception as e:
        print(f"✗ PCG solver test failed: {e}")
        return False

def test_simple_simulation():
    """Test a few simulation steps"""
    print("Testing simulation steps...")

    try:
        # Very small test case
        solver = IncompressibleMPMSolver(
            nx=8, ny=8, nz=8,
            dx=0.2,
            rho=1000.0,
            mu=1e-3,
            gamma=0.0,
            g=9.8,
            dt=1e-3,
            max_particles=200
        )

        # Initialize simple dam
        solver.initialize_particles_dam_break(
            x_min=0.0, x_max=0.4,
            y_min=0.0, y_max=0.6,
            z_min=0.0, z_max=0.4,
            ppc=8
        )

        # Initialize level set
        solver.level_set_method.initialize_box(0.0, 0.4, 0.0, 0.6, 0.0, 0.4)

        print(f"✓ Initialized test simulation with {solver.n_particles[None]} particles")

        # Run a few steps
        for step in range(3):
            try:
                iterations = solver.step()
                solver.compute_statistics()

                ke = solver.total_kinetic_energy[None]
                max_vel = solver.max_velocity[None]

                print(f"  Step {step+1}: KE={ke:.2e}, Max vel={max_vel:.3f}, PCG iters={iterations}")

            except Exception as e:
                print(f"  ✗ Step {step+1} failed: {e}")
                return False

        print("✓ Simulation steps completed successfully")
        return True

    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def test_3d_axis_mapping():
    """Test that axis mapping is correct: y=height, z=width"""
    print("Testing 3D axis mapping...")
    
    try:
        # Create solver with explicit 3D dimensions
        # x=length, y=height, z=width
        solver = IncompressibleMPMSolver(
            nx=16, ny=8, nz=4,  # Different in each dimension
            dx=0.1,
            rho=1000.0,
            mu=1e-3,
            g=9.8,
            dt=1e-3,
            max_particles=500
        )
        
        # Verify grid dimensions
        assert solver.nx == 16, f"Expected nx=16, got {solver.nx}"
        assert solver.ny == 8, f"Expected ny=8, got {solver.ny}"
        assert solver.nz == 4, f"Expected nz=4, got {solver.nz}"
        print("✓ Grid dimensions correct: (16, 8, 4)")
        
        # Verify gravity direction (should be negative y)
        gravity = solver.g.to_numpy()
        assert gravity[0] == 0.0, "Gravity x-component should be 0"
        assert gravity[1] < 0, "Gravity y-component should be negative (downward)"
        assert gravity[2] == 0.0, "Gravity z-component should be 0"
        print(f"✓ Gravity direction correct: {gravity}")
        
        # Initialize particles in a box spanning width (z) dimension
        solver.initialize_particles_dam_break(
            x_min=0.0, x_max=0.4,  # Length
            y_min=0.0, y_max=0.3,  # Height
            z_min=0.0, z_max=0.4,  # Width (full span)
            ppc=4
        )
        
        positions, _ = solver.export_particles_to_numpy()
        
        # Verify particles span the z (width) dimension
        z_range = positions[:, 2].max() - positions[:, 2].min()
        assert z_range > 0.3, f"Particles should span z-width, got range {z_range}"
        print(f"✓ Particles span z-width: {z_range:.3f}m")
        
        # Verify flow depth uses y-axis (height)
        y_range = positions[:, 1].max() - positions[:, 1].min()
        assert y_range >= 0.15, f"Flow height should use y-axis, got range {y_range}"
        print(f"✓ Flow height along y-axis: {y_range:.3f}m")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ 3D axis mapping test failed: {e}")
        return False

def test_3d_width_dimension():
    """Test simulation with explicit width dimension (nz > 1)"""
    print("Testing 3D width dimension simulation...")
    
    try:
        # Simulate with 2m physical width
        width = 2.0
        dx = 0.5  # Coarse for fast test
        nz = int(width / dx)  # Should give nz = 4
        
        solver = IncompressibleMPMSolver(
            nx=8, ny=8, nz=nz,
            dx=dx,
            rho=1000.0,
            mu=1e-3,
            g=9.8,
            dt=1e-3,
            max_particles=300
        )
        
        print(f"✓ Created solver with nz={nz} for {width}m width")
        
        # Initialize particles
        solver.initialize_particles_dam_break(
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=width,  # Full width
            ppc=4
        )
        
        # Initialize level set
        solver.level_set_method.initialize_box(0.0, 1.0, 0.0, 1.0, 0.0, width)
        
        n_particles = solver.n_particles[None]
        print(f"✓ Initialized {n_particles} particles across {width}m width")
        
        # Run 5 steps
        for step in range(5):
            iterations = solver.step()
            
        solver.compute_statistics()
        
        # Verify particles still exist
        positions, velocities = solver.export_particles_to_numpy()
        assert len(positions) > 0, "Particles should still exist after simulation"
        
        # Verify z-dimension is utilized
        z_extent = positions[:, 2].max() - positions[:, 2].min()
        assert z_extent > 0.5, f"Z-extent should be substantial, got {z_extent}"
        
        print(f"✓ 3D simulation completed: {len(positions)} particles, z-extent={z_extent:.2f}m")
        return True
        
    except Exception as e:
        print(f"✗ 3D width dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("iMPM Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Initialization", test_initialization),
        ("Level Set Method", test_level_set),
        ("PCG Solver", test_pcg_solver),
        ("Simple Simulation", test_simple_simulation),
        ("3D Axis Mapping", test_3d_axis_mapping),
        ("3D Width Dimension", test_3d_width_dimension),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)

        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! iMPM implementation is working.")
        print("\nYou can now try running the full dam break example:")
        print("python examples/dam_break_3d.py")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    print("=" * 60)

if __name__ == "__main__":
    main()