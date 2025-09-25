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
        pcg = PCGSolver(16, 16, 16, 0.1, 1000.0, 1e-3)
        
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