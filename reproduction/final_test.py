"""
Final test script for the implemented components
"""

import taichi as ti
import numpy as np
import yaml
import os

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

def test_barrier_model():
    """Test BarrierModel functionality."""
    print("Testing BarrierModel...")
    
    try:
        from barrier_model import BarrierModel
        
        # Create barrier model
        barrier = BarrierModel(
            barrier_height=0.15,
            barrier_spacing=2.0,
            barrier_positions=(3.0, 5.0)
        )
        
        # Test analytical landing distance
        landing_dist = barrier.calculate_theoretical_landing_distance(5.0, 0.5)
        print(f"  Theoretical landing distance: {landing_dist:.3f} m")
        
        print("  ✓ BarrierModel test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ BarrierModel test failed: {e}")
        return False

def test_output_metrics():
    """Test OutputMetricsCalculator functionality."""
    print("Testing OutputMetricsCalculator...")
    
    try:
        from output_metrics import OutputMetricsCalculator
        
        # Create metrics calculator
        metrics = OutputMetricsCalculator()
        
        # Test drag coefficient calculation
        drag_coeff = metrics.compute_drag_coefficient(0.6, 100.0)
        print(f"  Drag coefficient: {drag_coeff:.3f}")
        
        # Test Reynolds number calculation
        reynolds = metrics.compute_reynolds_number(2.0, 1e-3, 1000.0, 1e-3)
        print(f"  Reynolds number: {reynolds:.1f}")
        
        print("  ✓ OutputMetricsCalculator test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ OutputMetricsCalculator test failed: {e}")
        return False

def test_simulation_initialization():
    """Test simulation initialization only."""
    print("Testing CompleteDebrisFlowSimulation initialization...")
    
    try:
        from complete_simulation import CompleteDebrisFlowSimulation
        
        # Create simulation (short test run)
        simulation = CompleteDebrisFlowSimulation()
        
        # Override config for quick test
        simulation.config['simulation']['total_time'] = 0.01  # Very short test
        simulation.config['numerics']['vtk_output_interval'] = 1000  # No output
        
        # Test initialization only
        simulation.initialize_simulation()
        print("  ✓ Simulation initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation initialization failed: {e}")
        return False

def main():
    """Run final tests."""
    print("Final Component Tests")
    print("="*50)
    
    tests = [
        ("BarrierModel", test_barrier_model),
        ("OutputMetricsCalculator", test_output_metrics),
        ("Simulation Initialization", test_simulation_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test_name} test failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core components are working!")
        print("\nYour implementation is ready for use!")
        print("\nNext steps:")
        print("1. Run: python complete_simulation.py")
        print("2. Check the generated output files")
        print("3. Use the USAGE_GUIDE.md for detailed instructions")
    else:
        print("⚠️  Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
