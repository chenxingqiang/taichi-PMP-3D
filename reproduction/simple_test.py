"""
Simple test script for new components
"""

import taichi as ti
import numpy as np
import yaml
import os

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

def test_barrier_model_simple():
    """Simple test for BarrierModel."""
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

def test_output_metrics_simple():
    """Simple test for OutputMetricsCalculator."""
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

def main():
    """Run simple tests."""
    print("Simple Component Tests")
    print("="*40)
    
    tests = [
        ("BarrierModel", test_barrier_model_simple),
        ("OutputMetricsCalculator", test_output_metrics_simple)
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
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple tests passed!")
    else:
        print("⚠️  Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
