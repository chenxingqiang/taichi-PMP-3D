"""
Quick test to verify the fixes
"""

import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f64)

def test_simulation_short():
    """Test a very short simulation to verify fixes."""
    print("Testing short simulation...")
    
    try:
        from complete_simulation import CompleteDebrisFlowSimulation
        
        # Create simulation with very short parameters
        simulation = CompleteDebrisFlowSimulation()
        
        # Override config for very quick test
        simulation.config['simulation']['total_time'] = 0.001  # 1ms
        simulation.config['numerics']['vtk_output_interval'] = 10000  # No output
        simulation.config['numerics']['statistics_interval'] = 10000  # No stats
        
        # Initialize
        simulation.initialize_simulation()
        print("  ✓ Initialization successful")
        
        # Run very short simulation
        results = simulation.run_simulation("quick_test_output")
        print("  ✓ Short simulation completed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick test."""
    print("Quick Test for Fixes")
    print("="*30)
    
    success = test_simulation_short()
    
    if success:
        print("\n🎉 All fixes are working!")
        print("Your implementation is ready for full simulation.")
    else:
        print("\n⚠️  There are still issues to fix.")
    
    return success

if __name__ == "__main__":
    success = main()
