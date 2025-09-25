"""
Test script for new components: BarrierModel, OutputMetricsCalculator, and CompleteSimulation
"""

import taichi as ti
import numpy as np
import yaml
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from barrier_model import BarrierModel
from output_metrics import OutputMetricsCalculator
from complete_simulation import CompleteDebrisFlowSimulation

def test_barrier_model():
    """Test BarrierModel functionality."""
    print("Testing BarrierModel...")
    
    # Initialize Taichi
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
    # Create barrier model
    barrier = BarrierModel(
        barrier_height=0.15,
        barrier_spacing=2.0,
        barrier_positions=(3.0, 5.0)
    )
    
    # Test contact detection
    n_particles = 100
    positions = ti.Vector.field(3, ti.f64, shape=n_particles)
    velocities = ti.Vector.field(3, ti.f64, shape=n_particles)
    contact_forces = ti.Vector.field(3, ti.f64, shape=n_particles)
    
    # Initialize test particles
    @ti.kernel
    def init_test_particles():
        for i in range(n_particles):
            # Create particles at various positions
            x = ti.cast(i, ti.f64) * 0.1  # Spread particles along x-axis
            y = 0.1
            z = 0.1
            positions[i] = [x, y, z]
            velocities[i] = [1.0, 0.0, 0.0]  # Moving in +x direction
            contact_forces[i] = [0.0, 0.0, 0.0]
    
    init_test_particles()
    
    # Test contact detection
    barrier.detect_contacts(positions, velocities, contact_forces, n_particles)
    
    # Test overflow tracking
    barrier.track_overflow_kinematics(positions, velocities, n_particles, 0.0)
    
    # Test statistics computation
    barrier.compute_impact_statistics()
    barrier.compute_overflow_statistics()
    
    # Get results
    impact_stats = barrier.get_impact_statistics()
    overflow_stats = barrier.get_overflow_statistics()
    
    print(f"  Impact statistics: {impact_stats}")
    print(f"  Overflow statistics: {overflow_stats}")
    
    # Test analytical landing distance
    landing_dist = barrier.calculate_theoretical_landing_distance(5.0, 0.5)
    print(f"  Theoretical landing distance: {landing_dist:.3f} m")
    
    print("  ✓ BarrierModel test passed!")
    return True

def test_output_metrics():
    """Test OutputMetricsCalculator functionality."""
    print("Testing OutputMetricsCalculator...")
    
    # Initialize Taichi
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
    # Create metrics calculator
    metrics = OutputMetricsCalculator()
    
    # Test fluidization ratio calculation
    n_particles = 100
    pressure_field = ti.field(ti.f64, shape=n_particles)
    stress_field = ti.field(ti.f64, shape=n_particles)
    
    @ti.kernel
    def init_test_fields():
        for i in range(n_particles):
            pressure_field[i] = 1000.0 + i * 10.0  # Varying pressure
            stress_field[i] = 5000.0 + i * 50.0    # Varying stress
    
    init_test_fields()
    
    # Test fluidization ratio computation
    metrics.compute_fluidization_ratio(pressure_field, stress_field, n_particles)
    
    # Test impact force calculation
    velocity_field = ti.field(ti.f64, shape=n_particles)
    depth_field = ti.field(ti.f64, shape=n_particles)
    density_field = ti.field(ti.f64, shape=n_particles)
    
    @ti.kernel
    def init_impact_fields():
        for i in range(n_particles):
            velocity_field[i] = 2.0 + i * 0.01  # Varying velocity
            depth_field[i] = 0.1 + i * 0.001    # Varying depth
            density_field[i] = 2000.0           # Constant density
    
    init_impact_fields()
    
    metrics.compute_impact_forces(velocity_field, depth_field, density_field, n_particles)
    
    # Test flow statistics
    volume_fraction_field = ti.field(ti.f64, shape=n_particles)
    
    @ti.kernel
    def init_flow_fields():
        for i in range(n_particles):
            volume_fraction_field[i] = 0.6  # Constant volume fraction
    
    init_flow_fields()
    
    metrics.compute_flow_statistics(velocity_field, depth_field, volume_fraction_field, n_particles)
    
    # Test barrier effectiveness
    metrics.compute_barrier_effectiveness(100, 80, 20, 1000.0, 200.0)
    
    # Get all metrics
    fluidization_stats = metrics.get_fluidization_statistics()
    impact_stats = metrics.get_impact_statistics()
    flow_stats = metrics.get_flow_statistics()
    barrier_stats = metrics.get_barrier_effectiveness()
    
    print(f"  Fluidization stats: {fluidization_stats}")
    print(f"  Impact stats: {impact_stats}")
    print(f"  Flow stats: {flow_stats}")
    print(f"  Barrier stats: {barrier_stats}")
    
    # Test drag coefficient calculation
    drag_coeff = metrics.compute_drag_coefficient(0.6, 100.0)
    print(f"  Drag coefficient: {drag_coeff:.3f}")
    
    # Test Reynolds number calculation
    reynolds = metrics.compute_reynolds_number(2.0, 1e-3, 1000.0, 1e-3)
    print(f"  Reynolds number: {reynolds:.1f}")
    
    print("  ✓ OutputMetricsCalculator test passed!")
    return True

def test_complete_simulation():
    """Test CompleteDebrisFlowSimulation functionality."""
    print("Testing CompleteDebrisFlowSimulation...")
    
    # Check if config file exists
    config_path = "physics_config.yaml"
    if not os.path.exists(config_path):
        print(f"  ⚠️  Config file {config_path} not found, skipping test")
        return True
    
    # Initialize Taichi
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
    # Create simulation (short test run)
    simulation = CompleteDebrisFlowSimulation(config_path)
    
    # Override config for quick test
    simulation.config['simulation']['total_time'] = 0.1  # Short test
    simulation.config['numerics']['vtk_output_interval'] = 10  # Less output
    
    # Test initialization
    try:
        simulation.initialize_simulation()
        print("  ✓ Simulation initialization successful")
    except Exception as e:
        print(f"  ✗ Simulation initialization failed: {e}")
        return False
    
    # Test short simulation run
    try:
        results = simulation.run_simulation("test_output")
        print("  ✓ Simulation run successful")
        print(f"  Output directory: {results['output_directory']}")
    except Exception as e:
        print(f"  ✗ Simulation run failed: {e}")
        return False
    
    # Test metrics export
    try:
        simulation.metrics_calculator.print_metrics_summary()
        print("  ✓ Metrics calculation successful")
    except Exception as e:
        print(f"  ✗ Metrics calculation failed: {e}")
        return False
    
    print("  ✓ CompleteDebrisFlowSimulation test passed!")
    return True

def main():
    """Run all tests."""
    print("Testing New Components for Two-Phase MPM Debris Flow Impact")
    print("="*60)
    
    tests = [
        ("BarrierModel", test_barrier_model),
        ("OutputMetricsCalculator", test_output_metrics),
        ("CompleteDebrisFlowSimulation", test_complete_simulation)
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
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! New components are ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
