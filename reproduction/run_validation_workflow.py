"""
Complete Validation Workflow
Run simulation, extract data, and generate validation plots
"""

import os
import sys
import time
from datetime import datetime

def run_complete_validation_workflow():
    """Run complete validation workflow"""
    print("="*60)
    print("Two-Phase MPM Model Validation Workflow")
    print("Based on Ng et al. (2023)")
    print("="*60)

    # Step 1: Run simulation with data extraction
    print("\nStep 1: Running simulation with data extraction...")
    try:
        from data_extractor import extract_data_from_simulation
        validation_data, data_dir = extract_data_from_simulation()
        print(f"✓ Simulation completed and data extracted to: {data_dir}")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return False

    # Step 2: Generate validation plots from real data
    print("\nStep 2: Generating validation plots from real data...")
    try:
        from real_validation_plots import generate_real_validation_plots
        plot_dir = generate_real_validation_plots(data_dir)
        print(f"✓ Validation plots generated and saved to: {plot_dir}")
    except Exception as e:
        print(f"✗ Plot generation failed: {e}")
        return False

    # Step 3: Generate summary report
    print("\nStep 3: Generating validation summary...")
    try:
        generate_validation_summary(data_dir, plot_dir, validation_data)
        print("✓ Validation summary generated")
    except Exception as e:
        print(f"✗ Summary generation failed: {e}")
        return False

    print("\n" + "="*60)
    print("🎉 VALIDATION WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Plots directory: {plot_dir}")
    print("="*60)

    return True

def generate_validation_summary(data_dir, plot_dir, validation_data):
    """Generate validation summary report"""
    summary_file = f"{data_dir}/validation_summary.md"

    with open(summary_file, 'w') as f:
        f.write("# Two-Phase MPM Model Validation Summary\n\n")
        f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Simulation Results\n\n")

        if validation_data and 'time_series' in validation_data:
            time_series = validation_data['time_series']
            f.write(f"- **Total simulation time**: {max(time_series['time']):.2f} seconds\n")
            f.write(f"- **Data points extracted**: {len(time_series['time'])}\n")

            if 'max_impact' in time_series:
                max_impact = max(time_series['max_impact'])
                f.write(f"- **Maximum impact force**: {max_impact:.2f} N\n")

            if 'max_velocity' in time_series:
                max_velocity = max(time_series['max_velocity'])
                f.write(f"- **Maximum flow velocity**: {max_velocity:.2f} m/s\n")

        f.write("\n## Generated Files\n\n")
        f.write("### Data Files\n")
        f.write("- `time_series_data.csv` - Time series data\n")
        f.write("- `flow_profile_*.csv` - Flow profiles at key times\n")
        f.write("- `velocity_field_*.npz` - Velocity fields at key times\n")
        f.write("- `config.yaml` - Simulation configuration\n\n")

        f.write("### Validation Plots\n")
        f.write("- `1_real_flow_morphology_comparison.png` - Flow morphology evolution\n")
        f.write("- `2_real_impact_force_time_series.png` - Impact force evolution\n")
        f.write("- `3_real_velocity_field.png` - Velocity field analysis\n")
        f.write("- `4_flow_statistics.png` - Flow statistics over time\n\n")

        f.write("## Validation Status\n\n")
        f.write("✅ **Simulation completed successfully**\n")
        f.write("✅ **Data extraction completed**\n")
        f.write("✅ **Validation plots generated**\n")
        f.write("✅ **Ready for paper validation**\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review generated validation plots\n")
        f.write("2. Compare with experimental data from Ng et al. (2023)\n")
        f.write("3. Perform parameter sensitivity studies\n")
        f.write("4. Generate additional validation metrics as needed\n\n")

    print(f"Validation summary saved to: {summary_file}")

def run_quick_validation():
    """Run a quick validation with shorter simulation time"""
    print("Running quick validation (2 seconds simulation)...")

    # Modify config for quick run
    import yaml
    config_file = "physics_config.yaml"

    # Backup original config
    with open(config_file, 'r') as f:
        original_config = yaml.safe_load(f)

    # Modify for quick run
    quick_config = original_config.copy()
    quick_config['simulation']['total_time'] = 2.0
    quick_config['numerics']['statistics_interval'] = 0.05
    quick_config['numerics']['vtk_output_interval'] = 0.2

    # Save quick config
    with open("physics_config_quick.yaml", 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)

    try:
        # Run with quick config
        from data_extractor import SimulationDataExtractor
        extractor = SimulationDataExtractor("physics_config_quick.yaml")

        from complete_simulation import CompleteDebrisFlowSimulation
        simulation = CompleteDebrisFlowSimulation("physics_config_quick.yaml")
        simulation.initialize_simulation()

        validation_data = extractor.extract_from_simulation(simulation)
        data_dir = extractor.save_validation_data(validation_data)

        from real_validation_plots import generate_real_validation_plots
        plot_dir = generate_real_validation_plots(data_dir)

        print(f"✓ Quick validation completed!")
        print(f"Data: {data_dir}")
        print(f"Plots: {plot_dir}")

        return True

    except Exception as e:
        print(f"✗ Quick validation failed: {e}")
        return False

    finally:
        # Restore original config
        with open(config_file, 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)

        # Clean up quick config
        if os.path.exists("physics_config_quick.yaml"):
            os.remove("physics_config_quick.yaml")

if __name__ == "__main__":
    print("Two-Phase MPM Validation Workflow")
    print("Choose an option:")
    print("1. Run complete validation workflow")
    print("2. Run quick validation (2 seconds)")
    print("3. Exit")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        success = run_complete_validation_workflow()
    elif choice == "2":
        success = run_quick_validation()
    elif choice == "3":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)

    if success:
        print("\n🎉 Validation workflow completed successfully!")
    else:
        print("\n❌ Validation workflow failed. Check error messages above.")
