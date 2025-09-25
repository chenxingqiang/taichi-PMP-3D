"""
Data Extractor for Validation Plots
Extract real simulation data for generating validation charts
"""

import taichi as ti
import numpy as np
import yaml
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

class SimulationDataExtractor:
    """Extract validation data from simulation results"""

    def __init__(self, config_path="physics_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize data storage
        self.time_history = []
        self.flow_data = {}
        self.impact_data = {}
        self.velocity_data = {}
        self.barrier_data = {}

        # Key time points for validation
        self.key_times = [0.0, 0.2, 0.4, 2.0]  # From VALID_FIG.md

    def extract_from_simulation(self, simulation):
        """Extract data from running simulation"""
        print("Extracting validation data from simulation...")

        # Get simulation parameters
        total_time = self.config['simulation']['total_time']
        dt = self.config['numerics']['max_timestep']
        output_interval = self.config['numerics']['statistics_interval']

        total_steps = int(total_time / dt)
        output_steps = int(output_interval / dt)

        # Extract data at key time points
        key_step_indices = [int(t / dt) for t in self.key_times if t <= total_time]

        for step in range(total_steps):
            current_time = step * dt

            # Run simulation step
            pcg_iterations = simulation.solver.step()

            # Extract data at output intervals
            if step % output_steps == 0:
                self._extract_timestep_data(simulation, current_time, step)

            # Extract data at key time points
            if step in key_step_indices:
                self._extract_key_timestep_data(simulation, current_time, step)

            # Check for convergence issues
            if pcg_iterations >= 200:
                print(f"Warning: PCG did not converge at step {step}, time {current_time:.3f}s")

        print(f"Data extraction completed. Extracted {len(self.time_history)} time points.")
        return self._compile_validation_data()

    def _extract_timestep_data(self, simulation, current_time, step):
        """Extract data for current timestep"""
        # Get particle data
        positions, velocities = simulation.solver.export_particles_to_numpy()

        if len(positions) > 0:
            # Flow morphology data
            flow_front = self._calculate_flow_front(positions)
            flow_height = self._calculate_flow_height(positions)

            # Velocity field data
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            mean_velocity = np.mean(velocity_magnitudes)
            max_velocity = np.max(velocity_magnitudes)

            # Impact force data
            impact_stats = simulation.barrier_model.get_impact_statistics()

            # Store data
            self.time_history.append(current_time)

            if 'flow_front' not in self.flow_data:
                self.flow_data['flow_front'] = []
                self.flow_data['flow_height'] = []

            self.flow_data['flow_front'].append(flow_front)
            self.flow_data['flow_height'].append(flow_height)

            if 'mean_velocity' not in self.velocity_data:
                self.velocity_data['mean_velocity'] = []
                self.velocity_data['max_velocity'] = []

            self.velocity_data['mean_velocity'].append(mean_velocity)
            self.velocity_data['max_velocity'].append(max_velocity)

            if 'total_impact' not in self.impact_data:
                self.impact_data['total_impact'] = []
                self.impact_data['max_impact'] = []

            self.impact_data['total_impact'].append(impact_stats['total_impact_force'])
            self.impact_data['max_impact'].append(impact_stats['max_impact_force'])

    def _extract_key_timestep_data(self, simulation, current_time, step):
        """Extract detailed data at key validation time points"""
        positions, velocities = simulation.solver.export_particles_to_numpy()

        if len(positions) > 0:
            # Detailed flow profile at key times
            flow_profile = self._calculate_detailed_flow_profile(positions)

            # Velocity field at key times
            velocity_field = self._calculate_velocity_field(positions, velocities)

            # Store key time data
            key_time = f"t_{current_time:.1f}s"

            if key_time not in self.flow_data:
                self.flow_data[key_time] = {}

            self.flow_data[key_time]['profile'] = flow_profile
            self.flow_data[key_time]['velocity_field'] = velocity_field

    def _calculate_flow_front(self, positions):
        """Calculate flow front position"""
        if len(positions) == 0:
            return 0.0
        return np.max(positions[:, 0])  # Maximum x-coordinate

    def _calculate_flow_height(self, positions):
        """Calculate flow height"""
        if len(positions) == 0:
            return 0.0
        return np.max(positions[:, 1]) - np.min(positions[:, 1])  # Height range

    def _calculate_detailed_flow_profile(self, positions):
        """Calculate detailed flow profile for morphology analysis"""
        if len(positions) == 0:
            return np.array([])

        # Create normalized profile
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Normalize coordinates
        x_norm = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords) + 1e-8)
        y_norm = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords) + 1e-8)

        # Create profile bins
        n_bins = 100
        x_bins = np.linspace(0, 1, n_bins)
        y_profile = np.zeros(n_bins)

        for i, x_bin in enumerate(x_bins):
            # Find particles in this x-bin
            mask = np.abs(x_norm - x_bin) < 0.01
            if np.any(mask):
                y_profile[i] = np.max(y_norm[mask])

        return {'x': x_bins, 'y': y_profile}

    def _calculate_velocity_field(self, positions, velocities):
        """Calculate velocity field for streamlines"""
        if len(positions) == 0:
            return {'x': np.array([]), 'y': np.array([]), 'u': np.array([]), 'v': np.array([])}

        # Create grid for velocity field
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])

        # Normalize coordinates
        x_norm = (positions[:, 0] - x_min) / (x_max - x_min + 1e-8)
        y_norm = (positions[:, 1] - y_min) / (y_max - y_min + 1e-8)

        # Create velocity grid
        nx, ny = 50, 30
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate velocities to grid
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(nx):
            for j in range(ny):
                # Find nearby particles
                distances = np.sqrt((x_norm - x_grid[i])**2 + (y_norm - y_grid[j])**2)
                nearby_mask = distances < 0.1

                if np.any(nearby_mask):
                    # Average velocity of nearby particles
                    U[j, i] = np.mean(velocities[nearby_mask, 0])
                    V[j, i] = np.mean(velocities[nearby_mask, 1])

        return {'x': X, 'y': Y, 'u': U, 'v': V}

    def _compile_validation_data(self):
        """Compile extracted data into validation format"""
        validation_data = {
            'time_series': {
                'time': self.time_history,
                'flow_front': self.flow_data.get('flow_front', []),
                'flow_height': self.flow_data.get('flow_height', []),
                'mean_velocity': self.velocity_data.get('mean_velocity', []),
                'max_velocity': self.velocity_data.get('max_velocity', []),
                'total_impact': self.impact_data.get('total_impact', []),
                'max_impact': self.impact_data.get('max_impact', [])
            },
            'key_times': {},
            'config': self.config
        }

        # Compile key time data
        for key in self.flow_data:
            if key.startswith('t_'):
                validation_data['key_times'][key] = self.flow_data[key]

        return validation_data

    def save_validation_data(self, validation_data, output_dir="simulation_output"):
        """Save validation data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_dir = f"{output_dir}/validation_data_{timestamp}"
        os.makedirs(data_dir, exist_ok=True)

        # Save time series data
        time_series_df = pd.DataFrame(validation_data['time_series'])
        time_series_df.to_csv(f"{data_dir}/time_series_data.csv", index=False)

        # Save key time data
        for key_time, data in validation_data['key_times'].items():
            if 'profile' in data:
                profile_df = pd.DataFrame(data['profile'])
                profile_df.to_csv(f"{data_dir}/flow_profile_{key_time}.csv", index=False)

            if 'velocity_field' in data:
                vf_data = data['velocity_field']
                # Save velocity field as numpy arrays
                np.savez(f"{data_dir}/velocity_field_{key_time}.npz",
                        x=vf_data['x'], y=vf_data['y'],
                        u=vf_data['u'], v=vf_data['v'])

        # Save configuration
        with open(f"{data_dir}/config.yaml", 'w') as f:
            yaml.dump(validation_data['config'], f, default_flow_style=False)

        print(f"Validation data saved to: {data_dir}")
        return data_dir

def extract_data_from_simulation():
    """Main function to extract data from simulation"""
    print("Starting data extraction from simulation...")

    # Initialize Taichi
    ti.init(arch=ti.gpu, device_memory_fraction=0.8, device_memory_GB=20)

    # Create data extractor
    extractor = SimulationDataExtractor()

    # Import and create simulation
    from complete_simulation import CompleteDebrisFlowSimulation
    simulation = CompleteDebrisFlowSimulation()

    # Initialize simulation
    simulation.initialize_simulation()

    # Extract data
    validation_data = extractor.extract_from_simulation(simulation)

    # Save data
    data_dir = extractor.save_validation_data(validation_data)

    print("Data extraction completed successfully!")
    return validation_data, data_dir

if __name__ == "__main__":
    validation_data, data_dir = extract_data_from_simulation()
    print(f"Validation data extracted and saved to: {data_dir}")
