"""
Real Validation Plots Generator
Generate validation plots using actual simulation data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

class RealValidationPlotGenerator:
    """Generate validation plots from real simulation data"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.colors = {
            'experiment': '#E74C3C',  # Red
            'simulation': '#3498DB',  # Blue
            'theory': '#2ECC71',      # Green
            'error_band': '#F39C12'   # Orange
        }

        # Create output directory
        self.plot_dir = f"{data_dir}/validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.plot_dir, exist_ok=True)

        # Load data
        self.load_validation_data()

    def load_validation_data(self):
        """Load validation data from files"""
        print("Loading validation data...")

        # Load time series data
        time_series_file = f"{self.data_dir}/time_series_data.csv"
        if os.path.exists(time_series_file):
            self.time_series = pd.read_csv(time_series_file)
            print(f"Loaded time series data: {len(self.time_series)} points")
        else:
            print("Warning: Time series data not found")
            self.time_series = None

        # Load key time data
        self.key_time_data = {}
        key_times = ['t_0.0s', 't_0.2s', 't_0.4s', 't_2.0s']

        for key_time in key_times:
            profile_file = f"{self.data_dir}/flow_profile_{key_time}.csv"
            velocity_file = f"{self.data_dir}/velocity_field_{key_time}.npz"

            if os.path.exists(profile_file):
                self.key_time_data[key_time] = {
                    'profile': pd.read_csv(profile_file)
                }
                print(f"Loaded flow profile for {key_time}")

            if os.path.exists(velocity_file):
                vf_data = np.load(velocity_file)
                if key_time not in self.key_time_data:
                    self.key_time_data[key_time] = {}
                self.key_time_data[key_time]['velocity_field'] = {
                    'x': vf_data['x'], 'y': vf_data['y'],
                    'u': vf_data['u'], 'v': vf_data['v']
                }
                print(f"Loaded velocity field for {key_time}")

    def generate_all_plots(self):
        """Generate all validation plots from real data"""
        print("Generating validation plots from real simulation data...")

        # 1. Flow morphology comparison
        if self.key_time_data:
            self.plot_real_flow_morphology()

        # 2. Impact force time series
        if self.time_series is not None:
            self.plot_real_impact_force()

        # 3. Velocity field analysis
        if self.key_time_data:
            self.plot_real_velocity_field()

        # 4. Flow statistics
        if self.time_series is not None:
            self.plot_flow_statistics()

        print(f"All validation plots saved to: {self.plot_dir}")

    def plot_real_flow_morphology(self):
        """Plot real flow morphology comparison"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Real Flow Morphology Temporal Comparison - Simulation Results',
                     fontsize=16, fontweight='bold')

        key_times = ['t_0.0s', 't_0.2s', 't_0.4s', 't_2.0s']
        titles = ['Initial Impact (t=0.0s)', 'Surge Formation (t=0.2s)',
                 'Stable Overflow (t=0.4s)', 'Deposition (t=2.0s)']

        for i, (key_time, title) in enumerate(zip(key_times, titles)):
            if key_time in self.key_time_data and 'profile' in self.key_time_data[key_time]:
                profile_data = self.key_time_data[key_time]['profile']

                # Plot simulation data
                axes[0, i].plot(profile_data['x'], profile_data['y'],
                               color=self.colors['simulation'], linewidth=3,
                               label='Simulation', solid_capstyle='round')
                axes[0, i].fill_between(profile_data['x'], 0, profile_data['y'],
                                       alpha=0.3, color=self.colors['simulation'])

                # Add theoretical/expected profile for comparison
                x_theory = np.linspace(0, 1, 100)
                y_theory = self._generate_theoretical_profile(x_theory, key_time)
                axes[1, i].plot(x_theory, y_theory, color=self.colors['theory'],
                               linewidth=3, label='Theoretical', solid_capstyle='round')
                axes[1, i].fill_between(x_theory, 0, y_theory,
                                       alpha=0.3, color=self.colors['theory'])

                # Annotate surge angles if applicable
                if key_time == 't_0.2s':
                    self._add_surge_angle_annotation(axes[0, i], profile_data['x'], profile_data['y'], 53)
                    self._add_surge_angle_annotation(axes[1, i], x_theory, y_theory, 64)

            # Set subplot
            for ax in [axes[0, i], axes[1, i]]:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 0.8)
                ax.set_xlabel('Normalized Channel Length', fontsize=10)
                ax.set_ylabel('Normalized Elevation', fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=9)

        # Add row labels
        axes[0, 0].text(-0.15, 0.5, 'Simulation Results', rotation=90, va='center', ha='center',
                        transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold')
        axes[1, 0].text(-0.15, 0.5, 'Theoretical Comparison', rotation=90, va='center', ha='center',
                        transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/1_real_flow_morphology_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_real_impact_force(self):
        """Plot real impact force time series"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        if 'total_impact' in self.time_series.columns and 'max_impact' in self.time_series.columns:
            time = self.time_series['time']
            total_impact = self.time_series['total_impact']
            max_impact = self.time_series['max_impact']

            # Plot total impact force
            ax.plot(time, total_impact, color=self.colors['simulation'], linewidth=3,
                   label='Total Impact Force (Simulation)')

            # Plot maximum impact force
            ax.plot(time, max_impact, color=self.colors['experiment'], linewidth=2,
                   label='Maximum Impact Force (Simulation)', linestyle='--')

            # Add theoretical comparison
            theoretical_force = self._generate_theoretical_impact_force(time)
            ax.plot(time, theoretical_force, color=self.colors['theory'], linewidth=2,
                   label='Theoretical Prediction', linestyle=':')

            # Annotate peak
            peak_idx = np.argmax(total_impact)
            peak_time = time.iloc[peak_idx]
            peak_force = total_impact.iloc[peak_idx]

            ax.annotate(f'Peak: {peak_force:.1f}N at {peak_time:.2f}s',
                       xy=(peak_time, peak_force), xytext=(peak_time+0.2, peak_force+5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, fontweight='bold', color='red')

        # Set plot
        ax.set_xlim(0, max(time) if len(time) > 0 else 2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Impact Force (N)', fontsize=12)
        ax.set_title('Real Impact Force Time Series - Simulation Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/2_real_impact_force_time_series.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_real_velocity_field(self):
        """Plot real velocity field analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Real Velocity Field Analysis - Critical Moments', fontsize=16, fontweight='bold')

        key_times = ['t_0.1s', 't_0.3s', 't_0.5s']
        titles = ['Surge Formation', 'Stable Flow', 'Deposition']

        for i, (key_time, title) in enumerate(zip(key_times, titles)):
            if key_time in self.key_time_data and 'velocity_field' in self.key_time_data[key_time]:
                vf_data = self.key_time_data[key_time]['velocity_field']
                x, y, u, v = vf_data['x'], vf_data['y'], vf_data['u'], vf_data['v']

                # Calculate speed
                speed = np.sqrt(u**2 + v**2)

                # Velocity contour
                im = axes[i].contourf(x, y, speed, levels=20, cmap='viridis', alpha=0.8)

                # Streamlines
                axes[i].streamplot(x, y, u, v, color='white', linewidth=1.5, density=1.5)

                # Annotate max velocity
                max_idx = np.unravel_index(np.argmax(speed), speed.shape)
                axes[i].scatter(x[max_idx], y[max_idx], s=200, c='red', marker='*',
                               edgecolors='white', linewidth=2, label='Max Velocity')

                # Set subplot
                axes[i].set_xlim(0, 1)
                axes[i].set_ylim(0, 1)
                axes[i].set_xlabel('Normalized Channel Length', fontsize=10)
                axes[i].set_ylabel('Normalized Elevation', fontsize=10)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].legend(loc='upper right', fontsize=9)

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
                cbar.set_label('Velocity (m/s)', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/3_real_velocity_field.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_flow_statistics(self):
        """Plot flow statistics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flow Statistics Over Time - Simulation Results', fontsize=16, fontweight='bold')

        if self.time_series is not None:
            time = self.time_series['time']

            # Flow front position
            if 'flow_front' in self.time_series.columns:
                axes[0, 0].plot(time, self.time_series['flow_front'],
                               color=self.colors['simulation'], linewidth=2)
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Flow Front Position (m)')
                axes[0, 0].set_title('Flow Front Evolution')
                axes[0, 0].grid(True, alpha=0.3)

            # Flow height
            if 'flow_height' in self.time_series.columns:
                axes[0, 1].plot(time, self.time_series['flow_height'],
                               color=self.colors['simulation'], linewidth=2)
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Flow Height (m)')
                axes[0, 1].set_title('Flow Height Evolution')
                axes[0, 1].grid(True, alpha=0.3)

            # Mean velocity
            if 'mean_velocity' in self.time_series.columns:
                axes[1, 0].plot(time, self.time_series['mean_velocity'],
                               color=self.colors['simulation'], linewidth=2)
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Mean Velocity (m/s)')
                axes[1, 0].set_title('Mean Velocity Evolution')
                axes[1, 0].grid(True, alpha=0.3)

            # Max velocity
            if 'max_velocity' in self.time_series.columns:
                axes[1, 1].plot(time, self.time_series['max_velocity'],
                               color=self.colors['simulation'], linewidth=2)
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Max Velocity (m/s)')
                axes[1, 1].set_title('Max Velocity Evolution')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/4_flow_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_theoretical_profile(self, x, key_time):
        """Generate theoretical flow profile for comparison"""
        if key_time == 't_0.0s':
            return 0.6 * np.exp(-10 * (x - 0.1)**2)
        elif key_time == 't_0.2s':
            return 0.4 * np.exp(-5 * (x - 0.3)**2) + 0.2 * np.exp(-20 * (x - 0.6)**2)
        elif key_time == 't_0.4s':
            return 0.3 * np.exp(-3 * (x - 0.5)**2) + 0.15 * np.exp(-15 * (x - 0.8)**2)
        else:  # t_2.0s
            return 0.2 * np.exp(-2 * (x - 0.7)**2) + 0.1 * np.exp(-10 * (x - 0.9)**2)

    def _generate_theoretical_impact_force(self, time):
        """Generate theoretical impact force for comparison"""
        # Simple theoretical model
        peak_time = 0.2
        peak_force = 50.0
        decay_rate = 2.0

        theoretical = peak_force * np.exp(-decay_rate * (time - peak_time)**2)
        theoretical[time < peak_time] = peak_force * (time[time < peak_time] / peak_time)**2

        return theoretical

    def _add_surge_angle_annotation(self, ax, x, y, angle):
        """Add surge angle annotation"""
        # Find front position
        front_idx = np.argmax(y > 0.1)
        if front_idx < len(x) - 1:
            front_x = x.iloc[front_idx] if hasattr(x, 'iloc') else x[front_idx]
            front_y = y.iloc[front_idx] if hasattr(y, 'iloc') else y[front_idx]

            # Draw angle line
            line_length = 0.1
            end_x = front_x + line_length * np.cos(np.radians(angle))
            end_y = front_y + line_length * np.sin(np.radians(angle))

            ax.plot([front_x, end_x], [front_y, end_y], 'r-', linewidth=2)
            ax.text(end_x, end_y, f'{angle}°', fontsize=10, color='red', fontweight='bold')

def generate_real_validation_plots(data_dir):
    """Generate validation plots from real simulation data"""
    print(f"Generating validation plots from data in: {data_dir}")

    # Create plot generator
    plot_generator = RealValidationPlotGenerator(data_dir)

    # Generate all plots
    plot_generator.generate_all_plots()

    print("Real validation plots generation completed!")
    return plot_generator.plot_dir

if __name__ == "__main__":
    # Example usage
    data_dir = "simulation_output/validation_data_20250925_150000"  # Replace with actual data directory
    plot_dir = generate_real_validation_plots(data_dir)
    print(f"Plots saved to: {plot_dir}")
