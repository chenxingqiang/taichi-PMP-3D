#!/bin/bash

# Run Detailed MPM Simulation with Comprehensive Logging
# This script runs the simulation with detailed logging to diagnose convergence issues

echo "=== DETAILED MPM SIMULATION WITH LOGGING ==="
echo "This will run the simulation with comprehensive logging"
echo "to diagnose convergence and numerical stability issues."
echo "="*60

# Set environment variables
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p simulation_output
mkdir -p simulation_output/convergence_analysis

# Run the detailed simulation
echo "Starting detailed simulation with logging..."
echo "Logs will be saved in: logs/"
echo "Results will be saved in: simulation_output/"
echo ""

python detailed_simulation_with_logging.py

echo ""
echo "=== SIMULATION COMPLETED ==="
echo "Check the following for results:"
echo "  - logs/ directory for detailed logs"
echo "  - simulation_output/convergence_analysis/ for diagnostic plots"
echo "  - simulation_output/ for VTK files and other outputs"
echo ""
echo "To view the latest log file:"
echo "  tail -f logs/simulation_detailed_*.log"
echo ""
echo "To view convergence plots:"
echo "  open simulation_output/convergence_analysis/convergence_diagnostics.png"
