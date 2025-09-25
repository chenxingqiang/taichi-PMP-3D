# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This is Taichi-LBM3D, a 3D Lattice Boltzmann Method (LBM) solver implemented in Taichi programming language. It simulates fluid dynamics in porous media and includes support for single-phase, two-phase, and phase-change simulations with Multi-Relaxation-Time (MRT) collision schemes and sparse storage capabilities.

## Setup and Dependencies

### Installation
```bash
# Install core dependencies
pip install taichi>=1.0.2
pip install pyevtk
```

### Backend Configuration
- **CPU Backend**: `ti.init(arch=ti.cpu)`
- **GPU Backend**: `ti.init(arch=ti.gpu)` (uses OpenGL or CUDA if available)
- Performance note: Code achieves 900 MLUPS on A100 NVIDIA GPU

## Directory Structure and Architecture

The codebase is organized into simulation types:

### Core Solver Types
- **Single_phase/**: Single-phase fluid simulations
  - `LBM_3D_SinglePhase_Solver.py`: Main solver class with sparse/dense storage options
  - `lbm_solver_3d*.py`: Legacy solvers (non-class based)
  
- **2phase/**: Two-phase flow simulations (oil-water, etc.)
  - `lbm_solver_3d_2phase.py`: Two-phase solver with phase field methods
  - Supports interfacial tension, contact angle, and wetting properties
  
- **Phase_change/**: Thermal simulations with melting/solidification
  - `LBM_3D_SinglePhase_Solute_Solver.py`: Thermal solver with phase transitions
  - Includes buoyancy, thermal diffusivity, and latent heat effects
  
- **Grey_Scale/**: Gray-scale geometry handling for complex porous media

### Key Architectural Concepts

#### D3Q19 Lattice Model
- 19-velocity discrete velocity model for 3D simulations
- Multi-Relaxation-Time (MRT) collision operator for stability
- Transform matrices M and inv_M for moment space calculations

#### Storage Strategies
```python
# Dense storage (default)
solver = LB3D_Solver_Single_Phase(nx=131, ny=131, nz=131, sparse_storage=False)

# Sparse storage (memory efficient for complex geometries)
solver = LB3D_Solver_Single_Phase(nx=131, ny=131, nz=131, sparse_storage=True)
```

#### Boundary Conditions
- **0**: Periodic boundary
- **1**: Fixed pressure boundary  
- **2**: Fixed velocity boundary

#### Geometry Input Format
Geometries are ASCII files with 0 (fluid) and 1 (solid) values in F-order:
```
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            geometry[i,j,k]
```

## Common Development Commands

### Running Simulations
```bash
# Single-phase cavity flow
cd Single_phase
python example_cavity.py

# Porous medium flow  
python example_porous_medium.py

# Two-phase flow
cd ../2phase
python lbm_solver_3d_2phase.py

# Phase change simulation
cd ../Phase_change
python example_cavity_melting.py
```

### Building Documentation
```bash
cd docs
make html  # Builds Sphinx documentation
```

### Converting STL to Binary
```bash
cd Single_phase
# Compile C++ converter
g++ -o Convert_stl_to_binary Convert_stl_to_binary.cpp
```

## Core Solver Configuration Patterns

### Single-Phase Setup Example
```python
import LBM_3D_SinglePhase_Solver as lb3dsp

lb3d = lb3dsp.LB3D_Solver_Single_Phase(nx=131, ny=131, nz=131)
lb3d.init_geo('./geometry.dat')
lb3d.set_viscosity(0.1)
lb3d.set_bc_rho_x0(1.0)    # Inlet pressure
lb3d.set_bc_rho_x1(0.99)   # Outlet pressure  
lb3d.init_simulation()

for iter in range(10000):
    lb3d.step()
    if iter % 500 == 0:
        lb3d.export_VTK(iter)
```

### Two-Phase Configuration
Key parameters for two-phase flows:
- `niu_l`, `niu_g`: Viscosities for liquid/gas phases
- `CapA`: Interfacial tension coefficient
- `psi_solid`: Contact angle (cosine value between -1 and 1)
- Phase boundary conditions with `bc_psi_*` parameters

### Performance Optimization
- Use sparse storage for complex geometries with high solid fraction
- GPU backend recommended for large simulations (>100³ cells)
- Memory partitioning optimized for n_mem_partition=3

## Testing and Validation

### Benchmark Cases
- **Cavity flow**: Lid-driven cavity validation
- **Poiseuille flow**: Analytical solution comparison
- **Porous medium**: Permeability measurements
- **Two-phase drainage**: Capillary pressure curves

### Visualization
Results export to VTK format for visualization in ParaView:
```python
solver.export_VTK(iteration_number)
```

## Physical Units and Scaling

All quantities are in **lattice units**. Key relationships:
- Viscosity: `ν = (τ - 0.5)/3` where τ is relaxation time
- Reynolds number: `Re = UL/ν` 
- Capillary number (two-phase): `Ca = μU/σ`

## Documentation

- **Online docs**: https://yjhp1016.github.io/taichi_LBM3D/
- **Paper reference**: Yang et al., "Taichi-LBM3D: A Single-Phase and Multiphase Lattice Boltzmann Solver on Cross-Platform Multicore CPU/GPUs", Fluids 2022

## Development Notes

### Legacy vs Modern Code
- Class-based solvers (`LBM_3D_SinglePhase_Solver.py`) are recommended over legacy scripts
- Modern solvers support both dense and sparse storage automatically
- Parameter setting through class methods rather than global variables

### Memory Management
- Sparse storage uses Taichi's pointer-based data structures
- Dense storage for simple geometries and maximum performance
- Memory partitioning configured for optimal cache performance

### Taichi Version Compatibility  
- Main branch: Taichi >= 1.0.2
- Two-phase solver: May require Taichi <= 0.8.5 (legacy taichi_glsl dependency)