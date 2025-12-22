# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Taichi MPM 3D: Two-Phase Material Point Method for Debris Flow Simulation

This package provides high-performance implementations of the Material Point Method
for simulating single-phase and two-phase debris flows.

Modules:
    core: Core MPM solvers (single-phase iMPM, two-phase MPM)
    solvers: Linear solvers (PCG with preconditioners)
    models: Physical models (Drucker-Prager, drag, barrier)
    numerics: Numerical methods (level set, interpolation)
    utils: Utilities (config, I/O)

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.cpu, default_fp=ti.f64)
    >>> from taichi_mpm.core import TwoPhaseMPMSolver
    >>> solver = TwoPhaseMPMSolver(nx=100, ny=25, nz=20, dx=0.02)
    >>> solver.init_particles(x_min=0.04, x_max=0.44, y_min=0.04, y_max=0.34, z_min=0.04, z_max=0.44)
    >>> for step in range(1000):
    ...     solver.step()
"""

__version__ = "0.2.0"
__author__ = "Xingqiang Chen"
__email__ = "chen.xingqiang@turingai.cc"

# Lazy loading for all submodules (Taichi requires initialization first)
def __getattr__(name):
    """Lazy loading of modules and classes to allow Taichi initialization first."""
    # Submodules
    if name == "core":
        from taichi_mpm import core
        return core
    elif name == "solvers":
        from taichi_mpm import solvers
        return solvers
    elif name == "models":
        from taichi_mpm import models
        return models
    elif name == "numerics":
        from taichi_mpm import numerics
        return numerics
    elif name == "utils":
        from taichi_mpm import utils
        return utils
    # Core classes
    elif name == "IncompressibleMPMSolver":
        from taichi_mpm.core.single_phase_solver import IncompressibleMPMSolver
        return IncompressibleMPMSolver
    elif name == "TwoPhaseMPMSolver":
        from taichi_mpm.core.two_phase_solver import TwoPhaseMPMSolver
        return TwoPhaseMPMSolver
    elif name == "PCGSolver":
        from taichi_mpm.solvers.pcg_solver import PCGSolver
        return PCGSolver
    elif name == "LevelSetMethod":
        from taichi_mpm.numerics.level_set import LevelSetMethod
        return LevelSetMethod
    elif name == "BarrierModel":
        from taichi_mpm.models.barrier import BarrierModel
        return BarrierModel
    elif name == "DruckerPragerRheology":
        from taichi_mpm.models.drucker_prager import DruckerPragerRheology
        return DruckerPragerRheology
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Submodules
    "core",
    "solvers", 
    "models",
    "numerics",
    "utils",
    # Core classes
    "IncompressibleMPMSolver",
    "TwoPhaseMPMSolver",
    "PCGSolver",
    "LevelSetMethod",
    "BarrierModel",
]
