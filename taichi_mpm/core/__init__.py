# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Core MPM Solvers

This module contains the main MPM solver implementations:
- IncompressibleMPMSolver: Single-phase incompressible flow solver
- TwoPhaseMPMSolver: Two-phase debris flow solver with solid and fluid phases
"""

def __getattr__(name):
    """Lazy loading for Taichi compatibility."""
    if name == "IncompressibleMPMSolver":
        from taichi_mpm.core.single_phase_solver import IncompressibleMPMSolver
        return IncompressibleMPMSolver
    elif name == "TwoPhaseMPMSolver":
        from taichi_mpm.core.two_phase_solver import TwoPhaseMPMSolver
        return TwoPhaseMPMSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "IncompressibleMPMSolver",
    "TwoPhaseMPMSolver",
]



