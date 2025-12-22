# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Linear Solvers

This module contains linear solver implementations:
- PCGSolver: Preconditioned Conjugate Gradient solver for pressure Poisson equation
"""

def __getattr__(name):
    """Lazy loading for Taichi compatibility."""
    if name == "PCGSolver":
        from taichi_mpm.solvers.pcg_solver import PCGSolver
        return PCGSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PCGSolver",
]



