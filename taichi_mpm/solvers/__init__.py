# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Linear Solvers

This module contains linear solver implementations:
- PCGSolver: Preconditioned Conjugate Gradient solver for pressure Poisson equation
"""

from taichi_mpm.solvers.pcg_solver import PCGSolver

__all__ = [
    "PCGSolver",
]



