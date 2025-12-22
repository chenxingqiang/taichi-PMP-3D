# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Core MPM Solvers

This module contains the main MPM solver implementations:
- IncompressibleMPMSolver: Single-phase incompressible flow solver
- TwoPhaseMPMSolver: Two-phase debris flow solver with solid and fluid phases
"""

from taichi_mpm.core.single_phase_solver import IncompressibleMPMSolver
from taichi_mpm.core.two_phase_solver import TwoPhaseMPMSolver

__all__ = [
    "IncompressibleMPMSolver",
    "TwoPhaseMPMSolver",
]

