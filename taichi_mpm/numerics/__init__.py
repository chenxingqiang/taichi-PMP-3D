# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Numerical Methods

This module contains numerical method implementations:
- LevelSetMethod: Level set tracking with WENO3/RK3-TVD schemes
"""

from taichi_mpm.numerics.level_set import LevelSetMethod

__all__ = [
    "LevelSetMethod",
]

