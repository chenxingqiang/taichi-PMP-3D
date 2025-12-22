# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2025 Xingqiang Chen. All rights reserved.

"""
Numerical Methods

This module contains numerical method implementations:
- LevelSetMethod: Level set tracking with WENO3/RK3-TVD schemes
"""

def __getattr__(name):
    """Lazy loading for Taichi compatibility."""
    if name == "LevelSetMethod":
        from taichi_mpm.numerics.level_set import LevelSetMethod
        return LevelSetMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LevelSetMethod",
]



