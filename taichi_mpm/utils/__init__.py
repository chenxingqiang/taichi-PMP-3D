# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Utility Functions

This module contains utility functions:
- DataExtractor: Extract simulation data for analysis
- OutputMetrics: Compute and output simulation metrics
"""

def __getattr__(name):
    """Lazy loading for Taichi compatibility."""
    if name == "DataExtractor":
        from taichi_mpm.utils.data_extractor import DataExtractor
        return DataExtractor
    elif name == "OutputMetrics":
        from taichi_mpm.utils.output_metrics import OutputMetrics
        return OutputMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataExtractor",
    "OutputMetrics",
]



