# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Physical Models

This module contains physical model implementations:
- BarrierModel: Rigid barrier contact mechanics
- DruckerPragerRheology: Drucker-Prager elastoplastic constitutive model
- TwoPhaseCoupling: Two-phase flow coupling kernels
"""

def __getattr__(name):
    """Lazy loading for Taichi compatibility."""
    if name == "BarrierModel":
        from taichi_mpm.models.barrier import BarrierModel
        return BarrierModel
    elif name == "DruckerPragerRheology":
        from taichi_mpm.models.drucker_prager import DruckerPragerRheology
        return DruckerPragerRheology
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BarrierModel",
    "DruckerPragerRheology",
]



