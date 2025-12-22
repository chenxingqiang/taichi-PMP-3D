# -*- coding: utf-8 -*-
# Author: Xingqiang Chen
# Email: chen.xingqiang@turingai.cc
# Copyright (c) 2024 Xingqiang Chen. All rights reserved.

"""
Physical Models

This module contains physical model implementations:
- BarrierModel: Rigid barrier contact mechanics
- DruckerPragerModel: Drucker-Prager elastoplastic constitutive model
- TwoPhaseCoupling: Two-phase flow coupling kernels
"""

from taichi_mpm.models.barrier import BarrierModel
from taichi_mpm.models.drucker_prager import DruckerPragerModel

__all__ = [
    "BarrierModel",
    "DruckerPragerModel",
]



