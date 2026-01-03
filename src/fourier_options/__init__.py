"""
Fourier Options - FFT-based Option Pricing and Calibration.

This package provides tools for pricing European options using
Fourier transform methods (Carr-Madan FFT) and calibrating
stochastic volatility models like Heston.

Submodules:
    - pricing: FFT-based option pricing
    - domain: Characteristic functions (BS, Merton, Heston)
    - greeks: Option Greeks via FFT
    - calibration: Model calibration tools
    - data: Data pipeline for option chain processing
    - utils: Utility functions
"""

from . import pricing
from . import domain
from . import greeks
from . import calibration
from . import data
from . import utils

__all__ = [
    "pricing",
    "domain", 
    "greeks",
    "calibration",
    "data",
    "utils",
]
