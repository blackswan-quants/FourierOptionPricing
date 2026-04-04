"""
fourier_options — High-performance FFT-based option pricing library.

Public API:
    from fourier_options import cf_bs, cf_heston, cf_merton, cf_vg
    from fourier_options import fft_pricer, calibrate
"""

from fourier_options.characteristic_functions import cf_bs, cf_heston, cf_merton, cf_vg
from fourier_options.fft_pricer import fft_pricer
from fourier_options.calibration import calibrate

__all__ = [
    "cf_bs", "cf_heston", "cf_merton", "cf_vg",
    "fft_pricer",
    "calibrate",
]
