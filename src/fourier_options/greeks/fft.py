import numpy as np
from typing import Mapping, Tuple
from .integrands import (
    delta_integrand,
    gamma_integrand,
    vega_integrand
)

from fourier_options.pricing.fft_pricer import fft_pricer
def delta_fft_bs(
    params: Mapping[str, float],
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Delta(K) via FFT."""
    v = eta * np.arange(N)
    psi = delta_integrand(v, params, alpha)
    return fft_pricer(cf=None, params=params, alpha=alpha, N=N, eta=eta, psi_override=psi)


def gamma_fft_bs(
    params: Mapping[str, float],
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gamma(K) via FFT."""
    v = eta * np.arange(N)
    psi = gamma_integrand(v, params, alpha)
    return fft_pricer(cf=None, params=params, alpha=alpha, N=N, eta=eta, psi_override=psi)


def vega_fft_bs(
    params: Mapping[str, float],
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vega(K) via FFT."""
    v = eta * np.arange(N)
    psi = vega_integrand(v, params, alpha)
    return fft_pricer(cf=None, params=params, alpha=alpha, N=N, eta=eta, psi_override=psi)