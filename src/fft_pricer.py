import numpy as np
from typing import Tuple
from characteristic_functions import cf_bs


def fft_pricer(
    S0: float,
    r: float,
    T: float,
    sigma: float,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT pricer for European call options.

    Parameters
    ----------
    S0 : float
        Spot price
    r : float
        Continuous risk-free rate
    T : float
        Time to maturity
    sigma : float
        Volatility (Black-Scholes model)
    alpha : float
        Damping factor (>0)
    N : int
        Number of FFT grid points (power of 2)
    eta : float
        Frequency grid spacing
    
    -------
    K : np.ndarray
        Strike grid
    call_prices : np.ndarray
        Corresponding FFT call prices
    """

    j = np.arange(N)
    v = j * eta

    lambd = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambd

    m = np.arange(N)
    k = -b + m * lambd
    K = np.exp(k)

    u = v - 1j * (alpha + 1)   
    phi_vals = cf_bs(u, S0, T, r, sigma)

    discount = np.exp(-r * T)
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v   

    psi = discount * phi_vals / denom

    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta
    
    fft_input = np.exp(1j * b * v) * psi * w
    fft_output = np.fft.fft(fft_input)

    call_prices = np.exp(-alpha * k) * fft_output.real / np.pi

    return K, call_prices
