import numpy as np
from typing import Tuple, Callable, Mapping
import characteristic_functions


def fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    params: Mapping[str, float],
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Carr-Madan FFT pricer for European call options.

    Parameters
    ----------
    cf : callable
        Characteristic function Φ(u; params).
    
    params : dict
        Dictionary containing all model parameters required by the characteristic
        function. Must include at least:
            - "r": risk-free rate
            - "T": time to maturity
        and additional model-specific parameters.

     alpha : float
        Damping factor used in the Carr-Madan transform. Must be > 0.

    N : int
        Number of FFT grid points (recommend power of 2 for efficiency).

    eta : float
        Spacing of the frequency grid in the Fourier domain.

    
    -------
    K : np.ndarray
        Strike grid
    call_prices : np.ndarray
        Corresponding FFT call prices
    """

    r  = params["r"]
    T  = params["T"]

    j = np.arange(N)
    v = j * eta
    u = v - 1j * (alpha + 1.0)

    phi_vals = cf(u, params)

    lambd = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambd

    m = np.arange(N)
    k = -b + m * lambd
    K = np.exp(k)


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
