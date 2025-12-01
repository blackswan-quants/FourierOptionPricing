import numpy as np
from typing import Tuple, Callable, Mapping
import characteristic_functions


def fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    params: Mapping[str, float],
    S0: float = 1.0,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Carr-Madan FFT pricer for European call options.
    
    Implements the fast Fourier transform (FFT) method for pricing European call options
    using characteristic functions. This approach is efficient for computing option prices
    across multiple strikes simultaneously.

    Parameters
    ----------
    cf : callable
        Characteristic function Φ(u; params) that takes:
            - u: np.ndarray of complex frequency values
            - params: Mapping with model parameters
        Returns np.ndarray of characteristic function values.
    
    params : dict
        Dictionary containing all model parameters required by the characteristic function.
        Must include at least:
            - "r" (float): Risk-free rate (annualized, e.g., 0.03 for 3%)
            - "T" (float): Time to maturity in years (e.g., 0.25 for 3 months)
        Additional model-specific parameters depend on the chosen model (e.g., Heston,
        Merton, etc.) and should be included in this dictionary.

    S0 : float, optional
        Current spot price of the underlying asset. Used to center the strike grid.
        Default: 1.0

    alpha : float, optional
        Damping factor used in the Carr-Madan transform. Controls the integration path
        in the complex plane. Typical values: 1.5 for OTM calls. Must be > 0.
        Default: 1.5

    N : int, optional
        Number of FFT grid points (should be a power of 2 for computational efficiency).
        Larger values provide finer strike grid but increase computation time.
        Default: 2**12 (4096 points)

    eta : float, optional
        Spacing of the frequency grid in the Fourier domain. Smaller values give finer
        resolution in log-strike space. Default: 0.25

    Returns
    -------
    K : np.ndarray
        Strike prices grid. Array of shape (N,) containing strike values corresponding
        to the computed option prices.
    
    call_prices : np.ndarray
        European call option prices. Array of shape (N,) containing prices corresponding
        to each strike in K. Prices are in the same currency as the underlying spot price.
    
    Notes
    -----
    - The function uses the Carr-Madan FFT method, which is O(N log N) in complexity.
    - Results are typically accurate to machine precision for well-behaved characteristic functions.
    - Ensure that the characteristic function cf is properly normalized (integrates to 1).
    - For optimal performance, N should be a power of 2 (e.g., 2^10, 2^12, 2^14).
    
    References
    ----------
    Carr, P., & Madan, D. B. (1999). Option valuation using the fast Fourier transform.
    Journal of Computational Finance, 2(4), 61-73.
    """

    r  = params["r"]
    T  = params["T"]

    # --- 0. Prepare Parameters ---
    # Inject S0 into parameters so the characteristic function can use it
    calc_params = dict(params)
    calc_params["S0"] = S0

    # --- Frequency Grid Setup ---
    # Create frequency values v = j * eta where j = 0, 1, ..., N-1
    j = np.arange(N)
    v = j * eta
    # Shift frequency for integration path: u = v - i(alpha + 1)
    u = v - 1j * (alpha + 1.0)

    # --- Characteristic Function Evaluation ---
    # Evaluate the characteristic function at shifted frequencies
    phi_vals = cf(u, calc_params)

    # --- Strike Grid Construction ---
    # Lambda defines spacing in log-strike space
    lambd = 2 * np.pi / (N * eta)
    # b is half the log-strike range
    b = 0.5 * N * lambd

    # Create log-strike grid and convert to strike prices
    # Center the grid around ln(S0)
    m = np.arange(N)
    k = -b + m * lambd + np.log(S0)
    K = np.exp(k)

    # --- Carr-Madan Transform ---
    # Apply discount factor
    discount = np.exp(-r * T)
    # Denominator of the Carr-Madan formula
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v   

    # Compute the integrand (psi function)
    psi = discount * phi_vals / denom

    # --- Trapezoidal Weighting ---
    # Simpson's/trapezoidal rule weights for numerical integration
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta
    
    # --- FFT Computation ---
    # Prepare input for FFT with phase adjustment
    fft_input = np.exp(1j * b * v) * psi * w
    # Apply FFT
    fft_output = np.fft.fft(fft_input)

    # --- Extract Real Call Prices ---
    # Extract real part and apply damping factor to obtain call prices
    call_prices = np.exp(-alpha * k) * fft_output.real / np.pi

    return K, call_prices
