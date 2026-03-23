import numpy as np
from typing import Tuple, Callable, Mapping



def fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    params: Mapping[str, float],
    alpha: float = None,
    N: int = 2**12,
    eta: float = 0.25,
    psi_override: np.ndarray = None
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

    alpha : float or None
        Damping factor used in the Carr-Madan transform. Must be > 0 for calls.
        If None, an adaptive selection is performed: candidates in
        [0.25, 0.50, ..., 4.00] are tried in order and the first one that
        produces a finite, non-negative integrand is returned.

    N : int
        Number of FFT grid points (recommend power of 2 for efficiency).

    eta : float
        Spacing of the frequency grid in the Fourier domain.

    psi_override : np.ndarray, optional
        Optional complex-valued integrand to replace the standard Carr-Madan
        call-integrand.
        Intended for the computation of option Greeks(Delta, Gamma, Vega).

    
    -------
    K : np.ndarray
        Strike grid
    values : np.ndarray
        Output of the FFT inversion. Represents:
        - call prices under the Carr–Madan transform when `psi_override` is None,
        - otherwise, the quantity associated with the provided integrand(Delta, Gamma, Vega).
    """

    # Adaptive alpha: sweep candidates and pick the first numerically stable one.
    # A candidate is accepted if the integrand psi is finite and the resulting
    # prices in a central window are all non-negative.
    if alpha is None:
        S0 = params.get("S0", 1.0)
        central_mask = lambda K: (K > S0 * 0.5) & (K < S0 * 2.0)
        for _alpha in np.arange(0.25, 4.25, 0.25):
            _K, _values = fft_pricer(cf, params, alpha=_alpha, N=N, eta=eta,
                                     psi_override=psi_override)
            window = central_mask(_K)
            if np.all(np.isfinite(_values[window])) and np.all(_values[window] >= 0):
                return _K, _values
        # Fallback if no candidate passed (should not happen for standard models)
        return fft_pricer(cf, params, alpha=1.5, N=N, eta=eta,
                          psi_override=psi_override)

    r  = params["r"]
    T  = params["T"]

    j = np.arange(N)
    v = j * eta
    u = v - 1j * (alpha + 1.0)

    if psi_override is None:
        phi_vals = cf(u, params)

        discount = np.exp(-r * T)
        denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v 

        psi = discount * phi_vals / denom

    else:
        #Greek integrand
        psi = psi_override

    lambd = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambd

    m = np.arange(N)
    k = -b + m * lambd
    K = np.exp(k)

    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta
    
    fft_input = np.exp(1j * b * v) * psi * w
    fft_output = np.fft.fft(fft_input)

    values = np.exp(-alpha * k) * fft_output.real / np.pi

    return K, values
