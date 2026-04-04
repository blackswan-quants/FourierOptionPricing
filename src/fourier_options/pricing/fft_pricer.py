from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Mapping, Tuple

import numpy as np

from fourier_options.domain.characteristic_functions import cf_bs, cf_heston


def _load_cpp_pricer():
    """Load the compiled pybind11 module when available."""
    project_root = Path(__file__).resolve().parents[3]
    cpp_dir = project_root / "cpp_pricer"
    candidates = sorted(cpp_dir.glob("cpp_pricer*.so")) if cpp_dir.exists() else []

    if candidates:
        spec = importlib.util.spec_from_file_location("cpp_pricer", candidates[0])
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    try:
        return importlib.import_module("cpp_pricer")
    except ModuleNotFoundError:
        if cpp_dir.exists():
            cpp_dir_str = str(cpp_dir)
            if cpp_dir_str not in sys.path:
                sys.path.append(cpp_dir_str)
            try:
                return importlib.import_module("cpp_pricer")
            except ModuleNotFoundError:
                return None
        return None


_CPP_PRICER = _load_cpp_pricer()


def _python_fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray] | None,
    params: Mapping[str, float],
    alpha: float,
    N: int,
    eta: float,
    psi_override: np.ndarray | None,
    option_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python/Numpy Carr-Madan FFT implementation."""
    r = params["r"]
    T = params["T"]

    j = np.arange(N)
    v = j * eta
    u = v - 1j * (alpha + 1.0)

    if psi_override is None:
        if cf is None:
            raise ValueError("cf must be provided when psi_override is None.")
        phi_vals = cf(u, params)

        discount = np.exp(-r * T)
        denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

        psi = discount * phi_vals / denom
    else:
        # Greek integrand supplied externally.
        psi = psi_override

    lambd = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambd

    m = np.arange(N)
    k = -b + m * lambd
    K = np.exp(k)

    # Simpson's rule weights (Carr-Madan, 1999)
    w = (eta / 3.0) * (3.0 + (-1.0) ** (j + 1))
    w[0] = eta / 3.0

    fft_input = np.exp(1j * b * v) * psi * w
    fft_output = np.fft.fft(fft_input)

    values = np.exp(-alpha * k) * fft_output.real / np.pi

    if option_type.lower() == "put":
        S0 = params["S0"]
        discount = np.exp(-r * T)
        values = values - S0 + K * discount

    return K, values


def _can_use_cpp_backend(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray] | None,
    alpha: float | None,
    psi_override: np.ndarray | None,
) -> bool:
    """Use C++ only for supported models and standard price integrands."""
    return (
        _CPP_PRICER is not None
        and psi_override is None
        and alpha is not None
        and alpha > 0.0
        and cf in {cf_bs, cf_heston}
    )


def _cpp_fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    params: Mapping[str, float],
    alpha: float,
    N: int,
    eta: float,
    option_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the pybind11 backend for supported models."""
    if cf is cf_bs:
        return _CPP_PRICER.fft_pricer_bs(
            float(params["S0"]),
            float(params["r"]),
            float(params["sigma"]),
            float(params["T"]),
            float(alpha),
            int(N),
            float(eta),
            option_type,
        )

    if cf is cf_heston:
        return _CPP_PRICER.fft_pricer_heston(
            float(params["S0"]),
            float(params["r"]),
            float(params["T"]),
            float(params["kappa"]),
            float(params["theta"]),
            float(params["sigma_v"]),
            float(params["rho"]),
            float(params["v0"]),
            float(alpha),
            int(N),
            float(eta),
            option_type,
        )

    raise ValueError("Unsupported characteristic function for the C++ backend.")



def fft_pricer(
    cf: Callable[[np.ndarray, Mapping[str, float]], np.ndarray] | None,
    params: Mapping[str, float],
    alpha: float = None,
    N: int = 2**12,
    eta: float = 0.25,
    psi_override: np.ndarray = None,
    option_type: str = 'call'
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Carr-Madan FFT pricer for European options (call or put).

    Parameters
    ----------
    cf : callable
        Characteristic function Φ(u; params).
    
    params : dict
        Dictionary containing all model parameters required by the characteristic
        function. Must include at least:
            - "r": risk-free rate
            - "T": time to maturity
            - "S0": initial stock price
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

    option_type : str, default 'call'
        Type of option to price. Either 'call' or 'put'.
        If 'put', computed via put-call parity from call prices for numerical stability.
    
    Returns
    -------
    K : np.ndarray
        Strike grid
    values : np.ndarray
        Output of the FFT inversion. Represents:
        - call/put prices under the Carr–Madan transform when `psi_override` is None,
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
                                     psi_override=psi_override, option_type=option_type)
            window = central_mask(_K)
            if np.all(np.isfinite(_values[window])) and np.all(_values[window] >= 0):
                return _K, _values
        # Fallback if no candidate passed (should not happen for standard models)
        return fft_pricer(cf, params, alpha=1.5, N=N, eta=eta,
                          psi_override=psi_override, option_type=option_type)

    if _can_use_cpp_backend(cf, alpha, psi_override):
        return _cpp_fft_pricer(cf, params, alpha, N, eta, option_type)

    return _python_fft_pricer(cf, params, alpha, N, eta, psi_override, option_type)
