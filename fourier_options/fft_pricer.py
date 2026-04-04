"""
Carr-Madan FFT pricer — European options + Euler Greeks.

Main entry point:
    K, C, Delta = fft_pricer(cf, params, alpha=1.5)

The function dispatches to the compiled C++ backend (cpp_pricer.pyd) if available,
falling back to a pure NumPy implementation for portability.

Euler Delta theorem (O(N)):
    Δ = c(k) - ∂c(k)/∂k        (where c(k) = C/S₀, k = ln(K/S₀))

Euler Gamma theorem (O(N)):
    Γ = (∂²c/∂k² - ∂c/∂k) / S₀
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Callable, Mapping, Tuple

import numpy as np

from fourier_options.characteristic_functions import cf_bs, cf_heston, cf_merton, cf_vg


# ── C++ Backend Loader ───────────────────────────────────────────────────────
def _load_cpp():
    """Try to load the compiled pybind11 module."""
    root = Path(__file__).resolve().parents[1]
    for pat in ["cpp_pricer*.pyd", "cpp_pricer*.so"]:
        for p in root.glob(pat):
            spec = importlib.util.spec_from_file_location("cpp_pricer", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "fft_pricer_bs"):
                    return mod
    try:
        mod = importlib.import_module("cpp_pricer")
        return mod if hasattr(mod, "fft_pricer_bs") else None
    except ModuleNotFoundError:
        return None


_CPP = _load_cpp()
_SUPPORTED_CFS = {cf_bs, cf_heston, cf_merton, cf_vg}


# ── Pure-Python Fallback ─────────────────────────────────────────────────────
def _pricer_numpy(cf, params, alpha, N, eta, option_type) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, T, q = params["r"], params["T"], params.get("q", 0.0)
    S0 = params["S0"]
    j = np.arange(N); v = j * eta; u = v - 1j * (alpha + 1.0)

    discount = np.exp(-r * T)
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    psi = discount * cf(u, params) / denom

    lambd = 2 * np.pi / (N * eta); b = 0.5 * N * lambd
    k = -b + j * lambd; K = S0 * np.exp(k)

    # Simpson weights (1-4-2-4-...-4-1)
    w = (eta / 3.0) * np.where(j == 0, 1.0, np.where(j % 2 == 0, 2.0, 4.0))
    ck = np.exp(-alpha * k) * np.fft.fft(np.exp(1j * b * v) * psi * w).real / np.pi

    # Euler Delta Δ = c(k) - ∂c/∂k
    dck = np.gradient(ck, lambd)
    Delta = ck - dck

    values = S0 * ck
    if option_type == "put":
        values = values - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return K, values, Delta


# ── C++ Dispatch ─────────────────────────────────────────────────────────────
def _pricer_cpp(cf, params, alpha, N, eta, option_type) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S0, r, q, T = float(params["S0"]), float(params["r"]), float(params.get("q", 0.0)), float(params["T"])
    a, n, e, ty = float(alpha), int(N), float(eta), option_type
    if cf is cf_bs:
        return _CPP.fft_pricer_bs(S0, r, q, float(params["sigma"]), T, a, n, e, ty)
    if cf is cf_heston:
        return _CPP.fft_pricer_heston(S0, r, q, T, float(params["kappa"]), float(params["theta"]),
                                       float(params["sigma_v"]), float(params["rho"]), float(params["v0"]), a, n, e, ty)
    if cf is cf_merton:
        return _CPP.fft_pricer_merton(S0, r, q, float(params["sigma"]), T,
                                       float(params["lam"]), float(params["mu_j"]), float(params["sig_j"]), a, n, e, ty)
    if cf is cf_vg:
        return _CPP.fft_pricer_vg(S0, r, q, T, float(params["sigma"]),
                                   float(params["nu"]), float(params["theta_vg"]), a, n, e, ty)
    raise ValueError(f"Unsupported CF: {cf}")


# ── Public API ────────────────────────────────────────────────────────────────
def fft_pricer(
    cf: Callable,
    params: Mapping[str, float],
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
    option_type: str = "call",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT pricer for European options.

    Parameters
    ----------
    cf : callable
        Characteristic function φ(u, params) returning the CF of the log-return.
    params : dict
        Must contain at least: S0, r, T, q (optional).
        Model-specific params depend on the CF.
    alpha : float
        Carr-Madan damping factor. Default 1.5.
    N : int
        Number of FFT points (recommend power of 2). Default 4096.
    eta : float
        Grid spacing in frequency domain. Default 0.25.
    option_type : str
        'call' or 'put'. Default 'call'.

    Returns
    -------
    K : np.ndarray     — strike grid
    C : np.ndarray     — option prices
    Delta : np.ndarray — Euler Delta (∂C/∂S₀)
    """
    if _CPP is not None and cf in _SUPPORTED_CFS:
        return _pricer_cpp(cf, params, alpha, N, eta, option_type)
    return _pricer_numpy(cf, params, alpha, N, eta, option_type)


def euler_gamma(
    cf: Callable,
    params: Mapping[str, float],
    alpha: float = 1.1,
    N: int = 8192,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gamma via the Euler theorem on the FFT price grid.

    Γ = (∂²c/∂k² - ∂c/∂k) / S₀

    Returns
    -------
    K : np.ndarray     — strike grid
    Gamma : np.ndarray — Euler Gamma (∂²C/∂S₀²)
    """
    K, prices, _ = fft_pricer(cf, params, alpha=alpha, N=N, eta=eta)
    S0 = params["S0"]
    ck = prices / S0
    lambd = np.diff(np.log(K))[0]
    dck = np.gradient(ck, lambd)
    d2ck = np.gradient(dck, lambd)
    return K, (d2ck - dck) / S0
