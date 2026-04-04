"""
Characteristic functions for the risk-neutral log-price under various models.

Each function φ(u; params) returns the CF evaluated at complex frequencies u,
assuming S₀=1 (log-return normalization). The caller rescales by S₀.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cf_bs(u: NDArray, params: dict) -> NDArray:
    """Black-Scholes characteristic function (log-return, S0=1 normalized).

    params: r, q, sigma, T
    """
    r, q = params["r"], params.get("q", 0.0)
    sigma, T = params["sigma"], params["T"]
    mu = (r - q - 0.5 * sigma**2) * T
    return np.exp(1j * u * mu - 0.5 * sigma**2 * u**2 * T)


def cf_merton(u: NDArray, params: dict) -> NDArray:
    """Merton jump-diffusion characteristic function (log-return, S0=1).

    params: r, q, sigma, T, lam, mu_j, sig_j
    """
    r, q = params["r"], params.get("q", 0.0)
    sigma, T = params["sigma"], params["T"]
    lam, mu_j, sig_j = params["lam"], params["mu_j"], params["sig_j"]
    kappa = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
    drift = (r - q - 0.5 * sigma**2 - lam * kappa) * T
    diffusion = np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T)
    jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * sig_j**2 * u**2) - 1.0))
    return diffusion * jump


def cf_heston(u: NDArray, params: dict) -> NDArray:
    """Heston stochastic volatility characteristic function (log-return, S0=1).

    params: r, q, T, kappa, theta, sigma_v, rho, v0
    """
    r, q, T = params["r"], params.get("q", 0.0), params["T"]
    kappa, theta = params["kappa"], params["theta"]
    sigma_v, rho, v0 = params["sigma_v"], params["rho"], params["v0"]

    a = kappa * theta
    d = np.sqrt((rho * sigma_v * 1j * u - kappa)**2 + sigma_v**2 * (1j * u + u**2))
    b = kappa - rho * sigma_v * 1j * u
    g = (b - d) / (b + d)
    exp_neg_dT = np.exp(-d * T)

    C = (1j * u * (r - q) * T
         + a / sigma_v**2 * ((b - d) * T - 2.0 * np.log((1 - g * exp_neg_dT) / (1 - g))))
    D = (b - d) / sigma_v**2 * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT))
    return np.exp(C + D * v0)


def cf_vg(u: NDArray, params: dict) -> NDArray:
    """Variance Gamma characteristic function (log-return, S0=1).

    params: r, q, T, sigma, nu, theta_vg
    """
    r, q, T = params["r"], params.get("q", 0.0), params["T"]
    sigma, nu, theta_vg = params["sigma"], params["nu"], params["theta_vg"]
    conv_arg = 1.0 - theta_vg * nu - 0.5 * sigma**2 * nu
    if np.isscalar(conv_arg) and conv_arg <= 0:
        return np.zeros_like(u)
    conv_arg = np.where(conv_arg > 0, conv_arg, 1e-14)
    omega = (1.0 / nu) * np.log(conv_arg)
    phi_vg = (1.0 - 1j * u * theta_vg * nu + 0.5 * sigma**2 * nu * u**2)**(-T / nu)
    return np.exp(1j * u * (r - q + omega) * T) * phi_vg
