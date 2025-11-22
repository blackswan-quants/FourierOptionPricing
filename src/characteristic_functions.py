import numpy as np
from numpy.typing import NDArray

def cf_black_scholes(u: NDArray[np.complex128], params: dict[str, float]) -> NDArray[np.complex128]:
    """
    Characteristic function for Black-Scholes model.
    Returns the Fourier transform of the log-price distribution.
    """
    S0 = params["S0"]
    r  = params["r"]
    q  = params.get("q", 0.0)
    sigma = params["sigma"]
    T  = params["T"]

    # Mean of log-return under risk-neutral measure
    mu = np.log(S0) + r - 0.5 * sigma**2
    
    # CF of normal distribution
    return np.exp(1j * u * mu * T - 0.5 * sigma**2 * u**2 * T)

def cf_merton(u: NDArray[np.complex128], params: dict[str, float]) -> NDArray[np.complex128]:
    """
    Characteristic function for Merton jump-diffusion model.
    Combines continuous diffusion with discrete jumps in log-price.
    """
    S0 = params["S0"]
    r = params["r"]
    q = params.get("q", 0.0)
    sigma = params["sigma"]
    T = params["T"]

    lam = params["lam"]
    mu_j = params["mu_j"]
    sig_j = params["sig_j"]
    
    # Expected relative jump size (used for drift correction)
    kappa = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

    # Risk-neutral drift with jump compensation
    drift = np.log(S0) + (r - q - 0.5 * sigma**2 - lam * kappa) * T

    # Diffusion component (Brownian motion part)
    diffusion_part = np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T)
    
    # Jump component (compound Poisson process)
    jump_cf = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * sig_j**2 * u**2) - 1.0))

    return diffusion_part * jump_cf

def cf_heston(u: NDArray[np.complex128], params: dict[str, float]) -> NDArray[np.complex128]:
    """
    Characteristic function for Heston stochastic volatility model.
    Volatility follows a mean-reverting CIR process.
    """
    S0    = params["S0"]
    r     = params["r"]
    q     = params.get("q", 0.0)
    T     = params["T"]

    kappa = params["kappa"]
    theta = params["theta"]
    sigma_v = params["sigma_v"]
    rho   = params["rho"]
    v0    = params["v0"]

    # Heston CF uses complex logarithms - auxiliary variables
    a  = kappa * theta

    d  = np.sqrt((rho * sigma_v * 1j * u - kappa)**2 + sigma_v**2 * (1j * u + u**2))
    b  = kappa - rho * sigma_v * 1j * u
    g  = (b - d) / (b + d)

    exp_neg_dT = np.exp(-d * T)

    # Contribution from log-price drift
    C = (1j * u * (np.log(S0) + (r - q) * T)
         + a / sigma_v**2 * ((b - d) * T - 2.0 * np.log((1 - g * exp_neg_dT) / (1 - g))))
    
    # Contribution from stochastic variance
    D = (b - d) / sigma_v**2 * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT))

    return np.exp(C + D * v0)
