import numpy as np
from numpy.typing import NDArray

def cf_bs(u: NDArray[np.complex128], params: dict[str, float]) -> NDArray[np.complex128]:
    """
    Characteristic function for Black-Scholes model.
    
    The Black-Scholes model assumes constant volatility and log-normal price distribution.
    Returns the Fourier transform of the log-price distribution.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        q (float, optional): Dividend yield (default: 0.0)
    """
    S0 = params["S0"]
    r  = params["r"]
    q = params.get("q", 0.0)
    sigma = params["sigma"]
    T  = params["T"]

    # Expected log-price at maturity under the risk-neutral measure
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T

    # CF of normal distribution with mean mu and variance sigma^2*T
    return np.exp(1j * u * mu - 0.5 * sigma**2 * u**2 * T) 

def cf_merton(u: NDArray[np.complex128], params: dict[str, float]) -> NDArray[np.complex128]:
    """
    Characteristic function for Merton jump-diffusion model.
    
    Extends Black-Scholes by adding random jumps in price. The jumps follow a compound 
    Poisson process with log-normal jump sizes.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Diffusion volatility (annualized)
        T (float): Time to maturity (in years)
        lam (float): Jump intensity (average number of jumps per year)
        mu_j (float): Mean of log-jump size
        sig_j (float): Standard deviation of log-jump size
        q (float, optional): Dividend yield (default: 0.0)
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
    
    Volatility follows a mean-reverting CIR (Cox-Ingersoll-Ross) process, allowing 
    volatility to vary stochastically over time. Includes correlation between price 
    and volatility movements (leverage effect).
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        T (float): Time to maturity (in years)
        kappa (float): Mean reversion speed of variance
        theta (float): Long-run average variance
        sigma_v (float): Volatility of variance (vol-of-vol)
        rho (float): Correlation between price and variance (-1 to 1)
        v0 (float): Initial variance
        q (float, optional): Dividend yield (default: 0.0)
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
