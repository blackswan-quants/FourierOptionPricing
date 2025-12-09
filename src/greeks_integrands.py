import numpy as np
from numpy.typing import NDArray
from typing import Mapping
from characteristic_functions import cf_bs

def delta_integrand(
    v: NDArray[np.float64],
    params: Mapping[str, float],
    alpha: float,
) -> NDArray[np.complex128]:
    """
    Carr-Madan integrand for Delta (first derivative of option price w.r.t. S0).
    
    Delta measures the rate of change of option price with respect to the underlying
    stock price. Used for hedging and risk management.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        
    Args:
        v (NDArray): Integration variable (frequency domain)
        params (Mapping): Model parameters dictionary
        alpha (float): Damping parameter for Carr-Madan formula
        
    Returns:
        Complex integrand to be integrated for Delta calculation
    """
    S0 = params["S0"]
    r = params["r"]
    T = params["T"]

    # Complex frequency with damping shift
    u = v - 1j * (alpha + 1.0)
    
    # Characteristic function at frequency u
    phi = cf_bs(u.astype(np.complex128), params)

    # Derivative of CF w.r.t. S0 (via chain rule)
    dphi_dS0 = 1j * u * phi / S0

    # Present value discount factor
    discount = np.exp(-r * T)
    
    # Carr-Madan denominator for call option
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

    return discount * dphi_dS0 / denom


def gamma_integrand(
    v: NDArray[np.float64],
    params: Mapping[str, float],
    alpha: float,
) -> NDArray[np.complex128]:
    """
    Carr-Madan integrand for Gamma (second derivative of option price w.r.t. S0).
    
    Gamma measures the rate of change of Delta with respect to the underlying stock
    price. Indicates convexity of the option price and is crucial for portfolio hedging.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        
    Args:
        v (NDArray): Integration variable (frequency domain)
        params (Mapping): Model parameters dictionary
        alpha (float): Damping parameter for Carr-Madan formula
        
    Returns:
        Complex integrand to be integrated for Gamma calculation
    """
    S0 = params["S0"]
    r = params["r"]
    T = params["T"]

    # Complex frequency with damping shift
    u = v - 1j * (alpha + 1.0)
    
    # Characteristic function at frequency u
    phi = cf_bs(u.astype(np.complex128), params)

    # Second derivative of CF w.r.t. S0 (via chain rule)
    gamma_factor = (-u**2 - 1j * u) * phi / (S0**2)

    # Present value discount factor
    discount = np.exp(-r * T)
    
    # Carr-Madan denominator for call option
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

    return discount * gamma_factor / denom


def vega_integrand(
    v: NDArray[np.float64],
    params: Mapping[str, float],
    alpha: float,
) -> NDArray[np.complex128]:
    """
    Carr-Madan integrand for Vega (derivative of option price w.r.t. volatility).
    
    Vega measures the sensitivity of option price to changes in volatility. Important
    for volatility trading and assessing exposure to volatility risk.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        
    Args:
        v (NDArray): Integration variable (frequency domain)
        params (Mapping): Model parameters dictionary
        alpha (float): Damping parameter for Carr-Madan formula
        
    Returns:
        Complex integrand to be integrated for Vega calculation
    """
    sigma = params["sigma"]
    r = params["r"]
    T = params["T"]

    # Complex frequency with damping shift
    u = v - 1j * (alpha + 1.0)
    
    # Characteristic function at frequency u
    phi = cf_bs(u.astype(np.complex128), params)

    # Derivative of CF w.r.t. sigma (from BS CF structure)
    dphi_dsigma = phi * (T * sigma * (-1j * u - u**2))

    # Present value discount factor
    discount = np.exp(-r * T)
    
    # Carr-Madan denominator for call option
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

    return discount * dphi_dsigma / denom
