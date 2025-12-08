import numpy as np
from fft_pricer import fft_pricer
from characteristic_functions import cf_bs


def _bs_fft_prices_on_strikes(
    params: dict[str, float],
    strikes: np.ndarray,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> np.ndarray:
    """
    Black–Scholes call prices via Carr–Madan FFT, interpolated to given strikes.

    Computes European call option prices on a uniform grid using FFT, then interpolates
    to match the desired strike prices. Uses log-linear interpolation for stability.

    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)

    Args:
        params (dict): Model parameters
        strikes (np.ndarray): Target strike prices for interpolation
        alpha (float): Damping parameter for Carr-Madan formula
        N (int): Number of FFT points (power of 2)
        eta (float): Grid spacing in log-strike space

    Returns:
        Call option prices interpolated at the given strikes
    """
    K_grid, C_grid = fft_pricer(cf_bs, params, alpha=alpha, N=N, eta=eta)
    strikes = np.asarray(strikes, dtype=float)

    # Linear interpolation in log-strike space
    logK = np.log(strikes)
    return np.interp(logK, np.log(K_grid), C_grid)


def mse_loss(
    theta: dict[str, float],
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> float:
    """
    Mean squared error between market call prices and Black–Scholes model prices.

    Loss function for calibrating model parameters to market data. Computes model
    prices via Carr–Madan FFT for each unique maturity, then calculates MSE across
    all options. Lower MSE indicates better model fit to market prices.

    Required theta:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized) - parameter to calibrate

    Args:
        theta (dict): Model parameters (including sigma to calibrate)
        strikes (np.ndarray): Strike prices of market options
        maturities (np.ndarray): Times to maturity for each option
        market_prices (np.ndarray): Observed market prices for each option
        alpha (float): Damping parameter for Carr-Madan formula
        N (int): Number of FFT points (power of 2)
        eta (float): Grid spacing in log-strike space

    Returns:
        Mean squared error between model and market prices
    """

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)

    S0 = float(theta["S0"])
    r = float(theta["r"])
    sigma = float(theta["sigma"])

    # Find unique maturities to avoid redundant FFT calculations
    unique_T = np.unique(maturities)
    model_prices = np.empty_like(market_prices, dtype=float)

    # Compute model prices for each maturity
    for T in unique_T:
        mask = (maturities == T)

        params_T = {"S0": S0, "r": r, "sigma": sigma, "T": float(T)}

        model_prices[mask] = _bs_fft_prices_on_strikes(
            params_T, strikes[mask], alpha=alpha, N=N, eta=eta
        )

    residuals = model_prices - market_prices

    # Return mean squared error
    return float(np.mean(residuals**2))
