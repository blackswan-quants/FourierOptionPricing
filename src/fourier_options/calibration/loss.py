import numpy as np
from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs


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

from scipy.optimize import differential_evolution
from fourier_options.domain.characteristic_functions import cf_heston

def _heston_fft_prices_on_strikes(
    params: dict[str, float],
    strikes: np.ndarray,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> np.ndarray:
    """
    Heston call prices via Carr-Madan FFT, interpolated to given strikes.
    
    Args:
        params (dict): Heston model parameters (S0, r, T, kappa, theta, sigma_v, rho, v0)
        strikes (np.ndarray): Target strike prices for interpolation
        alpha (float): Damping parameter for Carr-Madan formula
        N (int): Number of FFT points (must be power of 2)
        eta (float): Grid spacing in log-strike space
        
    Returns:
        np.ndarray: Call option prices interpolated at the given strikes
    """
    # Compute prices on the regular FFT log-strike grid
    K_grid, C_grid = fft_pricer(cf_heston, params, alpha=alpha, N=N, eta=eta)
    strikes = np.asarray(strikes, dtype=float)

    # Linear interpolation in log-strike space to map FFT grid to actual market strikes
    logK = np.log(strikes)
    return np.interp(logK, np.log(K_grid), C_grid)


def heston_weighted_loss(
    theta_array: np.ndarray,
    S0: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    bid_ask_spreads: np.ndarray,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> float:
    """
    Computes the weighted MSE between market prices and Heston model prices.
    Includes a penalty term for violating the Feller condition.
    
    Args:
        theta_array (np.ndarray): Array of parameters to calibrate [kappa, theta, sigma_v, rho, v0]
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        strikes, maturities, market_prices, bid_ask_spreads (np.ndarray): Market data
        alpha, N, eta: FFT numerical parameters
        
    Returns:
        float: The calculated weighted loss (plus any Feller penalty)
    """
    # Unpack Heston parameters
    kappa, theta, sigma_v, rho, v0 = theta_array
    
    # 1. Evaluate Feller Condition: 2 * kappa * theta > sigma_v^2
    # If violated, the variance process can hit zero. We penalize the loss heavily.
    feller_val = 2 * kappa * theta - sigma_v**2
    penalty = 0.0
    if feller_val <= 0:
        penalty = 1e6 * abs(feller_val) + 1000  # Stiff penalty to push optimizer away
        
    # Ensure data arrays are typed correctly
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)
    bid_ask_spreads = np.asarray(bid_ask_spreads, dtype=float)

    # Calculate weights based on liquidity (inverse of bid-ask spread)
    # Add a small epsilon (1e-5) to avoid division by zero
    weights = 1.0 / (bid_ask_spreads + 1e-5)
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    unique_T = np.unique(maturities)
    model_prices = np.empty_like(market_prices, dtype=float)

    # 2. Compute model prices for each unique maturity (batching for performance)
    for T in unique_T:
        mask = (maturities == T)
        
        # Build the dictionary for the current maturity
        params_T = {
            "S0": S0, "r": r, "T": float(T),
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v,
            "rho": rho, "v0": v0
        }
        
        # Call the FFT pricer
        model_prices[mask] = _heston_fft_prices_on_strikes(
            params_T, strikes[mask], alpha=alpha, N=N, eta=eta
        )

    # 3. Calculate Weighted MSE
    residuals = model_prices - market_prices
    weighted_mse = float(np.sum(weights * (residuals**2)))

    return weighted_mse + penalty


def calibrate_heston_model(
    S0: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    bid_ask_spreads: np.ndarray,
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25
) -> dict:
    """
    Calibrates the Heston model parameters to market data using Differential Evolution.
    
    Returns:
        dict: Containing the optimal parameters and the final loss value.
    """
    # Define bounds for [kappa, theta, sigma_v, rho, v0]
    # These bounds prevent the optimizer from testing mathematically impossible regions
    bounds = [
        (0.01, 10.0),    # kappa (mean reversion speed)
        (0.01, 1.0),     # theta (long-term variance)
        (0.01, 2.0),     # sigma_v (volatility of variance)
        (-0.99, 0.99),   # rho (correlation)
        (0.01, 1.0)      # v0 (initial variance)
    ]

    # Package the additional arguments required by the loss function
    args = (S0, r, strikes, maturities, market_prices, bid_ask_spreads, alpha, N, eta)

    print("Starting Heston calibration via Differential Evolution...")
    
    # Run the Differential Evolution optimizer
    result = differential_evolution(
        heston_weighted_loss,
        bounds=bounds,
        args=args,
        strategy='best1bin',
        maxiter=100,      # Adjust based on your time budget
        popsize=15,       # Multiplier for population size
        tol=1e-6,         # Convergence tolerance
        disp=True,        # Print progress
        workers=-1        # Use all available CPU cores for parallel evaluation
    )

    if result.success:
        print("Calibration successful!")
    else:
        print("Calibration failed to converge:", result.message)

    # Map optimal array back to a readable dictionary
    optimal_params = {
        "kappa": result.x[0],
        "theta": result.x[1],
        "sigma_v": result.x[2],
        "rho": result.x[3],
        "v0": result.x[4],
        "final_loss": result.fun
    }
    
    return optimal_params
