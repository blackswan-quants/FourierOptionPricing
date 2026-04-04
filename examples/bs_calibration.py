import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs


TRUE_SIGMA = 0.20

# Generate synthetic market prices with the known true volatility
market_params = {"S0": 100.0, "r": 0.01, "T": 1.0, "sigma": TRUE_SIGMA}
K_market, C_market = fft_pricer(cf_bs, market_params, alpha=1.5, N=4096, eta=0.25)


def loss_function(sigma_guess: NDArray[np.float64]) -> float:
    """
    Quadratic pricing error for a given volatility guess.

    A fresh params dict is built each call so the market data is never
    mutated by the optimizer.
    """
    params = {"S0": 100.0, "r": 0.01, "T": 1.0, "sigma": float(sigma_guess[0])}
    _, C_model = fft_pricer(cf_bs, params, alpha=1.5, N=4096, eta=0.25)
    return float(np.sum((C_model - C_market) ** 2))


result = minimize(
    loss_function,
    x0=np.array([0.10]),
    bounds=[(0.001, 1.0)],
)

sigma_calibrated = result.x[0]
print(f"True volatility  : {TRUE_SIGMA:.4f}")
print(f"Calibrated sigma : {sigma_calibrated:.4f}")
print(f"Error            : {abs(sigma_calibrated - TRUE_SIGMA):.2e}")
