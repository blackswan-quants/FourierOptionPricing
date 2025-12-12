import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs


# Model setup
params = {
    "S0": 100.0,
    "r": 0.01,
    "T": 1.0,
    "sigma": 0.20 # true vol used to generate synthetic prices
}

# Generate synthetic market prices via FFT
K, C_market = fft_pricer(
    cf_bs,
    params,
    alpha=1.5,
    N=4096,
    eta=0.25
)


#Loss function
def loss_function(sigma_guess: NDArray[np.float64]) -> float:
    """
    Compute quadratic pricing error for a given volatility guess.

    Parameters
    ----------
    sigma_guess : NDArray[np.float64]
        Array of shape (1,) containing the trial volatility parameter.

    Returns
    -------
    float
        Sum of squared pricing errors between model prices and market prices.
    """

    params["sigma"] = sigma_guess[0]

    # Compute option prices using trial volatility
    _, C_model = fft_pricer(
        cf_bs,
        params,
        alpha=1.5,
        N=4096,
        eta=0.25
    )

    # Quadratic pricing error
    return np.sum((C_model - C_market)**2)

# Run volatility calibration using numerical optimization
result = minimize(
    loss_function,
    x0=np.array([0.10]),      # initial volatility guess
    bounds=[(0.001, 1.0)]     # reasonable bounds for volatility
)

sigma_calibrated = result.x[0]

print("Calibrated volatility:", sigma_calibrated)


