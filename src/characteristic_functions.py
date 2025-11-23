# Placeholder for characteristic_functions.py — Fourier Option Pricing Project (Sprint 2)
import numpy as np

def cf_bs(u, S_0, T, r, sigma):
    """
    Characteristic function of log-price under BS model.

    | Variable | Meaning                                                         |
    | -------- | --------------------------------------------------------------- |
    | (S_0)    | Initial price of the asset (e.g., a stock)                      |
    | (r)      | Annual risk-free interest rate                                  |
    | (\sigma) | Annual volatility of the asset                                  |
    | (T)      | Time to maturity (in years)                                     |
    | (u)      | Real variable over which the characteristic function is defined |

    Requires: numpy
    """
    mu = np.log(S_0) + (r - 0.5 * sigma**2) * T
    return np.exp(1j * u * mu - 0.5 * sigma**2 * u**2 * T) 
    