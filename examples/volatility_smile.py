
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fourier_options.pricing import fft_pricer
from fourier_options.domain.characteristic_functions import cf_heston

def bs_call_price(sigma, S0, K, T, r):
    """
    Computes the analytical Black-Scholes price for a European Call option.
    Used for implicit volatility inversion.
    """
    if sigma <= 1e-6:
        return max(S0 * np.exp(-r*0) - K * np.exp(-r*T), 0.0) # Approx intrinsic if vol is zero

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S0, K, T, r):
    """
    Computes Implied Volatility for a Call Option using Brent's method.
    """
    def objective(sigma):
        return bs_call_price(sigma, S0, K, T, r) - price

    # Check boundaries
    # Theoretical max price is S0, min is max(S0 - K*exp(-rT), 0)
    # If price is outside rational bounds, return NaN
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-6:
        return 1e-6 # Very low vol
    if price >= S0:
        return np.nan 

    try:
        # Solve for sigma where BS_price(sigma) == Market_price
        # Search range [0.01%, 500%]
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan

def main():
    # -------------------------------------------------------------------------
    # 1. Setup Heston Parameters
    # -------------------------------------------------------------------------
    S0 = 100.0
    r = 0.05
    T = 1.0 # 1 year maturity
    
    # Heston Parameters designed to show a smile/skew
    params = {
        "S0": S0,
        "r": r,
        "T": T,
        "kappa": 2.0,       # Mean reversion speed
        "theta": 0.04,      # Long run variance (sqrt(0.04) = 0.2 = 20% vol)
        "sigma_v": 0.5,      # Vol of Vol (high for smile effect)
        "rho": -0.7,        # Correlation (negative for equity skew)
        "v0": 0.04          # Initial variance
    }
    
    # FFT settings
    N = 4096
    alpha = 1.5
    eta = 0.25

    print("Generating Heston Prices via FFT...")
    
    # CASE 1: Skew (Equity-like)
    params_skew = params.copy()
    params_skew["rho"] = -0.7
    
    # CASE 2: Smile (Symmetric-like)
    params_smile = params.copy()
    params_smile["rho"] = 0.0

    # Retrieve prices for both
    _, prices_skew = fft_pricer.fft_pricer(cf_heston, params_skew, alpha, N, eta)
    k_fft, prices_smile = fft_pricer.fft_pricer(cf_heston, params_smile, alpha, N, eta)

    # Filter
    mask = (k_fft >= 80) & (k_fft <= 120)
    strikes = k_fft[mask]
    prices_skew = prices_skew[mask]
    prices_smile = prices_smile[mask]

    # Calculate IVs
    ivs_skew = []
    ivs_smile = []
    
    for K, P in zip(strikes, prices_skew):
        iv = implied_volatility(P, S0, K, T, r)
        ivs_skew.append(iv if not np.isnan(iv) else 0.0)

    for K, P in zip(strikes, prices_smile):
        iv = implied_volatility(P, S0, K, T, r)
        ivs_smile.append(iv if not np.isnan(iv) else 0.0)
        
    ivs_skew = np.array(ivs_skew)
    ivs_smile = np.array(ivs_smile)

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, ivs_skew, 'o-', label=f'Skew: rho={params_skew["rho"]}')
    plt.plot(strikes, ivs_smile, 's-', label=f'Smile: rho={params_smile["rho"]}')
    plt.axhline(np.sqrt(params['theta']), color='r', linestyle='--', label='Long-run Vol')
    
    plt.title(f'Volatility Smile/Skew (Heston Model)\nVol-of-Vol={params["sigma_v"]}')
    plt.xlabel('Strike Price ($K$)')
    plt.ylabel('Implied Volatility ($\sigma_{imp}$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("volatility_smile.svg", format="svg")
    print("Plot saved to volatility_smile.svg")
    plt.show()

    print("Plot generated. Skew case (rho=-0.7) should be linear-ish downwards.")
    print("Smile case (rho=0.0) should be a symmetric 'U' shape.")

if __name__ == "__main__":
    main()
