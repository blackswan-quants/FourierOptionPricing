import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os

# Add src to path so we can import fourier_options
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fourier_options.pricing import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs

def bs_call_price(S0, K, T, r, sigma):
    """
    Computes the analytical Black-Scholes price for a European Call option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def main():
    # -------------------------------------------------------------------------
    # 1. Setup Parameters
    # -------------------------------------------------------------------------
    S0 = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    params = {
        "S0": S0,
        "r": r,
        "T": T,
        "sigma": sigma
    }

    # FFT Parameters (Standard Carr-Madan choice)
    # N=4096 is usually sufficient for high accuracy
    N = 4096
    alpha = 1.5
    eta = 0.25

    print(f"Running FFT Error Analysis with parameters:")
    print(f"S0={S0}, r={r}, T={T}, sigma={sigma}")
    print(f"FFT(N={N}, alpha={alpha}, eta={eta})")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # 2. Compute Prices
    # -------------------------------------------------------------------------
    
    # Run FFT Pricing
    # Returns the full grid of strikes (K_fft) and prices (Price_fft)
    k_fft, price_fft = fft_pricer.fft_pricer(
        cf=cf_bs, 
        params=params, 
        alpha=alpha, 
        N=N, 
        eta=eta
    )

    # Filter to a reasonable range of strikes for analysis, e.g., 0.5*S0 to 1.5*S0
    # or a specific relevant range typical for trading.
    mask = (k_fft >= 80) & (k_fft <= 120)
    
    selected_k = k_fft[mask]
    selected_fft_prices = price_fft[mask]

    # Compute Analytical BS Prices for the exact same strikes
    analytical_prices = bs_call_price(S0, selected_k, T, r, sigma)

    # -------------------------------------------------------------------------
    # 3. Error Analysis
    # -------------------------------------------------------------------------
    
    abs_error = np.abs(selected_fft_prices - analytical_prices)
    rel_error = abs_error / analytical_prices

    # Create a DataFrame for nice display
    df_results = pd.DataFrame({
        "Strike": selected_k,
        "FFT Price": selected_fft_prices,
        "BS Analytical": analytical_prices,
        "Abs Error": abs_error,
        "Rel Error": rel_error
    })

    # Summary Statistics
    max_abs_err = np.max(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))
    mean_rel_err = np.mean(rel_error)

    print("\n--- Error Statistics (Strikes 80 to 120) ---")
    print(f"Max Absolute Error: {max_abs_err:.2e}")
    print(f"RMSE:               {rmse:.2e}")
    print(f"Mean Relative Err:  {mean_rel_err:.2e}")

    print("\n--- Detailed Results (Sample) ---")
    # Show a few specific strikes
    # Find indices closest to 80, 90, 100, 110, 120
    target_strikes = [80, 90, 100, 110, 120]
    indices = [np.abs(selected_k - k).argmin() for k in target_strikes]
    
    # -------------------------------------------------------------------------
    # 4. Alpha Sensitivity Analysis
    # -------------------------------------------------------------------------
    print("\n--- Alpha Sensitivity Analysis ---")
    alpha_values = np.linspace(0.5, 3.0, 20)
    rmse_values = []
    
    # Pre-compute BS prices for the specific strike range once (approximation: grid changes slightly with alpha?)
    # Actually, the grid K depends on eta and N, NOT on alpha.
    # N, eta are fixed. lambda = 2pi/(N*eta). k grid is fixed. 
    # So K grid is CONSTANT for fixed N, eta.
    # We can use the same analytical_prices.
    
    for a in alpha_values:
        _, p_fft = fft_pricer.fft_pricer(cf_bs, params, a, N, eta)
        p_sel = p_fft[mask]
        
        # Error metric: RMSE
        # Note: analytical_prices was computed on 'selected_k' which is from the same grid
        curr_error = np.sqrt(np.mean((p_sel - analytical_prices)**2))
        rmse_values.append(curr_error)
        print(f"Alpha: {a:.2f} | RMSE: {curr_error:.4e}")

    # Plot Error vs Alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, rmse_values, 'o-', color='purple')
    plt.yscale('log')
    plt.title(f'FFT Error (RMSE) vs Damping Parameter Alpha\nBS Model: T={T}, sigma={sigma}, r={r}')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("fft_error_vs_alpha.svg", format="svg")
    print("\nAlpha sensitivity plot saved to 'fft_error_vs_alpha.svg'")
    plt.show()

if __name__ == "__main__":
    main()
