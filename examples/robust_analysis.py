import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy.stats import norm

# Standard Layout
plt.style.use('seaborn-v0_8-darkgrid')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fourier_options.pricing import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs

# ---------------------------------------------------------
# Local Pricers for Benchmarking
# ---------------------------------------------------------

def bs_call_analytical(S0, K, T, r, sigma):
    """Vectorized BS Call Price"""
    # Avoid div by zero
    T = np.maximum(T, 1e-9)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_mc(S0, K, T, r, sigma, n_sims=100_000):
    """Simple Monte Carlo BS Call Price (returns single value)"""
    z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * np.mean(payoff)

# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------
def main():
    print("Starting Robust Analysis & Benchmarking...")
    
    # Base Params
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    
    # Fixed FFT Grid Params
    N = 4096
    eta = 0.25

    # ---------------------------------------------------------
    # 1. Extended Alpha Sensitivity
    # ---------------------------------------------------------
    print("\n[1/3] Extended Alpha Sensitivity (0.1 - 5.0)...")
    base_T = 1.0
    params = {"S0": S0, "r": r, "T": base_T, "sigma": sigma}
    
    alpha_range = np.linspace(0.1, 5.0, 50)
    rmse_list = []
    
    # Grid for error checking (Standard Moneyness)
    # We need to run FFT once to get the K grid? 
    # Actually K grid depends on N, eta. It's constant.
    # Let's get the K grid first.
    k_grid_ref, _ = fft_pricer.fft_pricer(cf_bs, params, 1.5, N, eta)
    mask = (k_grid_ref >= 50) & (k_grid_ref <= 150)
    k_eval = k_grid_ref[mask]
    
    # Ground Truth
    ref_prices = bs_call_analytical(S0, k_eval, base_T, r, sigma)
    
    for a in alpha_range:
        try:
            _, prices = fft_pricer.fft_pricer(cf_bs, params, a, N, eta)
            p_subset = prices[mask]
            err = np.sqrt(np.mean((p_subset - ref_prices)**2))
            rmse_list.append(err)
        except Exception as e:
            rmse_list.append(np.nan)
            
    # Plot Alpha
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_range, rmse_list, '.-')
    plt.yscale('log')
    plt.title('FFT RMS Error vs Damping Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE (Log Scale)')
    plt.tight_layout()
    plt.savefig("robust_alpha_sensitivity.svg", format="svg")
    print("Saved 'robust_alpha_sensitivity.svg'")

    # ---------------------------------------------------------
    # 2. Parameter Sensitivity (Maturity vs Error)
    # ---------------------------------------------------------
    print("\n[2/3] Maturity Sensitivity...")
    T_range = [0.1, 0.5, 1.0, 2.0, 5.0]
    alpha_best = 1.5
    
    print(f"{'Maturity':<10} {'RMSE':<15} {'MaxErr':<15}")
    for t_val in T_range:
        params["T"] = t_val
        _, p_fft = fft_pricer.fft_pricer(cf_bs, params, alpha_best, N, eta)
        p_eval = p_fft[mask]
        ref = bs_call_analytical(S0, k_eval, t_val, r, sigma)
        
        diff = np.abs(p_eval - ref)
        rmse = np.sqrt(np.mean(diff**2))
        max_e = np.max(diff)
        print(f"{t_val:<10.1f} {rmse:<15.2e} {max_e:<15.2e}")

    # ---------------------------------------------------------
    # 3. Computational Benchmark
    # ---------------------------------------------------------
    print("\n[3/3] Pricing Speed Benchmark...")
    params["T"] = 1.0
    
    # A) FFT Speed
    # Returns N prices at once
    n_loops = 50
    t0 = time.perf_counter()
    for _ in range(n_loops):
        fft_pricer.fft_pricer(cf_bs, params, alpha_best, N, eta)
    t_fft_total = time.perf_counter() - t0
    # Time per CALL (since it returns N prices, but usually we only want a subset. 
    # Let's count "Time to price the curve").
    time_fft = t_fft_total / n_loops
    
    # B) Analytical Speed
    # Calculate for the SAME N grid points
    full_k_grid = k_grid_ref
    t0 = time.perf_counter()
    for _ in range(n_loops):
        bs_call_analytical(S0, full_k_grid, 1.0, r, sigma)
    t_anal_total = time.perf_counter() - t0
    time_anal = t_anal_total / n_loops

    # C) Monte Carlo Speed
    # Standard MC is very slow if we do it for EVERY strike individually.
    # We will do it for just ONE strike (ATM) to be fair to the "Iterative" approach logic,
    # or Vectorized MC? Let's do scalar MC for 1 strike to show the "per option" cost vs FFT "batch" cost.
    n_mc_loops = 10
    t0 = time.perf_counter()
    for _ in range(n_mc_loops):
        bs_call_mc(S0, 100.0, 1.0, r, sigma, n_sims=100_000)
    t_mc_total = time.perf_counter() - t0
    time_mc = t_mc_total / n_mc_loops

    print("-" * 60)
    print(f"{'Method':<20} {'Time (per run)':<20} {'Notes':<20}")
    print("-" * 60)
    print(f"{'Analytical (Exact)':<20} {time_anal*1000:.4f} ms {'~4096 prices'}")
    print(f"{'FFT (Carr-Madan)':<20} {time_fft*1000:.4f} ms {'~4096 prices'}")
    print(f"{'Monte Carlo (10^5)':<20} {time_mc*1000:.4f} ms {'SINGLE price'}")
    print("-" * 60)
    print("Note: FFT generates the entire option chain in one go.")
    print(f"Speedup FFT vs MC (1 price): {time_mc/time_fft:.2f}x (showing batch efficiency)")

if __name__ == "__main__":
    main()
