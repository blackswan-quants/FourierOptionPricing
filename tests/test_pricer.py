import unittest
import numpy as np
from scipy.stats import norm
import sys
import os

# --- ENVIRONMENT SETUP ---
# Add 'src' directory to system path to import the pricing engine
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs

# --- 1. ANALYTICAL BENCHMARK ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculates exact Black-Scholes prices to serve as Ground Truth.
    """
    if T <= 1e-8:
        return np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --- 2. FINANCIAL PROPERTIES CHECKS ---
def check_monotonicity(prices, option_type='call', tolerance=1e-6):
    """
    Verifies that prices allow for no static arbitrage (monotonicity).
    Call: Decreasing w.r.t Strike. Put: Increasing w.r.t Strike.
    """
    diffs = np.diff(prices)
    
    if option_type == 'call':
        # Prices should be decreasing (diffs <= 0)
        return np.all(diffs <= tolerance)
    else:
        # Prices should be increasing (diffs >= 0)
        return np.all(diffs >= -tolerance)

def check_convexity(prices, strikes=None, tolerance=1e-6):
    """
    Verifies convexity (Butterfly Spread cost >= 0).
    Essential to ensure non-negative probability densities.
    Uses divided differences to handle non-equally-spaced strikes (e.g., log-spaced FFT grids).
    """
    if strikes is None:
        # Equally-spaced case: simple discrete 2nd derivative
        butterfly_cost = prices[:-2] - 2 * prices[1:-1] + prices[2:]
    else:
        # General case: 2nd divided difference for unequal spacing
        dK_left  = strikes[1:-1] - strikes[:-2]   # K_{i} - K_{i-1}
        dK_right = strikes[2:]   - strikes[1:-1]   # K_{i+1} - K_{i}
        
        slope_left  = (prices[1:-1] - prices[:-2]) / dK_left
        slope_right = (prices[2:]   - prices[1:-1]) / dK_right
        
        butterfly_cost = slope_right - slope_left  # Must be >= 0 for convexity

    return np.all(butterfly_cost >= -tolerance)


# --- 3. TEST SUITE ---
class TestFFTImplementation(unittest.TestCase):

    def setUp(self):
        """
        Initializes the environment, runs the FFT engine, and filters data.
        """
        # Market Parameters
        self.params = {
            "S0": 100.0,
            "r": 0.05,
            "T": 1.0,
            "sigma": 0.2
        }

        # Numerical Parameters (Carr-Madan)
        self.alpha = 1.25
        self.N = 4096
        self.eta = 0.25

        # Run FFT Engine for Calls
        self.raw_strikes, self.raw_prices = fft_pricer(
            cf_bs, self.params, 
            alpha=self.alpha, N=self.N, eta=self.eta
        )

        # Data Filtering (Crucial):
        # Remove extreme tails to avoid FFT aliasing errors and Gibbs phenomenon.
        # We focus on the financially relevant range [40% to 180% of Spot].
        S0 = self.params["S0"]
        mask = (self.raw_strikes > S0 * 0.4) & (self.raw_strikes < S0 * 1.8)
        
        self.k_clean = self.raw_strikes[mask]
        self.prices_clean = self.raw_prices[mask]

        # Compute Benchmark on the filtered grid only
        self.bs_benchmark = black_scholes_price(
            S0, self.k_clean, self.params["T"], self.params["r"], self.params["sigma"], 'call'
        )

    def test_1_convergence(self):
        """
        Verifies that FFT prices match Analytical BS prices within tolerance.
        """
        errors = np.abs(self.prices_clean - self.bs_benchmark)
        max_err = np.max(errors)
        
        # Tolerance set to 0.01 (1 cent)
        self.assertTrue(max_err < 0.01, f"Max Error too high: {max_err:.6f}")

    def test_2_monotonicity(self):
        """
        Verifies strict monotonicity of the option curve.
        """
        is_valid = check_monotonicity(self.prices_clean, 'call', tolerance=1e-6)
        self.assertTrue(is_valid, "Monotonicity violation detected.")

    def test_3_convexity(self):
        """
        Verifies convexity (absence of butterfly arbitrage).
        """
        # Note: If this fails, consider adjusting alpha or increasing N
        is_valid = check_convexity(self.prices_clean, strikes=self.k_clean, tolerance=1e-6)
        self.assertTrue(is_valid, "Convexity violation (Negative probabilities) detected.")

    def test_4_put_call_parity(self):
        """
        Verifies Put-Call Parity: C - P = S0 - K * exp(-rT)
        """
        # Calculate Put prices using put-call parity (numerically more stable)
        # P = C - S0 + K*exp(-rT)
        raw_strikes_put, put_prices_raw = fft_pricer(
            cf_bs, self.params, 
            alpha=self.alpha, N=self.N, eta=self.eta, option_type='put'
        )
        
        # Filter puts with the same mask used for calls
        S0 = self.params["S0"]
        mask = (raw_strikes_put > S0 * 0.4) & (raw_strikes_put < S0 * 1.8)
        put_prices_clean = put_prices_raw[mask]
        
        # Verify put-call parity holds: C - P = S0 - K*exp(-rT)
        discounted_strike = self.k_clean * np.exp(-self.params["r"] * self.params["T"])
        parity_diff = self.prices_clean - put_prices_clean - S0 + discounted_strike
        
        max_parity_err = np.max(np.abs(parity_diff))
        
        # Tolerance reflects numerical precision (floating point rounding)
        self.assertTrue(max_parity_err < 1e-10, f"Put-Call Parity violation: {max_parity_err:.12f}")

if __name__ == '__main__':
    unittest.main()