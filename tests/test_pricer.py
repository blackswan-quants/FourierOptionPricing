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

from fft_pricer import fft_pricer

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
def check_monotonicity(prices, option_type='call'):
    """
    Verifies that prices allow for no static arbitrage (monotonicity).
    Call: Decreasing w.r.t Strike. Put: Increasing w.r.t Strike.
    """
    diffs = np.diff(prices)
    # Tolerance 1e-9 accounts for floating point noise
    if option_type == 'call':
        return np.all(diffs <= 1e-9)
    else:
        return np.all(diffs >= -1e-9)

def check_convexity(prices):
    """
    Verifies convexity (Butterfly Spread cost >= 0).
    Essential to ensure non-negative probability densities.
    """
    # Discrete 2nd derivative: P(K-1) - 2P(K) + P(K+1)
    butterfly_cost = prices[:-2] - 2 * prices[1:-1] + prices[2:]
    return np.all(butterfly_cost >= -1e-9)


# --- 3. TEST SUITE ---
class TestFFTImplementation(unittest.TestCase):

    def setUp(self):
        """
        Initializes the environment, runs the FFT engine, and filters data.
        """
        # Market Parameters
        self.S0 = 100.0
        self.r = 0.05
        self.T = 1.0
        self.sigma = 0.2

        # Numerical Parameters (Carr-Madan)
        self.alpha = 1.25
        self.N = 4096
        self.eta = 0.25

        # Run FFT Engine
        self.raw_strikes, self.raw_prices = fft_pricer(
            self.S0, self.r, self.T, self.sigma, 
            alpha=self.alpha, N=self.N, eta=self.eta
        )

        # Data Filtering (Crucial):
        # Remove extreme tails to avoid FFT aliasing errors and Gibbs phenomenon.
        # We focus on the financially relevant range [40% to 180% of Spot].
        mask = (self.raw_strikes > self.S0 * 0.4) & (self.raw_strikes < self.S0 * 1.8)
        
        self.k_clean = self.raw_strikes[mask]
        self.prices_clean = self.raw_prices[mask]

        # Compute Benchmark on the filtered grid only
        self.bs_benchmark = black_scholes_price(
            self.S0, self.k_clean, self.T, self.r, self.sigma, 'call'
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
        is_valid = check_monotonicity(self.prices_clean, 'call')
        self.assertTrue(is_valid, "Monotonicity violation detected.")

    def test_3_convexity(self):
        """
        Verifies convexity (absence of butterfly arbitrage).
        """
        # Note: If this fails, consider adjusting alpha or increasing N
        is_valid = check_convexity(self.prices_clean)
        self.assertTrue(is_valid, "Convexity violation (Negative probabilities) detected.")

if __name__ == '__main__':
    unittest.main()