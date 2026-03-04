import unittest
import numpy as np
from scipy.stats import norm
import sys
import os

# --- ENVIRONMENT SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

from fourier_options.greeks.fft import delta_fft_bs, gamma_fft_bs


# --- 1. ANALYTICAL BENCHMARKS ---
def bs_delta(S, K, T, r, sigma):
    """Exact Black-Scholes Delta for a European call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    """Exact Black-Scholes Gamma for a European call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# --- 2. TEST SUITE ---
class TestGreeksFFT(unittest.TestCase):

    def setUp(self):
        """Initializes the environment, runs FFT Greeks, and filters data."""
        self.params = {
            "S0": 100.0,
            "r": 0.05,
            "T": 1.0,
            "sigma": 0.2
        }
        self.alpha = 1.5
        self.N = 4096
        self.eta = 0.25
        S0 = self.params["S0"]

        # Compute Delta and Gamma via FFT
        self.strikes_delta, self.raw_delta = delta_fft_bs(
            self.params, alpha=self.alpha, N=self.N, eta=self.eta
        )
        self.strikes_gamma, self.raw_gamma = gamma_fft_bs(
            self.params, alpha=self.alpha, N=self.N, eta=self.eta
        )

        # Filter to financially relevant range [60%, 140%] of Spot
        self.mask = (self.strikes_delta > S0 * 0.6) & (self.strikes_delta < S0 * 1.4)
        self.k_clean = self.strikes_delta[self.mask]
        self.delta_clean = self.raw_delta[self.mask]
        self.gamma_clean = self.raw_gamma[self.mask]

        # Analytical benchmarks on the filtered grid
        self.bs_delta = bs_delta(S0, self.k_clean, self.params["T"],
                                 self.params["r"], self.params["sigma"])
        self.bs_gamma = bs_gamma(S0, self.k_clean, self.params["T"],
                                 self.params["r"], self.params["sigma"])

    def test_1_delta_convergence(self):
        """FFT Delta must converge to analytical BS Delta."""
        errors = np.abs(self.delta_clean - self.bs_delta)
        max_err = np.max(errors)
        self.assertTrue(max_err < 0.01,
                        f"Delta max error too high: {max_err:.6f}")

    def test_2_delta_bounds(self):
        """Call Delta must lie in [0, 1]."""
        self.assertTrue(np.all(self.delta_clean >= -1e-6),
                        "Delta below 0 detected.")
        self.assertTrue(np.all(self.delta_clean <= 1.0 + 1e-6),
                        "Delta above 1 detected.")

    def test_3_delta_monotonicity(self):
        """Call Delta must be decreasing w.r.t. strike."""
        diffs = np.diff(self.delta_clean)
        self.assertTrue(np.all(diffs <= 1e-6),
                        "Delta monotonicity violation detected.")

    def test_4_gamma_convergence(self):
        """FFT Gamma must converge to analytical BS Gamma."""
        errors = np.abs(self.gamma_clean - self.bs_gamma)
        max_err = np.max(errors)
        self.assertTrue(max_err < 0.001,
                        f"Gamma max error too high: {max_err:.6f}")

    def test_5_gamma_non_negative(self):
        """Call Gamma must be non-negative (convexity of price w.r.t. S)."""
        self.assertTrue(np.all(self.gamma_clean >= -1e-6),
                        "Negative Gamma detected.")


if __name__ == '__main__':
    unittest.main()
