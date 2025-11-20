import unittest
import numpy as np
from scipy.stats import norm

# --- IMPORT THE ENGINE ---
# Assuming the code you pasted is in 'pricing_engine.py'
# If it's in the same file, just ensure the function is defined above.
from fft_pricer import fft_pricer 

# --- 1. GROUND TRUTH GENERATOR (Standard BS) ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Analytical Black-Scholes pricer. 
    Used as the 'Ground Truth' benchmark to validate FFT accuracy.
    """
    # Avoid division by zero for very small T
    if T <= 1e-8:
        return np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

# --- 2. VERIFICATION LOGIC ---
def check_monotonicity(prices, option_type='call'):
    """
    Checks if prices represent a valid cumulative distribution function proxy.
    Call: Decreasing w.r.t Strike. Put: Increasing w.r.t Strike.
    """
    differences = np.diff(prices)
    if option_type == 'call':
        # Allow small float error (1e-9)
        return np.all(differences <= 1e-9)
    else:
        return np.all(differences >= -1e-9)

def check_convexity(prices):
    """
    Checks for Arbitrage opportunities (Butterfly Spread).
    Discrete 2nd derivative must be >= 0.
    """
    # vector[i-1] - 2*vector[i] + vector[i+1]
    convexity = prices[:-2] - 2 * prices[1:-1] + prices[2:]
    return np.all(convexity >= -1e-9)

# --- 3. INTEGRATION TEST SUITE ---
class TestFFTImplementation(unittest.TestCase):

    def setUp(self):
        """
        PREPARE THE TEST ENVIRONMENT.
        1. Define financial parameters.
        2. Run the FFT Engine to get the grid and prices.
        3. Filter the output to remove numerical noise at tails.
        4. Compute Benchmark (BS) on the valid grid.
        """
        # Market Params
        self.S0 = 100.0
        self.r = 0.05
        self.T = 1.0
        self.sigma = 0.2

        # FFT Numerical Params (Standard settings)
        self.alpha = 1.5
        self.N = 4096     # 2^12
        self.eta = 0.25

        # --- EXECUTE ENGINE ---
        # Capture both the Strike Grid (K) and Prices
        self.raw_strikes, self.raw_prices = fft_pricer(
            self.S0, self.r, self.T, self.sigma, 
            alpha=self.alpha, N=self.N, eta=self.eta
        )

        # --- FILTERING STAGE (CRITICAL) ---
        # FFT produces garbage at K -> 0 and K -> infinity due to aliasing.
        # We strictly test only the financially relevant range (e.g., 0.5*S0 to 1.5*S0).
        
        mask = (self.raw_strikes > self.S0 * 0.4) & (self.raw_strikes < self.S0 * 1.8)
        
        self.k_clean = self.raw_strikes[mask]
        self.prices_clean = self.raw_prices[mask]
        
        '''
        self.k_clean = self.raw_strikes  # Prendi tutto!
        self.prices_clean = self.raw_prices # Prendi tutto!
        '''

        # --- GENERATE BENCHMARK ---
        # Compute exact BS prices ONLY for the filtered strikes
        self.bs_benchmark = black_scholes_price(
            self.S0, self.k_clean, self.T, self.r, self.sigma, 'call'
        )

    def test_1_convergence(self):
        """
        Does FFT match Black-Scholes analytical formula?
        Pass condition: Max absolute error < tolerance.
        """
        print("\n[Test] Convergence to Black-Scholes...")
        errors = np.abs(self.prices_clean - self.bs_benchmark)
        max_err = np.max(errors)
        avg_err = np.mean(errors)
        
        print(f"   Max Error: {max_err:.6f}")
        print(f"   Avg Error: {avg_err:.6f}")

        # Tolerance: 1 cent (0.01) is usually acceptable for FFT with N=4096
        self.assertTrue(max_err < 0.01, f"FFT diverges! Max error: {max_err}")

    def test_2_monotonicity(self):
        """
        Does the FFT pricing curve strictly decrease as Strike increases?
        """
        print("[Test] Monotonicity Check...")
        is_valid = check_monotonicity(self.prices_clean, 'call')
        self.assertTrue(is_valid, "FFT prices are not monotonic.")

    def test_3_convexity(self):
        """
        Does the FFT pricing curve satisfy No-Arbitrage (Convexity)?
        """
        print("[Test] Convexity Check...")
        is_valid = check_convexity(self.prices_clean)
        
        # If this fails, check alpha or eta.
        self.assertTrue(is_valid, "FFT prices violate convexity (Butterfly Arbitrage).")

if __name__ == '__main__':
    unittest.main()