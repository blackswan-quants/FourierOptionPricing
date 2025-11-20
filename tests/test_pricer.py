# Tests placeholder: test_pricer.py
import unittest
import numpy as np
from scipy.stats import norm


# --- 1. SYNTHETIC DATA GENERATOR ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Generates 'ground truth' Black-Scholes prices for testing verification logic.
    This is NOT the FFT model; it is used solely for validation.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# --- 2. VERIFICATION FUNCTIONS (YOUR TASK) ---

def check_put_call_parity(call_prices, put_prices, S, K_array, r, T, tolerance=1e-5):
    """
    Verifies Put-Call Parity: Call - Put = S - K * exp(-rT).
    Returns validity boolean and error array.
    """
    lhs = call_prices - put_prices
    rhs = S - K_array * np.exp(-r * T)

    # Check if difference is within tolerance
    errors = np.abs(lhs - rhs)
    is_valid = np.all(errors < tolerance)
    return is_valid, errors


def check_monotonicity(prices, option_type='call'):
    """
    Checks price monotonicity w.r.t. Strike.
    - Call: Prices must decrease (or stay flat) as K increases.
    - Put: Prices must increase (or stay flat) as K increases.
    Assumes 'prices' array is sorted by ascending Strike.
    """
    differences = np.diff(prices)  # price[i+1] - price[i]

    if option_type == 'call':
        # Differences must be <= 0
        is_valid = np.all(differences <= 1e-9)
    else:
        # Differences must be >= 0
        is_valid = np.all(differences >= -1e-9)

    return is_valid


def check_convexity(prices):
    """
    Verifies price curve convexity w.r.t. Strike.
    Uses discrete 2nd derivative approximation: Price(K-1) - 2*Price(K) + Price(K+1) >= 0.
    Equivalent to checking for non-negative Butterfly Spread costs.
    """
    # Vectorized discrete 2nd derivative
    convexity_check = prices[:-2] - 2 * prices[1:-1] + prices[2:]

    # Must be >= 0 (allowing for small float tolerance)
    is_valid = np.all(convexity_check >= -1e-9)
    return is_valid


# --- 3. TEST SUITE (JUNIT STYLE) ---

class TestFinancialProperties(unittest.TestCase):

    def setUp(self):
        # Initialize common parameters before each test
        self.S0 = 100.0
        self.r = 0.05
        self.T = 1.0
        self.sigma = 0.2

        # Create Strike range (e.g., 50 to 150)
        self.K_array = np.linspace(50, 150, 100)

        # Generate synthetic Call and Put prices (Ground Truth)
        self.call_prices = black_scholes_price(self.S0, self.K_array, self.T, self.r, self.sigma, 'call')
        self.put_prices = black_scholes_price(self.S0, self.K_array, self.T, self.r, self.sigma, 'put')

    def test_put_call_parity(self):
        print("\nTesting Put-Call Parity...")
        is_valid, errors = check_put_call_parity(
            self.call_prices, self.put_prices, self.S0, self.K_array, self.r, self.T
        )
        self.assertTrue(is_valid, f"Put-Call Parity violated! Max error: {np.max(errors)}")

    def test_monotonicity_call(self):
        print("Testing Monotonicity (Call)...")
        is_valid = check_monotonicity(self.call_prices, 'call')
        self.assertTrue(is_valid, "Call prices are not decreasing w.r.t. Strike")

    def test_monotonicity_put(self):
        print("Testing Monotonicity (Put)...")
        is_valid = check_monotonicity(self.put_prices, 'put')
        self.assertTrue(is_valid, "Put prices are not increasing w.r.t. Strike")

    def test_convexity(self):
        print("Testing Convexity...")
        is_valid = check_convexity(self.call_prices)
        self.assertTrue(is_valid, "Call price curve is not convex")


if __name__ == '__main__':
    unittest.main()