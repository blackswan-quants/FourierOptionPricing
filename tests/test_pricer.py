# Tests placeholder: test_pricer.py
import unittest
import numpy as np
from scipy.stats import norm


# --- 1. GENERATORE DI DATI SINTETICI ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Genera prezzi 'corretti' usando Black-Scholes per testare le nostre funzioni di verifica.
    Non è il modello FFT, serve solo come 'ground truth' per i test.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# --- 2. FUNZIONI DI VERIFICA (IL TUO TASK) ---

def check_put_call_parity(call_prices, put_prices, S, K_array, r, T, tolerance=1e-5):
    """
    Verifica: Call - Put = S - K * exp(-rT)
    """
    lhs = call_prices - put_prices
    rhs = S - K_array * np.exp(-r * T)

    # Verifica che la differenza sia quasi zero (entro la tolleranza)
    errors = np.abs(lhs - rhs)
    is_valid = np.all(errors < tolerance)
    return is_valid, errors


def check_monotonicity(prices, option_type='call'):
    """
    Call: Prezzo scende se K sale.
    Put: Prezzo sale se K sale.
    Assumiamo che l'array 'prices' sia ordinato per Strike crescente.
    """
    differences = np.diff(prices)  # Calcola price[i+1] - price[i]

    if option_type == 'call':
        # Le differenze devono essere negative o zero (decrescente)
        is_valid = np.all(differences <= 1e-9)
    else:
        # Le differenze devono essere positive o zero (crescente)
        is_valid = np.all(differences >= -1e-9)

    return is_valid


def check_convexity(prices):
    """
    Verifica che la curva dei prezzi sia convessa rispetto allo Strike.
    Approssimazione discreta della derivata seconda >= 0.
    Price(K-1) - 2*Price(K) + Price(K+1) >= 0
    """
    # Calcola la derivata seconda discreta
    convexity_check = prices[:-2] - 2 * prices[1:-1] + prices[2:]

    # Deve essere >= 0 (con una piccola tolleranza per errori di float)
    is_valid = np.all(convexity_check >= -1e-9)
    return is_valid


# --- 3. TEST SUITE (JUNIT STYLE) ---

class TestFinancialProperties(unittest.TestCase):

    def setUp(self):
        # Setup dei dati comuni prima di ogni test
        self.S0 = 100.0  # Spot price
        self.r = 0.05  # Risk-free rate
        self.T = 1.0  # Maturity (1 anno)
        self.sigma = 0.2  # Volatilità

        # Creiamo un range di Strike prices (es. da 50 a 150)
        self.K_array = np.linspace(50, 150, 100)

        # Generiamo i prezzi sintetici (Calls e Puts)
        self.call_prices = black_scholes_price(self.S0, self.K_array, self.T, self.r, self.sigma, 'call')
        self.put_prices = black_scholes_price(self.S0, self.K_array, self.T, self.r, self.sigma, 'put')

    def test_put_call_parity(self):
        print("\nTesting Put-Call Parity...")
        is_valid, errors = check_put_call_parity(
            self.call_prices, self.put_prices, self.S0, self.K_array, self.r, self.T
        )
        self.assertTrue(is_valid, f"Put-Call Parity violata! Max errore: {np.max(errors)}")

    def test_monotonicity_call(self):
        print("Testing Monotonicity (Call)...")
        is_valid = check_monotonicity(self.call_prices, 'call')
        self.assertTrue(is_valid, "Le Call non sono decrescenti rispetto allo Strike")

    def test_monotonicity_put(self):
        print("Testing Monotonicity (Put)...")
        is_valid = check_monotonicity(self.put_prices, 'put')
        self.assertTrue(is_valid, "Le Put non sono crescenti rispetto allo Strike")

    def test_convexity(self):
        print("Testing Convexity...")
        is_valid = check_convexity(self.call_prices)
        self.assertTrue(is_valid, "La curva dei prezzi Call non è convessa")


if __name__ == '__main__':
    unittest.main()