import numpy as np
from scipy.stats import norm
import json
from greeks_integrands import delta_integrand


def bs_call_delta(params: dict[str, float], K: np.ndarray) -> np.ndarray:
    """
    Analytic Black–Scholes Delta for European calls (no dividends).
    
    Delta measures the sensitivity of option price to changes in the underlying
    stock price. Uses the closed-form formula Delta = N(d1).
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        
    Args:
        params (dict): Model parameters
        K (np.ndarray): Strike prices
        
    Returns:
        Delta values for each strike
    """
    S0 = float(params["S0"])
    r = float(params["r"])
    sigma = float(params["sigma"])
    T = float(params["T"])

    K = np.asarray(K, dtype=float)

    # Edge case: at expiry, Delta is 1 if ITM, 0 otherwise
    if T <= 0:
        return (S0 > K).astype(float)

    # d1 from Black-Scholes formula
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Delta = N(d1) for call options
    return norm.cdf(d1)


def fft_delta_bs(
    params: dict[str, float],
    alpha: float = 1.25,
    N: int = 2**12,
    eta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delta of a European call via Carr–Madan FFT in the Black–Scholes model.
    
    Computes Delta numerically using Fourier transform methods. Returns Delta
    on a grid of strikes, which can be interpolated to desired values.
    
    Required params:
        S0 (float): Initial stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        
    Args:
        params (dict): Model parameters
        alpha (float): Damping parameter for Carr-Madan formula
        N (int): Number of FFT points (power of 2)
        eta (float): Grid spacing in frequency domain
        
    Returns:
        Tuple of (strike grid, Delta values)
    """
    # Frequency grid
    j = np.arange(N)
    v = j * eta

    # Evaluate Delta integrand at each frequency
    psi = delta_integrand(v, params, alpha)

    # Log-strike grid spacing and offset
    lambd = 2.0 * np.pi / (N * eta)
    b = 0.5 * N * lambd

    # Build strike grid
    m = np.arange(N)
    k = -b + m * lambd
    K = np.exp(k)

    # Trapezoidal rule weights
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    w *= eta

    # FFT computation
    fft_input = np.exp(1j * b * v) * psi * w
    fft_output = np.fft.fft(fft_input)

    # Undo damping and normalize
    Delta_vals = np.exp(-alpha * k) * fft_output.real / np.pi
    
    return K, Delta_vals


if __name__ == "__main__":
    # Load parameters from config file 
    with open("../config.json", "r") as f:
        config = json.load(f)
    
    params = {
        "S0": config["S0"],
        "r": config["r"],
        "sigma": config["sigma"],
        "T": config["T"],
    }
    
    alpha = config["alpha"]
    eta = config["eta"]
    N = config["N"]

    # ITM, ATM, OTM strikes
    K_test = np.array([80.0, 100.0, 120.0])

    # FFT Delta: compute on grid, then interpolate
    K_grid, delta_grid = fft_delta_bs(params, alpha=alpha, N=N, eta=eta)
    delta_fft = np.interp(np.log(K_test), np.log(K_grid), delta_grid)

    # Analytic Delta
    delta_bs = bs_call_delta(params, K_test)

    # Compare results
    print("K      Delta_FFT      Delta_BS      diff")
    print("-------------------------------------------")
    for K, d_fft, d_bs in zip(K_test, delta_fft, delta_bs):
        diff = d_fft - d_bs
        print(f"{K:6.1f}  {d_fft:11.6f}  {d_bs:11.6f}  {diff: .3e}")
