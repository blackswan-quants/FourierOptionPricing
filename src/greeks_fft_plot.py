import numpy as np
import matplotlib.pyplot as plt

from greeks_fft import delta_fft_bs, gamma_fft_bs, vega_fft_bs

# --- Model parameters ---
params = {
    "S0": 100.0,
    "r": 0.01,
    "T": 1.0,
    "sigma": 0.20,
}

alpha = 1.5
N = 4096
eta = 0.25

# --- Compute Greeks via FFT  ---

# Delta(K)
K_delta, Delta = delta_fft_bs(
    params=params,
    alpha=alpha,
    N=N,
    eta=eta,
)

# Gamma(K)
K_gamma, Gamma = gamma_fft_bs(
    params=params,
    alpha=alpha,
    N=N,
    eta=eta,
)

# Vega(K)
K_vega, Vega = vega_fft_bs(
    params=params,
    alpha=alpha,
    N=N,
    eta=eta,
)

# sanity check grid alignment
assert np.allclose(K_delta, K_gamma)
assert np.allclose(K_delta, K_vega)

K = K_delta  # common strike grid

# Focus on a reasonable strike range around S0
mask = (K > 40) & (K < 160)
K_zoom = K[mask]
Delta_zoom = Delta[mask]
Gamma_zoom = Gamma[mask]
Vega_zoom  = Vega[mask]

# --- Plot Delta vs K (zoomed) ---
plt.figure()
plt.plot(K_zoom, Delta_zoom)
plt.xlabel("Strike K")
plt.ylabel("Delta (call)")
plt.title("Black–Scholes Call Delta vs Strike (FFT, zoomed)")
plt.grid(True)

# --- Plot Gamma vs K (zoomed) ---
plt.figure()
plt.plot(K_zoom, Gamma_zoom)
plt.xlabel("Strike K")
plt.ylabel("Gamma")
plt.title("Black–Scholes Gamma vs Strike (FFT, zoomed)")
plt.grid(True)

# --- Plot Vega vs K (zoomed) ---
plt.figure()
plt.plot(K_zoom, Vega_zoom)
plt.xlabel("Strike K")
plt.ylabel("Vega")
plt.title("Black–Scholes Vega vs Strike (FFT, zoomed)")
plt.grid(True)

plt.show()
