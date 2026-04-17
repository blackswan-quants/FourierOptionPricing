# Fourier Pricing of European Options

**BlackSwan Quants** — A Low-Latency FFT Pricing Architecture with Analytical Greeks in C++

> *Edoardo Mocchi, Lorenzo Pirozzi, Pietro Bottan, Simone Copetti, Achille Galante, Simone Lorello, David Piana*
> BlackSwan Quants, Milan, Italy

---

## Overview

This project implements a production-grade, hybrid **C++/Python** option pricing engine based on the **Carr-Madan Fast Fourier Transform (FFT)** framework. The engine prices entire European option surfaces by operating directly on the **characteristic function** of the log-price under the risk-neutral measure — no explicit probability density function required.

The key insight of the Carr-Madan approach is that, while advanced stochastic models (Heston, Merton, VG) lack closed-form densities, they all possess analytically tractable **characteristic functions**. By shifting the valuation problem to the frequency domain and leveraging the FFT algorithm, the engine computes an entire option surface across thousands of strikes in **O(N log N)** time.

To eliminate the computational bottleneck of Python's interpreted loops during intensive calibration procedures (thousands of objective function evaluations), the core FFT routines, characteristic functions, and IV inversion are implemented natively in **C++** and exposed to Python via **Pybind11**, bypassing the GIL and achieving order-of-magnitude speedups.

---

## Highlights

- **O(N log N) surface generation** — FFT simultaneously prices N strikes with a single DFT execution
- **Analytical Greeks** — Delta, Gamma, and Vega computed directly in the frequency domain via characteristic function differentiation, no finite-difference bumping
- **Hybrid C++/Python architecture** — performance-critical loops compiled via Pybind11; clean Python API for research and calibration
- **IV-space calibration** — loss function defined in implied volatility space (not price space), ensuring symmetric gradient dynamics across all moneyness regimes
- **4 stochastic models** — Black-Scholes, Heston, Merton Jump-Diffusion, Variance Gamma, all plug into the same FFT integrator
- **Live market calibration** — Differential Evolution optimizer on real SPY option data via `yfinance`

---

## Models

| Model | Key Phenomenon | Params | Characteristic Function |
|-------|---------------|--------|------------------------|
| **Black-Scholes** | Constant volatility, GBM | σ | `cf_bs` |
| **Merton Jump-Diffusion** | Fat tails, overnight gaps | σ, λ, μⱼ, σⱼ | `cf_merton` |
| **Variance Gamma** | Infinite-activity jumps, skew asymmetry | σ, ν, θ_vg | `cf_vg` |
| **Heston** | Stochastic volatility, leverage effect | κ, θ, σᵥ, ρ, v₀ | `cf_heston` |

The FFT integrator is **completely agnostic** to the underlying dynamics. Adding a new model only requires implementing its characteristic function — the pricing, Greek, and calibration machinery are inherited automatically.

---

## Quick Start

```bash
# 1. Clone and install (compiles the C++ backend automatically)
git clone <repo>
cd FourierOptionPricing
uv pip install -e .

# 2. Generate all paper figures (calibrates on live SPY data)
uv run python generate_figs.py
```

Figures are saved to `paper/figs/` as PDFs.

> **Requirements:** Python ≥ 3.12, CMake ≥ 3.15, a C++14-capable compiler (MSVC on Windows, GCC/Clang on Linux/macOS).

---

## Library Usage

```python
from fourier_options import cf_heston, fft_pricer
from fourier_options.calibration import calibrate, ivol_vec

# --- Pricing a full European call surface ---
params = dict(
    S0=100, r=0.05, q=0.013, T=1.0,
    kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04
)
K, prices, delta = fft_pricer(cf_heston, params, alpha=1.5)

# --- Calibrate Heston to market implied volatilities ---
result = calibrate("heston", S0, r, q, T, market_strikes, market_ivols)
print(result.x)  # [kappa, theta, sigma_v, rho, v0]
```

---

## Architecture

The engine is structured around a strict **separation of concerns**:

```
FourierOptionPricing/
├── fourier_options/              ← installable Python library (public API)
│   ├── __init__.py
│   ├── characteristic_functions.py  ← thin Python wrappers over C++ CFs
│   ├── fft_pricer.py             ← Carr-Madan FFT + Euler Greeks dispatch
│   └── calibration.py            ← ivol_vec (Newton-Raphson), calibrate (DE)
│
├── engine/                       ← C++ backend (pybind11 + CMake)
│   ├── fft_pricer.cpp            ← Radix-2 FFT, CFs, IV inversion, Greeks
│   └── CMakeLists.txt            ← compiled by scikit-build-core on `pip install`
│
├── generate_figs.py              ← paper figure generation pipeline
├── paper/figs/                   ← output PDFs (git-ignored)
├── notebooks/                    ← exploratory Jupyter notebooks
├── tests/                        ← unit & validation tests
└── pyproject.toml
```

### C++ Engine (`engine/fft_pricer.cpp`)

The compiled core contains:
- **Radix-2 Cooley-Tukey FFT** — in-place bit-reversal permutation, O(N log N)
- **Characteristic functions** — BS, Heston (Riccati ODE solution), MJD, VG; all evaluated under log-return normalization (S₀ = 1) to suppress aliasing from the phase term e^(iu ln S₀)
- **Simpson's rule integration** — fourth-order convergence with weights 1-4-2-4-1
- **Adaptive dual-rate grids** — N=8192, η=0.5 for short-dated (T < 0.1); N=4096, η=0.25 otherwise
- **Robust IV inversion** — Newton-Raphson with OTM/ITM switching to prevent Vega vanishing near deep-ITM strikes

### Python API (`fourier_options/`)

- `characteristic_functions.py`: exposes model CFs as callable objects that forward to the C++ backend, enforcing S₀=1 normalization
- `fft_pricer.py`: implements Carr-Madan inversion and dispatches to C++ when available, with a pure-NumPy fallback
- `calibration.py`: vector-mapped Newton-Raphson for IV inversion; Differential Evolution wrapper for global calibration

---

## Pricing Framework

For a European call with log-strike k = ln(K), the Carr-Madan formula introduces a damping factor α > 0 to enforce L² integrability:

```
ψ_T(v) = e^{-rT} φ_T(v − (α+1)i) / [α² + α − v² + i(2α+1)v]
```

where φ_T(u) is the characteristic function of the log-price. Call prices are recovered via:

```
C_T(k_m) ≈ (e^{−αk_m} / π) · Re[FFT(ψ_T)]_m
```

The simultaneous pricing of N strikes requires a **single FFT execution**, reducing complexity from O(M·N) (per-strike quadrature) to **O(N log N)**.

> **Damping constraint:** α must satisfy α + 1 < k_max(Θ, T), where k_max is the moment explosion threshold of the characteristic function. A static α can silently fail during calibration when the optimizer traverses parameter regions with restricted strip of analyticity (notably in Heston when σᵥ is large or ρ is very negative).

---

## Analytical Greeks via FFT

Rather than finite-difference bumping, Greeks are computed by **differentiating the characteristic function directly in the complex plane**. For any Greek G, the integrand is obtained via an O(N) algebraic modulation of the pricing vector ψ_T(v), followed by a single additional FFT inversion:

| Greek | Modulated Integrand | Cost |
|-------|---------------------|------|
| **Delta** (∂C/∂S₀) | `ψ_Δ(v) = [(α+1+iv) / S₀] · ψ_T(v)` | O(N) mod + O(N log N) FFT |
| **Gamma** (∂²C/∂S₀²) | `ψ_Γ(v) = [(iv+α+1)(iv+α) / S₀²] · ψ_T(v)` | O(N) mod + O(N log N) FFT |
| **Vega** (∂C/∂σ, BS) | `ψ_V(v) = σT · [(iv+α+1)(iv+α)] · ψ_T(v)` | O(N) mod + O(N log N) FFT |

Delta and Gamma are **model-independent** in the transformed space (they follow from the homogeneous structure of C w.r.t. S₀). Vega requires model-specific derivation but follows the same integration pattern.

Validation against Black-Scholes closed-form confirms errors of order **10⁻⁵–10⁻⁶** across the ATM region.

---

## Performance

The FFT approach achieves a **two-order-of-magnitude speedup** over Monte Carlo for surface generation:

| Method | Complexity | Strike grid N=4096 |
|--------|-----------|-------------------|
| FFT (this engine) | O(N log N) | ~1 ms |
| Monte Carlo (CV, 10k paths) | O(N) with large constant | ~100 ms |
| MC with precision-consistency | O(N³) | impractical |

The MC Control Variates baseline (Heston + Black-Scholes control, Euler-Maruyama) maintains an O(N) slope, but the path simulation overhead sustains a **~100× gap** in the full operationally relevant regime. When constant relative accuracy is required across a refined strike grid, MC complexity degrades to O(M³), making full surface generation entirely impractical.

---

## Calibration Results on Real SPY Data

Market data is sourced from live S&P 500 ETF (SPY) options via `yfinance`, filtered for liquidity (volume > 10) and moneyness (K/S₀ ∈ [0.94, 1.10]). Calibration minimizes MSE in **implied volatility space** (not price space) via Differential Evolution.

**Key finding:** At short maturities (T ≈ 0.013y), the Variance Gamma model outperforms Heston:

| Model | RMSE (IV space) | Params |
|-------|----------------|--------|
| Variance Gamma | **0.767** | 3 |
| Heston | 2.521 | 5 |
| Merton Jump-Diffusion | ~1.5 | 4 |
| Black-Scholes | ~4.0 (flat smile) | 1 |

This is structurally explained by the nature of the processes: at ultra-short maturities, Heston's continuous diffusion has insufficient time to build up variance, collapsing toward Black-Scholes. Variance Gamma, as an **infinite-activity pure jump process**, can instantaneously price the risk of market gaps regardless of time to expiry — making it structurally superior for short-dated index options where jump risk dominates.

---

## Numerical Error Analysis

The total FFT pricing error decomposes into two independent components:

**Truncation error** (replacing ∫₀^∞ with ∫₀^a): decays **exponentially** in N for models whose characteristic function decays exponentially in frequency (Heston, MJD, VG), provided α lies strictly within the strip of analyticity.

**Discretization error** (Simpson's rule on N points): O(η⁴ · max|ψ_T^(4)(v)|). Choosing small η reduces this error but, via the Nyquist relation ληη = 2π/N, coarsens the strike grid λ — the fundamental FFT trade-off. The natural resolution is the **Fractional FFT (FrFFT)**, which introduces an independent fractional scalar to decouple the two grids (planned for a future iteration).

---

## References

- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies.
- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics.
- Madan, D., Carr, P. & Chang, E. (1998). *The Variance Gamma process and option pricing.* European Finance Review.
- Cooley, J. W. & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series.* Mathematics of Computation.
- Fang, F. & Oosterlee, C. W. (2008). *A novel pricing method for European options based on Fourier-Cosine series expansions.* SIAM Journal on Scientific Computing.
- Lord, R. & Kahl, C. (2010). *Complex Logarithms in Heston-Like Models.* Mathematical Finance.
- Andersen, L. B. & Piterbarg, V. V. (2007). *Moment explosions in stochastic volatility models.* Finance and Stochastics.
- Storn, R. & Price, K. (1997). *Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces.* Journal of Global Optimization.
- Quarteroni, A., Sacco, R. & Saleri, F. (2014). *Numerical Mathematics.* Springer.

---

## About

Developed by **BlackSwan Quants (BSQ)** — a quantitative finance student organization at Politecnico di Milano focused on advanced research and algorithmic trading engineering.

[![LinkedIn](https://img.shields.io/badge/BlackSwan_Quants-LinkedIn-blue)](https://www.linkedin.com/company/blackswan-quants/)

---

*Full technical details, mathematical derivations, and empirical results are documented in* [`paper/A_Low_Latency_FFT_Pricing_Architecture_with_Analytical_Greeks_in_C++.pdf`](paper/A_Low_Latency_FFT_Pricing_Architecture_with_Analytical_Greeks_in_C__.pdf).
