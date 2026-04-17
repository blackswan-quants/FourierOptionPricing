# Fourier Pricing of European Options

**BlackSwan Quants** — High-Performance Option Pricing Engine via Carr-Madan FFT

---

## Overview

This project implements a production-grade, hybrid C++/Python option pricing engine based on the **Carr-Madan Fast Fourier Transform (FFT)** framework. The engine prices European options by operating directly on the **characteristic function** of the log-price under the risk-neutral measure — no explicit probability density function required.

The architecture supports four stochastic models:

| Model | Key phenomenon | CF |
|-------|---------------|-----|
| Black-Scholes | Constant volatility | `cf_bs` |
| Merton Jump-Diffusion | Fat tails, overnight gaps | `cf_merton` |
| Variance Gamma | Infinite-activity jumps, skew | `cf_vg` |
| Heston | Stochastic volatility, leverage | `cf_heston` |

---

## Library Usage

```python
from fourier_options import cf_heston, fft_pricer
from fourier_options.calibration import ivol_vec

# Price a full European call surface
params = dict(S0=100, r=0.05, q=0.013, T=1.0,
              kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)

K, prices, delta = fft_pricer(cf_heston, params, alpha=1.5)
```

---

## Quick Start

```bash
# 1. Install Python dependencies
uv sync

# 2. Build the C++ backend (first time only)
cd engine
uv run --with setuptools --with pybind11 python setup.py build_ext --inplace
Move-Item cpp_pricer*.pyd ..
cd ..

# 3. Generate all paper figures (calibrates on live SPY data)
uv run python generate_figs.py
```

Figures are saved to `paper/figs/`.

---

## Repository Structure

```
FourierOptionPricing/
├── fourier_options/             ← installable Python library
│   ├── __init__.py              ← public API
│   ├── characteristic_functions.py
│   ├── fft_pricer.py            ← Carr-Madan FFT + Euler Greeks
│   └── calibration.py           ← IV inversion + calibrate()
├── engine/                      ← C++ backend (pybind11)
│   ├── fft_pricer.cpp
│   └── setup.py
├── cpp_pricer.cp312-win_amd64.pyd  ← compiled extension
├── generate_figs.py             ← main figure generation script
├── paper/figs/                  ← generated PDFs
├── notebooks/                   ← exploratory Jupyter notebooks
├── tests/                       ← unit tests
├── STRUCTURE.md                 ← architecture reference
└── pyproject.toml
```

---

## Architecture

The engine follows a strict **separation of concerns**:

- **`fourier_options/`** — Python API. Characteristic functions implement $S_0=1$ log-return normalization to prevent high-frequency aliasing. The `fft_pricer()` function dispatches to the C++ backend when available, falling back to a pure NumPy implementation.
- **`engine/fft_pricer.cpp`** — Compiled C++ core with Radix-2 Cooley-Tukey FFT, Simpson's rule integration ($1$-$4$-$2$-$4$-$1$ weights), robust Newton-Raphson IV inversion with OTM switching, and adaptive dual-rate grids ($N=8192$, $\eta=0.5$ for short-dated maturities).
- **`generate_figs.py`** — End-to-end pipeline: fetches live SPY data, calibrates all 4 models via Differential Evolution, and produces the academic figure suite.

---

## References

- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies.
- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics.
- Fang, F. & Oosterlee, C. W. (2008). *A novel pricing method for European options based on Fourier-Cosine series expansions.* SIAM Journal on Scientific Computing.
- Lord, R. & Kahl, C. (2010). *Complex Logarithms in Heston-Like Models.* Mathematical Finance.
