# Repository Structure

```
FourierOptionPricing/
│
├── fourier_options/              ← installable Python library
│   ├── __init__.py               ← public API: cf_*, fft_pricer, calibrate
│   ├── characteristic_functions.py  ← BS, Heston, Merton, VG (S0=1 normalized)
│   ├── fft_pricer.py             ← Carr-Madan FFT + Euler Greeks (Delta, Gamma)
│   └── calibration.py            ← ivol_vec (NR), calibrate (DE wrapper)
│
├── engine/                       ← C++ backend (pybind11)
│   ├── fft_pricer.cpp            ← FFT + Greek + IV routines in C++
│   └── setup.py                  ← build: uv run --with setuptools python setup.py build_ext --inplace
│
├── cpp_pricer.cp312-win_amd64.pyd  ← compiled C++ extension (auto-loaded by fourier_options)
│
├── generate_figs.py              ← paper figure generation (main entry point)
├── paper/figs/                   ← generated PDFs (git-ignored)
│   ├── real_smile.pdf
│   ├── vol_surface.pdf
│   ├── greeks_validation.pdf
│   └── timing_benchmark.pdf
│
├── tests/
│   ├── test_pricer.py
│   └── test_greeks.py
│
├── notebooks/                    ← exploratory Jupyter notebooks (kept on main)
│   ├── delta_comparison.ipynb
│   ├── fft_visualization.ipynb
│   ├── numerical_and_performance_validation.ipynb
│   └── sensitivity_study.ipynb
│
├── pyproject.toml                ← project metadata + deps (v1.0.0)
└── README.md
```

## Quick Start

```bash
# Install deps
uv sync

# Build C++ backend (first time only)
cd engine
uv run --with setuptools --with pybind11 python setup.py build_ext --inplace
cp cpp_pricer*.pyd ..
cd ..

# Generate all paper figures
uv run python generate_figs.py
```

## Library Usage

```python
from fourier_options import cf_heston, fft_pricer
from fourier_options.calibration import ivol_vec

params = dict(S0=100, r=0.05, q=0.013, T=1.0,
              kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)

K, prices, delta = fft_pricer(cf_heston, params, alpha=1.5)
```

## Architecture

| Layer | Tech | Purpose |
|-------|------|---------|
| Python API | `fourier_options/` | Clean public interface |
| C++ Engine | `engine/fft_pricer.cpp` | FFT, IV inversion, Greeks |
| Entry Point | `generate_figs.py` | Paper figure generation |

## Models Supported

| Model | CF | Key params |
|-------|----|------------|
| Black-Scholes | `cf_bs` | σ |
| Merton Jump | `cf_merton` | σ, λ, μⱼ, σⱼ |
| Variance Gamma | `cf_vg` | σ, ν, θ\_vg |
| Heston | `cf_heston` | κ, θ, σᵥ, ρ, v₀ |
