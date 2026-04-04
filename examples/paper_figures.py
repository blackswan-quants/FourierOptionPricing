"""
paper_figures.py
================
Generates all publication-quality figures for the Fourier Option Pricing paper.
Run from the project root:
    python examples/paper_figures.py
Figures are saved under paper/figures/.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs, cf_heston, cf_merton
from fourier_options.greeks.fft import delta_fft_bs, gamma_fft_bs, vega_fft_bs

# ── Output directory ─────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, kind="call", tol=1e-8):
    """Scalar implied vol via bisection."""
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = bs_price(S, K, T, r, mid, kind) - price
        if abs(val) < tol:
            return mid
        if val > 0:
            hi = mid
        else:
            lo = mid
    return np.nan


def ivol_curve(K_arr, price_arr, S, T, r, kind="call"):
    return np.array([implied_vol(p, S, k, T, r, kind) for k, p in zip(K_arr, price_arr)])


def fft_window(cf, params, alpha=1.5, N=4096, eta=0.25, lo=0.5, hi=2.0):
    """Return (K, prices) filtered to [lo*S0, hi*S0]."""
    S0 = params["S0"]
    K, V = fft_pricer(cf, params, alpha=alpha, N=N, eta=eta)
    mask = (K > S0 * lo) & (K < S0 * hi)
    return K[mask], V[mask]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – FFT vs Analytical BS: prices and pointwise error
# ─────────────────────────────────────────────────────────────────────────────
def fig1_convergence():
    params = {"S0": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}
    S0 = params["S0"]

    K_fft, C_fft = fft_window(cf_bs, params, alpha=1.5, N=4096)
    C_bs  = bs_price(S0, K_fft, params["T"], params["r"], params["sigma"])
    err   = np.abs(C_fft - C_bs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.5]})

    ax1.plot(K_fft, C_bs,  "k--", lw=1.4, label="Black–Scholes (exact)", zorder=3)
    ax1.plot(K_fft, C_fft, color="#2166ac", lw=1.8, label="Carr–Madan FFT  ($N=4096$)", zorder=2)
    ax1.set_ylabel("Call price ($)")
    ax1.legend()
    ax1.set_title("FFT vs. Analytical Black–Scholes Prices")

    ax2.semilogy(K_fft, np.maximum(err, 1e-16), color="#d6604d", lw=1.4)
    ax2.set_ylabel("Absolute error")
    ax2.set_xlabel("Strike $K$")
    ax2.set_title("Pointwise Absolute Error")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_fft_vs_bs.pdf")
    fig.savefig(f"{OUT}/fig1_fft_vs_bs.png")
    print("  fig1_fft_vs_bs  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Convergence in N: RMSE as a function of grid size
# ─────────────────────────────────────────────────────────────────────────────
def fig2_convergence_in_N():
    params = {"S0": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}
    S0 = params["S0"]
    Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    rmse_vals = []

    for N in Ns:
        K, C = fft_window(cf_bs, params, alpha=1.5, N=N)
        ref  = bs_price(S0, K, params["T"], params["r"], params["sigma"])
        rmse_vals.append(np.sqrt(np.mean((C - ref)**2)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(Ns, rmse_vals, "o-", color="#1a9641", lw=1.6, ms=5)
    # Reference slope -1 line
    ax.loglog(Ns, rmse_vals[0] * (np.array(Ns) / Ns[0])**(-1.0),
              "k--", lw=0.8, label="$O(N^{-1})$")
    ax.set_xlabel("FFT grid size $N$")
    ax.set_ylabel("RMSE vs. exact BS")
    ax.set_title("FFT Pricing Error vs. Grid Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_convergence_N.pdf")
    fig.savefig(f"{OUT}/fig2_convergence_N.png")
    print("  fig2_convergence_N  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Alpha sensitivity: RMSE vs damping factor
# ─────────────────────────────────────────────────────────────────────────────
def fig3_alpha_sensitivity():
    params = {"S0": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}
    S0 = params["S0"]
    alphas = np.linspace(0.1, 5.0, 80)
    rmse_vals = []

    # Reference grid (fixed N=4096)
    K_ref, _ = fft_window(cf_bs, params, alpha=1.5, N=4096)
    ref = bs_price(S0, K_ref, params["T"], params["r"], params["sigma"])

    for a in alphas:
        try:
            K, C = fft_pricer(cf_bs, params, alpha=a, N=4096, eta=0.25)
            S0_ = params["S0"]
            mask = (K > S0_ * 0.5) & (K < S0_ * 2.0)
            # Interpolate to same grid
            C_interp = np.interp(np.log(K_ref), np.log(K[mask]), C[mask])
            rmse_vals.append(np.sqrt(np.mean((C_interp - ref)**2)))
        except Exception:
            rmse_vals.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(alphas, rmse_vals, color="#762a83", lw=1.6)
    ax.axvline(1.5, color="gray", lw=0.8, ls="--", label=r"$\alpha=1.5$ (default)")
    ax.set_xlabel(r"Damping factor $\alpha$")
    ax.set_ylabel("RMSE vs. exact BS")
    ax.set_title(r"Sensitivity of FFT Error to Damping Factor $\alpha$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_alpha_sensitivity.pdf")
    fig.savefig(f"{OUT}/fig3_alpha_sensitivity.png")
    print("  fig3_alpha_sensitivity  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Implied Volatility Smile: BS vs Merton vs Heston
# ─────────────────────────────────────────────────────────────────────────────
def fig4_smile_comparison():
    S0, r, T = 100.0, 0.05, 1.0

    bs_params = {"S0": S0, "r": r, "T": T, "sigma": 0.20}

    merton_params = {"S0": S0, "r": r, "T": T, "sigma": 0.15,
                     "lam": 0.5, "mu_j": -0.10, "sig_j": 0.15}

    heston_params = {"S0": S0, "r": r, "T": T,
                     "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30,
                     "rho": -0.70, "v0": 0.04}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (cf, p, label, color, ls) in [
        (cf_bs,     bs_params,     "Black–Scholes",          "#2166ac", "-"),
        (cf_merton, merton_params, "Merton jump-diffusion",  "#d6604d", "--"),
        (cf_heston, heston_params, "Heston stoch. vol.",     "#1a9641", "-."),
    ]:
        K, C = fft_window(cf, p, alpha=1.5, N=4096)
        iv = ivol_curve(K, C, S0, T, r)
        # Keep only strike range with valid IVs
        valid = np.isfinite(iv) & (iv > 0)
        moneyness = K[valid] / S0
        ax.plot(moneyness, iv[valid] * 100, color=color, lw=1.8, ls=ls, label=label)

    ax.axvline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Moneyness  $K/S_0$")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Implied Volatility Smile: Model Comparison  ($T=1$ yr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_smile_comparison.pdf")
    fig.savefig(f"{OUT}/fig4_smile_comparison.png")
    print("  fig4_smile_comparison  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Heston parameter sensitivity on the smile
# ─────────────────────────────────────────────────────────────────────────────
def fig5_heston_sensitivity():
    S0, r, T = 100.0, 0.05, 1.0
    base = {"S0": S0, "r": r, "T": T,
            "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30,
            "rho": -0.70, "v0": 0.04}

    configs = [
        # (param, values, label_fmt, ax_title)
        ("rho",     [-0.9, -0.5, 0.0,  0.5],  r"$\rho={:.1f}$",   r"(a) Correlation $\rho$"),
        ("sigma_v", [0.10, 0.25, 0.50, 0.80],  r"$\xi={:.2f}$",    r"(b) Vol-of-vol $\xi$"),
        ("kappa",   [0.5,  1.0,  3.0,  6.0],   r"$\kappa={:.1f}$", r"(c) Mean-reversion $\kappa$"),
        ("v0",      [0.01, 0.04, 0.09, 0.16],  r"$v_0={:.2f}$",    r"(d) Initial variance $v_0$"),
    ]

    colors = ["#1a9641", "#2166ac", "#d6604d", "#762a83"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

    for ax, (param, values, fmt, title) in zip(axes.flat, configs):
        for val, col in zip(values, colors):
            p = {**base, param: val}
            try:
                K, C = fft_window(cf_heston, p, alpha=1.5, N=4096)
                iv   = ivol_curve(K, C, S0, T, r)
                valid = np.isfinite(iv) & (iv > 0)
                ax.plot(K[valid] / S0, iv[valid] * 100,
                        color=col, lw=1.6, label=fmt.format(val))
            except Exception:
                pass
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_title(title)
        ax.set_xlabel("$K/S_0$")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Heston Model — Parameter Sensitivity on Implied Volatility Smile", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_heston_sensitivity.pdf")
    fig.savefig(f"{OUT}/fig5_heston_sensitivity.png")
    print("  fig5_heston_sensitivity  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Heston Term Structure: smile across maturities
# ─────────────────────────────────────────────────────────────────────────────
def fig6_heston_term_structure():
    S0, r = 100.0, 0.05
    maturities = [0.25, 0.5, 1.0, 2.0]
    colors = ["#d6604d", "#f4a582", "#4393c3", "#2166ac"]

    heston_params = {"S0": S0, "r": r, "T": 1.0,
                     "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30,
                     "rho": -0.70, "v0": 0.04}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for T, col in zip(maturities, colors):
        p = {**heston_params, "T": T}
        K, C = fft_window(cf_heston, p, alpha=1.5, N=4096)
        iv = ivol_curve(K, C, S0, T, r)
        valid = np.isfinite(iv) & (iv > 0)
        ax.plot(K[valid] / S0, iv[valid] * 100, color=col, lw=1.8,
                label=f"$T={T}$ yr")

    ax.axvline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Moneyness  $K/S_0$")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Heston Model — Implied Volatility Term Structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_heston_term_structure.pdf")
    fig.savefig(f"{OUT}/fig6_heston_term_structure.png")
    print("  fig6_heston_term_structure  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – Greeks via FFT vs Analytical BS
# ─────────────────────────────────────────────────────────────────────────────
def fig7_greeks():
    S0, r, T, sigma = 100.0, 0.05, 1.0, 0.20
    params = {"S0": S0, "r": r, "T": T, "sigma": sigma}

    K_d, delta_fft = delta_fft_bs(params, alpha=1.5, N=4096, eta=0.25)
    K_g, gamma_fft = gamma_fft_bs(params, alpha=1.5, N=4096, eta=0.25)
    K_v, vega_fft  = vega_fft_bs(params, alpha=1.5, N=4096, eta=0.25)

    # Analytical BS Greeks
    def bs_delta(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    def bs_gamma(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def bs_vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    lo, hi = S0 * 0.5, S0 * 2.0
    mask_d = (K_d > lo) & (K_d < hi)
    mask_g = (K_g > lo) & (K_g < hi)
    mask_v = (K_v > lo) & (K_v < hi)

    K_bench = np.linspace(lo, hi, 400)
    d_bench = bs_delta(S0, K_bench, T, r, sigma)
    g_bench = bs_gamma(S0, K_bench, T, r, sigma)
    v_bench = bs_vega (S0, K_bench, T, r, sigma)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    data = [
        (axes[0], K_d[mask_d], delta_fft[mask_d], K_bench, d_bench, "Delta $\Delta$"),
        (axes[1], K_g[mask_g], gamma_fft[mask_g], K_bench, g_bench, "Gamma $\Gamma$"),
        (axes[2], K_v[mask_v], vega_fft[mask_v],  K_bench, v_bench, "Vega $\mathcal{V}$"),
    ]
    for ax, K_f, val_f, K_b, val_b, title in data:
        ax.plot(K_b, val_b, "k--", lw=1.2, label="Analytical")
        ax.plot(K_f, val_f, color="#2166ac", lw=1.8, label="FFT")
        ax.axvline(S0, color="gray", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Strike $K$")
        ax.legend()

    fig.suptitle("Option Greeks via Carr–Madan FFT vs. Analytical BS  ($T=1$ yr, $\sigma=20\%$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_greeks.pdf")
    fig.savefig(f"{OUT}/fig7_greeks.png")
    print("  fig7_greeks  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 – BS Calibration: loss landscape + convergence
# ─────────────────────────────────────────────────────────────────────────────
def fig8_bs_calibration():
    S0, r, T = 100.0, 0.05, 1.0
    true_sigma = 0.25

    # Synthetic market data
    true_params = {"S0": S0, "r": r, "T": T, "sigma": true_sigma}
    K_mkt, C_mkt = fft_window(cf_bs, true_params, alpha=1.5, N=4096)

    # Loss landscape
    sigmas = np.linspace(0.05, 0.60, 120)
    losses = []
    for s in sigmas:
        K_m, C_m = fft_pricer(cf_bs, {"S0": S0, "r": r, "T": T, "sigma": s},
                               alpha=1.5, N=4096, eta=0.25)
        # Interpolate to market grid
        C_interp = np.interp(np.log(K_mkt), np.log(K_m), C_m)
        losses.append(np.mean((C_interp - C_mkt)**2))

    # Optimization trace
    trace = []
    def loss_with_trace(x):
        s = float(x[0])
        K_m, C_m = fft_pricer(cf_bs, {"S0": S0, "r": r, "T": T, "sigma": s},
                               alpha=1.5, N=4096, eta=0.25)
        C_interp = np.interp(np.log(K_mkt), np.log(K_m), C_m)
        val = float(np.mean((C_interp - C_mkt)**2))
        trace.append((s, val))
        return val

    result = minimize(loss_with_trace, x0=[0.10], bounds=[(0.01, 0.99)],
                      method="L-BFGS-B")
    sigma_cal = result.x[0]
    trace = np.array(trace)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss landscape
    ax1.semilogy(sigmas, losses, color="#2166ac", lw=1.6)
    ax1.axvline(true_sigma, color="#1a9641", ls="--", lw=1.2,
                label=f"True $\sigma={true_sigma:.2f}$")
    ax1.axvline(sigma_cal, color="#d6604d", ls=":",  lw=1.2,
                label=f"Calibrated $\sigma={sigma_cal:.4f}$")
    ax1.set_xlabel(r"Volatility $\sigma$")
    ax1.set_ylabel("MSE (log scale)")
    ax1.set_title("BS Calibration — Loss Landscape")
    ax1.legend()

    # Optimization trace
    ax2.semilogy(np.arange(len(trace)), trace[:, 1], "o-", ms=3,
                 color="#762a83", lw=1.4)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_title(f"Calibration Convergence  (final $\sigma={sigma_cal:.4f}$)")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_bs_calibration.pdf")
    fig.savefig(f"{OUT}/fig8_bs_calibration.png")
    print(f"  fig8_bs_calibration  saved  (calibrated σ = {sigma_cal:.5f}, true = {true_sigma})")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 – Heston Calibration: fitted smile vs synthetic market
# ─────────────────────────────────────────────────────────────────────────────
def fig9_heston_calibration():
    S0, r = 100.0, 0.05
    true_p = {"S0": S0, "r": r, "T": 1.0,
              "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30,
              "rho": -0.70, "v0": 0.04}

    # Synthetic market: 15 strikes × 2 maturities
    Ks     = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130])
    Ts     = np.array([0.5, 1.0])
    strikes    = np.tile(Ks, len(Ts))
    maturities = np.repeat(Ts, len(Ks))

    market_prices = []
    for K_i, T_i in zip(strikes, maturities):
        p = {**true_p, "T": T_i}
        K_g, C_g = fft_pricer(cf_heston, p, alpha=1.5, N=4096, eta=0.25)
        market_prices.append(np.interp(np.log(K_i), np.log(K_g), C_g))
    market_prices = np.array(market_prices)
    bid_ask       = 0.05 * market_prices + 0.01  # synthetic spread

    # Calibration with Differential Evolution (tight bounds around truth)
    bounds = [(0.5, 5.0), (0.01, 0.20), (0.05, 0.80), (-0.99, 0.0), (0.01, 0.20)]

    def loss(theta_arr):
        kappa, theta, sigma_v, rho, v0 = theta_arr
        feller = 2 * kappa * theta - sigma_v**2
        pen = 0.0 if feller > 0 else 1e6 * abs(feller) + 1000
        weights = 1.0 / (bid_ask + 1e-5); weights /= weights.sum()
        model_p = []
        for K_i, T_i in zip(strikes, maturities):
            p = {"S0": S0, "r": r, "T": T_i,
                 "kappa": kappa, "theta": theta, "sigma_v": sigma_v,
                 "rho": rho, "v0": v0}
            K_g, C_g = fft_pricer(cf_heston, p, alpha=1.5, N=4096, eta=0.25)
            model_p.append(np.interp(np.log(K_i), np.log(K_g), C_g))
        model_p = np.array(model_p)
        return float(np.sum(weights * (model_p - market_prices)**2)) + pen

    print("  Running Heston DE calibration (this takes ~30 s)…")
    result = differential_evolution(loss, bounds, maxiter=60, popsize=10,
                                    tol=1e-5, seed=42, workers=1)
    kappa_c, theta_c, sigma_v_c, rho_c, v0_c = result.x
    print(f"  Calibrated: κ={kappa_c:.3f} θ={theta_c:.4f} ξ={sigma_v_c:.3f} "
          f"ρ={rho_c:.3f} v0={v0_c:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, T_i in zip(axes, Ts):
        mask = maturities == T_i
        K_m  = strikes[mask]
        iv_mkt = ivol_curve(K_m, market_prices[mask], S0, T_i, r)

        p_cal  = {"S0": S0, "r": r, "T": T_i,
                  "kappa": kappa_c, "theta": theta_c, "sigma_v": sigma_v_c,
                  "rho": rho_c, "v0": v0_c}
        K_c, C_c = fft_window(cf_heston, p_cal, alpha=1.5, N=4096)
        iv_cal   = ivol_curve(K_c, C_c, S0, T_i, r)
        valid    = np.isfinite(iv_cal) & (iv_cal > 0)

        ax.plot(K_c[valid] / S0, iv_cal[valid] * 100,
                color="#2166ac", lw=1.8, label="Calibrated Heston")
        ax.scatter(K_m / S0, iv_mkt * 100,
                   color="#d6604d", zorder=5, s=40, label="Synthetic market")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"$T = {T_i}$ yr")
        ax.set_xlabel("$K/S_0$")
        ax.set_ylabel("IV (%)")
        ax.legend()

    fig.suptitle("Heston Calibration — Fitted Smile vs. Synthetic Market Data")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_heston_calibration.pdf")
    fig.savefig(f"{OUT}/fig9_heston_calibration.png")
    print("  fig9_heston_calibration  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 – Computational benchmark: FFT vs Analytical vs Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def fig10_benchmark():
    import time
    params = {"S0": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}
    S0, r, T, sigma = params["S0"], params["r"], params["T"], params["sigma"]

    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
    times_fft  = []
    times_anal = []

    for N in Ns:
        K_grid = np.exp(-np.pi / 0.25 + np.arange(N) * (2 * np.pi / (N * 0.25)))

        # FFT
        reps = max(3, int(2000 / N))
        t0 = time.perf_counter()
        for _ in range(reps):
            fft_pricer(cf_bs, params, alpha=1.5, N=N, eta=0.25)
        times_fft.append((time.perf_counter() - t0) / reps * 1000)

        # Analytical (same N strikes)
        t0 = time.perf_counter()
        for _ in range(reps):
            bs_price(S0, K_grid, T, r, sigma)
        times_anal.append((time.perf_counter() - t0) / reps * 1000)

    # Monte Carlo for a single ATM strike
    mc_sims = [1_000, 5_000, 10_000, 50_000, 100_000]
    mc_times = []
    mc_errs  = []
    ref = bs_price(S0, S0, T, r, sigma)
    rng = np.random.default_rng(0)
    for ns in mc_sims:
        t0 = time.perf_counter()
        z  = rng.standard_normal(ns)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        mc_p = np.exp(-r * T) * np.mean(np.maximum(ST - S0, 0))
        mc_times.append((time.perf_counter() - t0) * 1000)
        mc_errs.append(abs(mc_p - ref))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(Ns, times_fft,  "o-", color="#2166ac", lw=1.6, ms=5, label="Carr–Madan FFT")
    ax1.loglog(Ns, times_anal, "s-", color="#1a9641", lw=1.6, ms=5, label="Analytical BS")
    ax1.set_xlabel("Number of strikes $N$")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("FFT vs. Analytical BS — Pricing Speed")
    ax1.legend()

    ax2.loglog(mc_sims, mc_errs,  "o-", color="#d6604d", lw=1.6, ms=5, label="MC error")
    ax2_r = ax2.twinx()
    ax2_r.loglog(mc_sims, mc_times, "^--", color="#762a83", lw=1.2, ms=5,
                 label="MC time (ms)")
    ax2.set_xlabel("MC simulations $N_{\\mathrm{sim}}$")
    ax2.set_ylabel("Absolute error ($)")
    ax2_r.set_ylabel("Wall-clock time (ms)", color="#762a83")
    ax2.set_title("Monte Carlo (ATM, single strike) — Error vs. Cost")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_benchmark.pdf")
    fig.savefig(f"{OUT}/fig10_benchmark.png")
    print("  fig10_benchmark  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 – Risk-neutral density: BS vs Merton vs Heston
# ─────────────────────────────────────────────────────────────────────────────
def fig11_risk_neutral_density():
    """
    Extract the risk-neutral density q(S_T) from call prices via
    the Breeden-Litzenberger relation:  q(K) = e^{rT} * d²C/dK²

    On the FFT log-spaced grid the correct discretisation is the
    weighted butterfly spread (same formula used in check_convexity):
        q(K_b) ≈ e^{rT} * [w_a C_a + w_c C_c – C_b] / A
    where  w_a = (K_c–K_b)/(K_c–K_a),  w_c = (K_b–K_a)/(K_c–K_a),
    and A  = ½(K_c–K_a)(K_b–K_a)(K_c–K_b)/(K_c–K_a)²  is the area
    normalisation that turns the butterfly value into a density.
    """
    S0, r, T = 100.0, 0.05, 1.0
    disc = np.exp(r * T)

    models = [
        (cf_bs,
         {"S0": S0, "r": r, "T": T, "sigma": 0.20},
         "Black–Scholes  ($\\sigma=20\\%$)",       "#2166ac", "-"),
        (cf_merton,
         {"S0": S0, "r": r, "T": T, "sigma": 0.15,
          "lam": 0.5, "mu_j": -0.10, "sig_j": 0.15},
         "Merton  ($\\lambda=0.5,\\,\\mu_j=-10\\%$)", "#d6604d", "--"),
        (cf_heston,
         {"S0": S0, "r": r, "T": T,
          "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30,
          "rho": -0.70, "v0": 0.04},
         "Heston  ($\\rho=-0.7,\\,\\xi=0.30$)",     "#1a9641", "-."),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for cf, params, label, color, ls in models:
        K, C = fft_pricer(cf, params, alpha=1.5, N=8192, eta=0.10)
        mask = (K > S0 * 0.4) & (K < S0 * 2.0)
        K, C = K[mask], C[mask]

        Ka, Kb, Kc = K[:-2], K[1:-1], K[2:]
        Ca, Cb, Cc = C[:-2], C[1:-1], C[2:]
        wa = (Kc - Kb) / (Kc - Ka)
        wc = (Kb - Ka) / (Kc - Ka)

        # Breeden-Litzenberger: q(K_b) = e^{rT} * butterfly / (area)
        butterfly = wa * Ca + wc * Cc - Cb
        # Area factor converts butterfly price → density (units: 1/$)
        area = 0.5 * (Kc - Ka) * (Kb - Ka) * (Kc - Kb) / (Kc - Ka) ** 2
        q = disc * butterfly / area
        q = np.maximum(q, 0)  # numerical noise can give tiny negatives

        ax1.plot(Kb, q, color=color, lw=1.8, ls=ls, label=label)
        ax2.plot(Kb / S0, q * Kb, color=color, lw=1.8, ls=ls, label=label)

    for ax in (ax1, ax2):
        ax.axvline(S0 if ax is ax1 else 1.0, color="gray", lw=0.6, ls=":")
        ax.legend(fontsize=9)
        ax.set_ylabel("Density")

    ax1.set_xlabel("Terminal stock price  $S_T$")
    ax1.set_title("Risk-neutral density  $q(S_T)$")
    ax2.set_xlabel("Moneyness  $S_T / S_0$")
    ax2.set_title("Scaled density  $S_T \\cdot q(S_T)$")

    fig.suptitle("Breeden–Litzenberger Risk-neutral Density  ($T=1$ yr)", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig11_risk_neutral_density.pdf")
    fig.savefig(f"{OUT}/fig11_risk_neutral_density.png")
    print("  fig11_risk_neutral_density  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12 – Merton jump parameter sensitivity on the smile
# ─────────────────────────────────────────────────────────────────────────────
def fig12_merton_sensitivity():
    S0, r, T = 100.0, 0.05, 1.0
    base = {"S0": S0, "r": r, "T": T,
            "sigma": 0.15, "lam": 0.5, "mu_j": -0.10, "sig_j": 0.15}

    configs = [
        ("lam",   [0.1, 0.5, 1.0, 2.0],   r"$\lambda={:.1f}$",    r"(a) Jump intensity $\lambda$"),
        ("mu_j",  [-0.20, -0.10, 0.0, 0.10], r"$\mu_j={:.2f}$",   r"(b) Mean log-jump $\mu_j$"),
        ("sig_j", [0.05, 0.10, 0.20, 0.40],  r"$\delta={:.2f}$",  r"(c) Jump std $\delta$"),
        ("sigma", [0.05, 0.10, 0.20, 0.30],  r"$\sigma={:.2f}$",  r"(d) Diffusion vol $\sigma$"),
    ]

    colors = ["#1a9641", "#2166ac", "#d6604d", "#762a83"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

    for ax, (param, values, fmt, title) in zip(axes.flat, configs):
        for val, col in zip(values, colors):
            p = {**base, param: val}
            try:
                K, C = fft_pricer(cf_merton, p, alpha=1.5, N=4096, eta=0.25)
                mask = (K > S0 * 0.6) & (K < S0 * 1.6)
                K_w, C_w = K[mask], C[mask]
                iv = ivol_curve(K_w, C_w, S0, T, r)
                valid = np.isfinite(iv) & (iv > 0)
                ax.plot(K_w[valid] / S0, iv[valid] * 100,
                        color=col, lw=1.6, label=fmt.format(val))
            except Exception:
                pass
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_title(title)
        ax.set_xlabel("$K/S_0$")
        ax.set_ylabel("IV (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Merton Jump-Diffusion — Parameter Sensitivity on Implied Volatility Smile",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig12_merton_sensitivity.pdf")
    fig.savefig(f"{OUT}/fig12_merton_sensitivity.png")
    print("  fig12_merton_sensitivity  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13 – Carr-Madan integrand visualisation
# Shows the complex integrand ψ(v) and how α shapes its decay,
# explaining why a good damping factor is essential for FFT accuracy.
# ─────────────────────────────────────────────────────────────────────────────
def fig13_carr_madan_integrand():
    params = {"S0": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}
    r, T = params["r"], params["T"]
    alphas = [0.25, 1.0, 1.5, 3.0]
    colors = ["#d6604d", "#f4a582", "#2166ac", "#762a83"]

    eta = 0.01          # fine grid to show the function shape
    N   = 2048
    v   = np.arange(N) * eta

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for alpha, col in zip(alphas, colors):
        u = v - 1j * (alpha + 1.0)
        from fourier_options.domain.characteristic_functions import cf_bs as _cf_bs
        phi = _cf_bs(u, params)
        denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        psi = np.exp(-r * T) * phi / denom

        axes[0].plot(v, np.abs(psi), color=col, lw=1.5,
                     label=rf"$\alpha={alpha}$")
        axes[1].semilogy(v, np.abs(psi) + 1e-20, color=col, lw=1.5,
                         label=rf"$\alpha={alpha}$")

    for ax in axes:
        ax.set_xlabel(r"Integration variable $v$")
        ax.set_ylabel(r"$|\psi(v)|$")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 15)
        ax.axvline(0, color="gray", lw=0.5)

    axes[0].set_title("Carr–Madan integrand  $|\\psi(v)|$  (linear)")
    axes[1].set_title("Carr–Madan integrand  $|\\psi(v)|$  (log scale)")

    fig.suptitle(
        "Effect of Damping Factor $\\alpha$ on the Carr–Madan Integrand  "
        "($N=2048$, $\\eta=0.01$)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig13_carr_madan_integrand.pdf")
    fig.savefig(f"{OUT}/fig13_carr_madan_integrand.png")
    print("  fig13_carr_madan_integrand  saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving figures to {OUT}/\n")

    fig1_convergence()
    fig2_convergence_in_N()
    fig3_alpha_sensitivity()
    fig4_smile_comparison()
    fig5_heston_sensitivity()
    fig6_heston_term_structure()
    fig7_greeks()
    fig8_bs_calibration()
    fig9_heston_calibration()
    fig10_benchmark()
    fig11_risk_neutral_density()
    fig12_merton_sensitivity()
    fig13_carr_madan_integrand()

    print(f"\nAll figures saved to {OUT}/")
