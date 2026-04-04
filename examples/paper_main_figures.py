"""
paper_main_figures.py
=====================
Two publication-ready figures for the Fourier Option Pricing paper.

Figure A – Heston model calibration on a realistic IV surface
           (SPX-like parameters, Gaussian IV noise, realistic bid-ask spreads).

Figure B – Model comparison: Black-Scholes vs Merton vs Heston
           calibrated to the same synthetic market, with IV residuals.

Run from the project root:
    python examples/paper_main_figures.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs, cf_heston, cf_merton
from fourier_options.calibration.loss import heston_weighted_loss

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r, tol=1e-8):
    if price <= max(S - K * np.exp(-r * T), 0) + 1e-10 or price >= S:
        return np.nan
    lo, hi = 1e-5, 5.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        v   = bs_call(S, K, T, r, mid) - price
        if abs(v) < tol:
            return mid
        hi, lo = (mid, lo) if v > 0 else (hi, mid)
    return mid


def ivol_vec(K_arr, C_arr, S, T, r):
    return np.array([implied_vol(c, S, k, T, r) for k, c in zip(K_arr, C_arr)])


def fft_at(cf, params, strikes, alpha=1.5, N=4096, eta=0.25):
    K_g, C_g = fft_pricer(cf, params, alpha=alpha, N=N, eta=eta)
    return np.interp(np.log(strikes), np.log(K_g), C_g)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA GENERATION
# SPX-inspired: S0=100, r=5%, Heston with moderate vol-of-vol and leverage.
# Four maturities × 11 strikes. IV noise ≈ 0.3%, bid-ask widens for OTM.
# ─────────────────────────────────────────────────────────────────────────────
def build_market(seed=42):
    S0, r = 100.0, 0.05
    rng   = np.random.default_rng(seed)

    true_params = dict(S0=S0, r=r,
                       kappa=1.50, theta=0.040, sigma_v=0.40,
                       rho=-0.70,  v0=0.060)

    maturities   = np.array([1/12, 3/12, 6/12, 1.0])
    log_moneyness = np.linspace(-0.25, 0.20, 11)

    market = {}
    for T in maturities:
        K      = S0 * np.exp(log_moneyness)
        C_true = fft_at(cf_heston, {**true_params, "T": T}, K)
        iv_true = ivol_vec(K, C_true, S0, T, r)

        valid = np.isfinite(iv_true) & (iv_true > 0.02)
        K, iv_true, lm = K[valid], iv_true[valid], log_moneyness[valid]

        # Bid-ask half-spread: base 0.5% + 1.5% per unit |log-moneyness|
        half_spread = 0.005 + 0.015 * np.abs(lm)
        noise       = rng.normal(0, 0.003, size=len(K))

        iv_mkt  = iv_true  + noise
        C_mkt   = np.array([bs_call(S0, k, T, r, iv) for k, iv in zip(K, iv_mkt)])
        C_bid   = np.array([bs_call(S0, k, T, r, max(iv - h, 0.01))
                            for k, iv, h in zip(K, iv_mkt, half_spread)])
        C_ask   = np.array([bs_call(S0, k, T, r, iv + h)
                            for k, iv, h in zip(K, iv_mkt, half_spread)])

        market[T] = dict(K=K, lm=lm, iv_mkt=iv_mkt,
                         iv_bid=iv_mkt - half_spread,
                         iv_ask=iv_mkt + half_spread,
                         C_mkt=C_mkt,
                         spread=C_ask - C_bid)

    return S0, r, true_params, maturities, market


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATIONS
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_heston(S0, r, market, maturities):
    all_K = np.concatenate([market[T]["K"]      for T in maturities])
    all_T = np.concatenate([[T] * len(market[T]["K"]) for T in maturities])
    all_C = np.concatenate([market[T]["C_mkt"]  for T in maturities])
    all_s = np.concatenate([market[T]["spread"] for T in maturities])

    bounds = [(0.1, 8.0), (0.01, 0.40), (0.05, 1.50), (-0.99, 0.0), (0.01, 0.50)]
    args   = (S0, r, all_K, all_T, all_C, all_s, 1.5, 4096, 0.25)

    print("  Calibrating Heston (Differential Evolution)…")
    res = differential_evolution(heston_weighted_loss, bounds, args=args,
                                 maxiter=120, popsize=15, tol=1e-7,
                                 seed=0, workers=1, disp=False)
    kappa, theta, sigma_v, rho, v0 = res.x
    print(f"  κ={kappa:.3f}  θ={theta:.4f}  ξ={sigma_v:.3f}  "
          f"ρ={rho:.3f}  v₀={v0:.4f}  loss={res.fun:.2e}")
    return dict(S0=S0, r=r, kappa=kappa, theta=theta,
                sigma_v=sigma_v, rho=rho, v0=v0)


def calibrate_bs_per_maturity(S0, r, market, maturities):
    """One flat vol per maturity (ATM region minimisation)."""
    sigmas = {}
    for T in maturities:
        d = market[T]
        atm = np.abs(d["lm"]) < 0.10

        def loss(x):
            C_m = fft_at(cf_bs, dict(S0=S0, r=r, T=T, sigma=x[0]), d["K"][atm])
            return float(np.mean((C_m - d["C_mkt"][atm])**2))

        res = minimize(loss, [0.20], bounds=[(0.02, 0.80)], method="L-BFGS-B")
        sigmas[T] = res.x[0]
    return sigmas


def calibrate_merton(S0, r, market, maturities):
    """
    Joint calibration of Merton parameters across all maturities.
    Parameters: sigma, lam, mu_j, sig_j  (sigma fixed at BS ATM level).
    """
    all_K = np.concatenate([market[T]["K"]     for T in maturities])
    all_T = np.concatenate([[T]*len(market[T]["K"]) for T in maturities])
    all_C = np.concatenate([market[T]["C_mkt"] for T in maturities])

    def loss(x):
        sigma, lam, mu_j, sig_j = x
        total = 0.0
        for T in maturities:
            mask = (all_T == T)
            p    = dict(S0=S0, r=r, T=T,
                        sigma=sigma, lam=lam, mu_j=mu_j, sig_j=sig_j)
            C_m  = fft_at(cf_merton, p, all_K[mask])
            total += np.mean((C_m - all_C[mask])**2)
        return total / len(maturities)

    best, best_val = None, np.inf
    starts = [(0.15, 0.5, -0.10, 0.15),
              (0.12, 1.0, -0.08, 0.12),
              (0.18, 0.3, -0.15, 0.20)]
    for x0 in starts:
        res = minimize(loss, x0,
                       bounds=[(0.01,0.50),(0.01,5.0),(-0.50,0.10),(0.01,0.60)],
                       method="L-BFGS-B",
                       options={"maxiter": 300})
        if res.fun < best_val:
            best_val = res.fun
            best     = res.x
    sigma, lam, mu_j, sig_j = best
    print(f"  Merton: σ={sigma:.3f}  λ={lam:.3f}  "
          f"μⱼ={mu_j:.3f}  δⱼ={sig_j:.3f}  loss={best_val:.2e}")
    return dict(sigma=sigma, lam=lam, mu_j=mu_j, sig_j=sig_j)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A – Heston calibration: fitted smile vs market IV surface
# ─────────────────────────────────────────────────────────────────────────────
def fig_heston_calibration(S0, r, market, maturities, cal_heston_params):
    labels = {1/12: "T = 1 month",  3/12: "T = 3 months",
              6/12: "T = 6 months", 1.0:  "T = 1 year"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flat

    all_iv_mkt, all_iv_fit = [], []

    for ax, T in zip(axes, maturities):
        d   = market[T]
        K   = d["K"]
        lm  = d["lm"]

        # Fitted Heston IV on a smooth curve
        lm_fine  = np.linspace(lm.min() - 0.02, lm.max() + 0.02, 200)
        K_fine   = S0 * np.exp(lm_fine)
        C_fit    = fft_at(cf_heston, {**cal_heston_params, "T": T}, K_fine)
        iv_fit   = ivol_vec(K_fine, C_fit, S0, T, r)

        # Fitted IV at market strikes (for RMSE)
        C_fit_mkt = fft_at(cf_heston, {**cal_heston_params, "T": T}, K)
        iv_fit_mkt = ivol_vec(K, C_fit_mkt, S0, T, r)
        rmse = np.sqrt(np.nanmean((iv_fit_mkt - d["iv_mkt"])**2)) * 100
        all_iv_mkt.extend(d["iv_mkt"].tolist())
        all_iv_fit.extend(iv_fit_mkt.tolist())

        # Bid-ask band
        ax.fill_between(lm, d["iv_bid"]*100, d["iv_ask"]*100,
                        color="#aec7e8", alpha=0.45, label="Bid–ask spread")

        # Market midpoints
        ax.scatter(lm, d["iv_mkt"]*100, s=22, color="#333333",
                   zorder=4, label="Market mid-IV")

        # Fitted curve
        valid = np.isfinite(iv_fit) & (iv_fit > 0)
        ax.plot(lm_fine[valid], iv_fit[valid]*100,
                color="#1f6fad", lw=2.0, label="Heston (calibrated)")

        ax.axvline(0, color="gray", lw=0.7, ls=":")
        ax.set_title(f"{labels[T]}  (IV RMSE = {rmse:.2f}%)", fontsize=10)
        ax.set_xlabel(r"Log-moneyness  $\ln(K/F)$")
        ax.set_ylabel("Implied volatility (%)")
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    # Parameter box
    p = cal_heston_params
    param_str = (f"Calibrated parameters\n"
                 f"$\\kappa={p['kappa']:.2f}$   "
                 f"$\\theta={p['theta']:.4f}$\n"
                 f"$\\xi={p['sigma_v']:.3f}$      "
                 f"$\\rho={p['rho']:.3f}$\n"
                 f"$v_0={p['v0']:.4f}$")
    fig.text(0.5, -0.01, param_str, ha="center", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", lw=0.8))

    fig.suptitle("Heston Model Calibration — Implied Volatility Surface  "
                 r"($S_0=100$, $r=5\%$, SPX-inspired parameters)",
                 fontsize=12, y=1.01)
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    fig.savefig(f"{OUT}/figA_heston_calibration.pdf")
    fig.savefig(f"{OUT}/figA_heston_calibration.png")
    print("  figA_heston_calibration  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B – Model comparison: BS vs Merton vs Heston on the same market
# Top row: smile fit for T=3M and T=1Y
# Bottom row: IV residuals (model − market) for all maturities
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_comparison(S0, r, market, maturities,
                         bs_sigmas, merton_params, cal_heston_params):

    show_T  = [3/12, 1.0]
    T_label = {1/12: "1M", 3/12: "3M", 6/12: "6M", 1.0: "1Y"}

    colors  = {"Black–Scholes": "#d62728",
               "Merton":        "#ff7f0e",
               "Heston":        "#1f77b4"}
    ls_map  = {"Black–Scholes": (0, (4, 2)),
               "Merton":        (0, (2, 2)),
               "Heston":        "-"}

    fig = plt.figure(figsize=(12, 9))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.45, wspace=0.30)
    ax_top  = [fig.add_subplot(gs[0, j]) for j in range(2)]
    ax_bot  = [fig.add_subplot(gs[1, j]) for j in range(2)]

    # Collect RMSE per model per maturity for bottom panel
    rmse_records = {m: [] for m in ["Black–Scholes", "Merton", "Heston"]}

    for col, T in enumerate(show_T):
        ax = ax_top[col]
        d  = market[T]
        K, lm = d["K"], d["lm"]

        # Smooth grid
        lm_f = np.linspace(lm.min() - 0.02, lm.max() + 0.02, 250)
        K_f  = S0 * np.exp(lm_f)

        # Market
        ax.fill_between(lm, d["iv_bid"]*100, d["iv_ask"]*100,
                        color="#bbbbbb", alpha=0.40, label="Bid–ask")
        ax.scatter(lm, d["iv_mkt"]*100,
                   s=18, color="#222222", zorder=5, label="Market")

        # Three models
        models_cfg = [
            ("Black–Scholes",
             lambda K_arr, T=T: fft_at(
                 cf_bs,
                 dict(S0=S0, r=r, T=T, sigma=bs_sigmas[T]),
                 K_arr)),
            ("Merton",
             lambda K_arr, T=T: fft_at(
                 cf_merton,
                 dict(S0=S0, r=r, T=T, **merton_params),
                 K_arr)),
            ("Heston",
             lambda K_arr, T=T: fft_at(
                 cf_heston,
                 {**cal_heston_params, "T": T},
                 K_arr)),
        ]
        for name, pricer in models_cfg:
            C_f   = pricer(K_f)
            iv_f  = ivol_vec(K_f, C_f, S0, T, r)
            valid = np.isfinite(iv_f) & (iv_f > 0)
            ax.plot(lm_f[valid], iv_f[valid]*100,
                    color=colors[name], lw=1.9,
                    ls=ls_map[name], label=name)

        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"$T$ = {T_label[T]}", fontsize=11)
        ax.set_xlabel(r"Log-moneyness  $\ln(K/F)$")
        ax.set_ylabel("Implied volatility (%)")
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Collect RMSE across all maturities
    for T in maturities:
        d  = market[T]
        K  = d["K"]
        ms = [("Black–Scholes",
               fft_at(cf_bs, dict(S0=S0, r=r, T=T, sigma=bs_sigmas[T]), K)),
              ("Merton",
               fft_at(cf_merton, dict(S0=S0, r=r, T=T, **merton_params), K)),
              ("Heston",
               fft_at(cf_heston, {**cal_heston_params, "T": T}, K))]
        for name, C_m in ms:
            iv_m  = ivol_vec(K, C_m, S0, T, r)
            rmse  = np.sqrt(np.nanmean((iv_m - d["iv_mkt"])**2)) * 100
            rmse_records[name].append(rmse)

    # Bottom: grouped bar chart of IV RMSE
    T_names  = [T_label[T] for T in maturities]
    n_T      = len(maturities)
    x        = np.arange(n_T)
    width    = 0.24
    model_names = ["Black–Scholes", "Merton", "Heston"]

    # Left panel: bar chart
    ax_b0 = ax_bot[0]
    for i, name in enumerate(model_names):
        ax_b0.bar(x + (i - 1) * width, rmse_records[name],
                  width, color=colors[name], alpha=0.85,
                  label=name, zorder=3)
    ax_b0.set_xticks(x)
    ax_b0.set_xticklabels(T_names)
    ax_b0.set_xlabel("Maturity")
    ax_b0.set_ylabel("IV RMSE (%)")
    ax_b0.set_title("IV Fit Error by Maturity")
    ax_b0.legend(fontsize=8)

    # Right panel: IV residuals for T=1Y (most discriminative)
    ax_b1 = ax_bot[1]
    T_res  = 1.0
    d_res  = market[T_res]
    K_res  = d_res["K"]
    lm_res = d_res["lm"]
    ms_res = [("Black–Scholes",
               fft_at(cf_bs, dict(S0=S0, r=r, T=T_res, sigma=bs_sigmas[T_res]), K_res)),
              ("Merton",
               fft_at(cf_merton, dict(S0=S0, r=r, T=T_res, **merton_params), K_res)),
              ("Heston",
               fft_at(cf_heston, {**cal_heston_params, "T": T_res}, K_res))]
    for name, C_m in ms_res:
        iv_m = ivol_vec(K_res, C_m, S0, T_res, r)
        resid = (iv_m - d_res["iv_mkt"]) * 100
        ax_b1.plot(lm_res, resid, "o", ms=4.5,
                   color=colors[name], label=name)
        ax_b1.plot(lm_res, resid, lw=1.5,
                   color=colors[name], ls=ls_map[name])

    # Bid-ask half-spread as shaded band
    half_ba = (d_res["iv_ask"] - d_res["iv_bid"]) / 2 * 100
    ax_b1.fill_between(lm_res, -half_ba, half_ba,
                       color="#bbbbbb", alpha=0.35, label="±½ bid-ask")
    ax_b1.axhline(0, color="gray", lw=0.7, ls=":")
    ax_b1.axvline(0, color="gray", lw=0.7, ls=":")
    ax_b1.set_xlabel(r"Log-moneyness  $\ln(K/F)$")
    ax_b1.set_ylabel("IV residual (%)")
    ax_b1.set_title(r"Residuals — $T = 1$ year")
    ax_b1.legend(fontsize=8, loc="upper left")

    fig.suptitle("Model Comparison: Black–Scholes vs Merton vs Heston  "
                 r"($S_0=100$, $r=5\%$)",
                 fontsize=12, y=1.01)
    fig.savefig(f"{OUT}/figB_model_comparison.pdf")
    fig.savefig(f"{OUT}/figB_model_comparison.png")
    print("  figB_model_comparison  saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output → {OUT}/\n")

    S0, r, true_params, maturities, market = build_market()

    print("Calibrating models…")
    cal_h  = calibrate_heston(S0, r, market, maturities)
    print("  Calibrating BS (per maturity)…")
    bs_sig = calibrate_bs_per_maturity(S0, r, market, maturities)
    print("  Calibrating Merton (joint)…")
    merton = calibrate_merton(S0, r, market, maturities)

    print("\nGenerating figures…")
    fig_heston_calibration(S0, r, market, maturities, cal_h)
    fig_model_comparison(S0, r, market, maturities, bs_sig, merton, cal_h)

    print(f"\nDone — figures saved to {OUT}/")
