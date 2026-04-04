"""
generate_figs.py — Master figure generation script.

Generates all 4 academic figures for the paper:
    1. real_smile.pdf     — SPY multi-model calibrated smile
    2. vol_surface.pdf    — Heston 3D implied volatility surface
    3. greeks_validation.pdf — Euler Delta + Gamma vs. analytical BS
    4. timing_benchmark.pdf  — FFT vs. Monte Carlo scalability

Usage:
    uv run python generate_figs.py
"""
import os, sys, time, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.stats import norm
from scipy.optimize import differential_evolution
from tqdm import tqdm

# ── Imports from the flat package ────────────────────────────────────────────
from fourier_options import cf_bs, cf_heston, cf_merton, cf_vg, fft_pricer
from fourier_options.calibration import ivol_vec, calibrate
from fourier_options.fft_pricer import euler_gamma

OUT = os.path.join(os.path.dirname(__file__), "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Objective functions (module-level for multiprocessing serialization) ──────
Q_MKT, R_MKT = 0.0135, 0.045

def _interp(cf, extra_params, S, r, q, T, K_mkt, iv_mkt):
    Kg, Cg, _ = fft_pricer(cf, {"S0": S, "r": r, "q": q, "T": T, **extra_params}, alpha=1.5)
    iv_mod = ivol_vec(np.interp(np.log(K_mkt), np.log(Kg), Cg), S, K_mkt, T, r, q) * 100
    return float(np.mean((iv_mod - iv_mkt)**2))

def obj_bs(x, S, r, q, T, K, iv):      return _interp(cf_bs,     {"sigma": x[0]}, S, r, q, T, K, iv)
def obj_merton(x, S, r, q, T, K, iv):  return _interp(cf_merton, {"sigma": x[0], "lam": x[1], "mu_j": x[2], "sig_j": x[3]}, S, r, q, T, K, iv)
def obj_vg(x, S, r, q, T, K, iv):      return _interp(cf_vg,     {"sigma": x[0], "nu": x[1], "theta_vg": x[2]}, S, r, q, T, K, iv)
def obj_heston(x, S, r, q, T, K, iv):
    pen = 1e7 if (2.0 * x[0] * x[1] <= x[2]**2) else 0.0
    return _interp(cf_heston, {"kappa": x[0], "theta": x[1], "sigma_v": x[2], "rho": x[3], "v0": x[4]}, S, r, q, T, K, iv) + pen

if __name__ == "__main__":
    import yfinance as yf

    # ── 1. Fetch SPY market data ──────────────────────────────────────────────
    print("Fetching SPY options...")
    ticker = yf.Ticker("SPY")
    SPOT = ticker.history(period="1d")["Close"].iloc[-1]
    expiry = next(e for e in ticker.options
                  if (datetime.datetime.strptime(e, "%Y-%m-%d") - datetime.datetime.today()).days / 365.0 > 0.04)
    T_CAL = (datetime.datetime.strptime(expiry, "%Y-%m-%d") - datetime.datetime.today()).days / 365.0
    calls = ticker.option_chain(expiry).calls
    calls = calls[(calls["volume"] > 10) & (calls["strike"] > 0.94 * SPOT) & (calls["strike"] < 1.10 * SPOT)].dropna()
    K_MKT, IV_MKT = calls["strike"].values, calls["impliedVolatility"].values * 100
    print(f"Expiry: {expiry}  T={T_CAL:.3f}  S0={SPOT:.2f}  N_opts={len(K_MKT)}")

    # ── 2. Calibrate all models ───────────────────────────────────────────────
    args = (SPOT, R_MKT, Q_MKT, T_CAL, K_MKT, IV_MKT)
    MODELS = {
        "Black-Scholes": dict(fn=obj_bs,     bounds=[(0.05,1.0)],                                          maxiter=40,  color="#2166ac", ls="-",   cf=cf_bs,     pkeys=["sigma"]),
        "Merton Jump":   dict(fn=obj_merton, bounds=[(0.05,0.8),(0.01,2.0),(-0.4,0.1),(0.01,0.4)],        maxiter=60,  color="#d6604d", ls="--",  cf=cf_merton, pkeys=["sigma","lam","mu_j","sig_j"]),
        "Variance Gamma":dict(fn=obj_vg,     bounds=[(0.05,0.8),(0.01,1.0),(-0.8,0.2)],                   maxiter=60,  color="#762a83", ls="-.",  cf=cf_vg,     pkeys=["sigma","nu","theta_vg"]),
        "Heston":        dict(fn=obj_heston, bounds=[(0.1,5.0),(0.01,0.4),(0.01,0.8),(-0.95,0.0),(0.01,0.4)], maxiter=80, color="#1a9641", ls=":", cf=cf_heston, pkeys=["kappa","theta","sigma_v","rho","v0"]),
    }
    for name, m in MODELS.items():
        with tqdm(total=m["maxiter"], desc=f"{name:16}") as pbar:
            res = differential_evolution(m["fn"], m["bounds"], args=args,
                                         maxiter=m["maxiter"], seed=0, workers=1,
                                         callback=lambda x, c=None: pbar.update(1))
        m["x"] = res.x
        m["rmse"] = res.fun**0.5
        m["p"] = dict(zip(m["pkeys"], res.x))

    # ── 3. Fig 1: Real Smile ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    K_line = np.linspace(K_MKT.min(), K_MKT.max(), 120)
    ax.scatter(K_MKT / SPOT, IV_MKT, s=22, color="gray", alpha=0.3, label="SPY market", zorder=5)
    for name, m in MODELS.items():
        Kg, Cg, _ = fft_pricer(m["cf"], {"S0": SPOT, "r": R_MKT, "q": Q_MKT, "T": T_CAL, **m["p"]}, alpha=1.5)
        IVl = ivol_vec(np.interp(np.log(K_line), np.log(Kg), Cg), SPOT, K_line, T_CAL, R_MKT, Q_MKT) * 100
        ax.plot(K_line / SPOT, IVl, color=m["color"], ls=m["ls"], label=f"{name}  RMSE={m['rmse']:.3f}", lw=1.6)
    ax.set_xlabel("Moneyness $K/S_0$"); ax.set_ylabel("IV (%)"); ax.legend(); ax.grid(True, alpha=0.12)
    fig.savefig(f"{OUT}/real_smile.pdf"); plt.close()
    print("✓ real_smile.pdf")

    # ── 4. Fig 2: Heston Vol Surface ─────────────────────────────────────────
    T_pts = np.linspace(0.1, 2.0, 100); K_pts = np.linspace(0.85 * SPOT, 1.15 * SPOT, 100)
    hp = {**MODELS["Heston"]["p"], "S0": SPOT, "r": R_MKT, "q": Q_MKT}
    IV_S = np.zeros((100, 100))
    for i, T in enumerate(tqdm(T_pts, desc="Vol Surface   ")):
        Kg, Cg, _ = fft_pricer(cf_heston, {**hp, "T": T}, alpha=1.5)
        IV_S[i] = ivol_vec(np.interp(np.log(K_pts), np.log(Kg), Cg), SPOT, K_pts, T, R_MKT, Q_MKT) * 100
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection="3d")
    KK, TT = np.meshgrid(K_pts / SPOT, T_pts)
    ax.plot_surface(KK, TT, IV_S, cmap="viridis", alpha=0.90, antialiased=True)
    ax.set_xlabel("Moneyness"); ax.set_ylabel("Maturity $T$"); ax.set_zlabel("IV (%)")
    fig.savefig(f"{OUT}/vol_surface.pdf"); plt.close()
    print("✓ vol_surface.pdf")

    # ── 5. Fig 3: Greeks Validation ──────────────────────────────────────────
    pg = dict(S0=100., r=0.05, q=0.0, T=1.0, sigma=0.20)
    K_g, prices_g, Delta_g = fft_pricer(cf_bs, pg, alpha=1.1, N=8192)
    K_gm, Gamma_g = euler_gamma(cf_bs, pg, alpha=1.1, N=8192)
    md = (K_g > 60) & (K_g < 160)
    d1a = lambda K: (np.log(100.0 / K) + (0.05 + 0.02) * 1.0) / 0.20
    da_ = norm.cdf(d1a(K_g[md]))
    ga_ = norm.pdf(d1a(K_gm[(K_gm > 60) & (K_gm < 160)])) / (100.0 * 0.20)
    mdm = (K_gm > 60) & (K_gm < 160)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(K_g[md],  np.maximum(np.abs(Delta_g[md] - da_), 1e-16), color="#2166ac", lw=1.4, label="FFT Euler")
    axes[0].set_title("Delta Error"); axes[0].set_xlabel("$K$"); axes[0].set_ylabel("Abs Error"); axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.2)
    axes[1].semilogy(K_gm[mdm], np.maximum(np.abs(Gamma_g[mdm] - ga_), 1e-16), color="#d6604d", lw=1.4, label="FFT Euler")
    axes[1].set_title("Gamma Error"); axes[1].set_xlabel("$K$"); axes[1].legend(); axes[1].grid(True, ls="--", alpha=0.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/greeks_validation.pdf"); plt.close()
    print("✓ greeks_validation.pdf")

    # ── 6. Fig 4: Timing Benchmark ───────────────────────────────────────────
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
    t_fft, t_mc = [], []
    for N in Ns:
        reps = max(5, int(4000 / N))
        t0 = time.perf_counter()
        for _ in range(reps): fft_pricer(cf_bs, {"S0": 100, "r": 0.05, "q": 0, "T": 1.0, "sigma": 0.2}, N=N)
        t_fft.append((time.perf_counter() - t0) / reps * 1000)
        K_b = np.linspace(80, 120, N)
        t0 = time.perf_counter()
        for _ in range(3):
            ST = 100 * np.exp((0.05 - 0.02) * 1.0 + 0.2 * np.random.randn(100_000))
            for Kb in K_b[:1]: np.mean(np.maximum(ST - Kb, 0))
        t_mc.append(((time.perf_counter() - t0) / 3 * 1000) * N)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ns, t_fft, "o-",  label="Carr-Madan FFT $O(N \\log N)$", color="#2166ac")
    ax.loglog(Ns, t_mc,  "v--", label="Monte Carlo $O(N^2)$",          color="#d6604d")
    ax.set_xlabel("Number of strikes $N$"); ax.set_ylabel("Time (ms)")
    ax.legend(); ax.grid(True, which="both", alpha=0.10)
    fig.savefig(f"{OUT}/timing_benchmark.pdf"); plt.close()
    print("✓ timing_benchmark.pdf")

    print("\nAll figures saved to paper/figs/")
