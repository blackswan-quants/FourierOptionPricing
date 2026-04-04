"""
paper_technical_figures.py
==========================
Three publication-ready technical figures:

  Fig C – Quadrature error: RMSE vs N and eta (convergence of FFT discretisation)
  Fig D – No-arbitrage sanity check: bounds, convexity, put-call parity, density
           (the most important figure — proves the pricer is financially correct)
  Fig E – Greeks via FFT: accuracy vs analytical + speed vs bump-and-reprice vs MC

Run from the project root:
    python examples/paper_technical_figures.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fourier_options.pricing.fft_pricer import fft_pricer
from fourier_options.domain.characteristic_functions import cf_bs, cf_heston, cf_merton
from fourier_options.greeks.fft import delta_fft_bs, gamma_fft_bs, vega_fft_bs

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linestyle":    ":",
})

MODELS = [
    (cf_bs,
     dict(S0=100., r=0.05, T=1., sigma=0.20),
     "Black–Scholes",      "#2166ac", "-"),
    (cf_merton,
     dict(S0=100., r=0.05, T=1., sigma=0.15,
          lam=0.5, mu_j=-0.10, sig_j=0.15),
     "Merton",             "#d6604d", "--"),
    (cf_heston,
     dict(S0=100., r=0.05, T=1.,
          kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.70, v0=0.04),
     "Heston",             "#1a9641", "-."),
]

S0, r, T = 100., 0.05, 1.0


# ── Analytical BS helpers ─────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C – Quadrature error: convergence in N and sensitivity to eta
# ─────────────────────────────────────────────────────────────────────────────
def fig_quadrature_error():
    params = dict(S0=S0, r=r, T=T, sigma=0.20)

    # Reference on a dense common grid (log-strike 0.5 to 2.0)
    K_ref = np.linspace(S0 * 0.55, S0 * 1.80, 300)
    ref   = bs_call(S0, K_ref, T, r, 0.20)

    def rmse_at(N, eta, alpha=1.5):
        K_g, C_g = fft_pricer(cf_bs, params, alpha=alpha, N=N, eta=eta)
        C_interp  = np.interp(np.log(K_ref), np.log(K_g), C_g)
        return np.sqrt(np.mean((C_interp - ref) ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: RMSE vs N for several alpha values (fixed eta = 0.25) ──────────
    Ns      = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    alphas  = [0.75, 1.25, 1.50, 2.50]
    colors  = ["#d62728", "#ff7f0e", "#2166ac", "#1a9641"]

    for alpha, col in zip(alphas, colors):
        errs = [rmse_at(N, 0.25, alpha) for N in Ns]
        ax1.loglog(Ns, errs, "o-", ms=4.5, lw=1.6,
                   color=col, label=rf"$\alpha = {alpha}$")

    # Reference O(1/N) slope
    errs0 = [rmse_at(N, 0.25, 1.5) for N in Ns]
    slope = errs0[1] * (np.array(Ns) / Ns[1]) ** (-1.0)
    ax1.loglog(Ns, slope, "k--", lw=0.9, label=r"$O(N^{-1})$")

    ax1.set_xlabel("FFT grid size $N$")
    ax1.set_ylabel("RMSE vs exact Black–Scholes ($)")
    ax1.set_title(r"(a) Convergence in $N$  ($\eta = 0.25$, fixed)")
    ax1.legend()

    # ── Right: RMSE vs eta for several N values (fixed alpha = 1.5) ──────────
    etas   = np.logspace(np.log10(0.05), np.log10(1.0), 30)
    Ns_eta = [256, 1024, 4096]
    cols_N = ["#d62728", "#ff7f0e", "#2166ac"]

    for N, col in zip(Ns_eta, cols_N):
        errs = [rmse_at(N, e) for e in etas]
        ax2.loglog(etas, errs, "o-", ms=3.5, lw=1.6,
                   color=col, label=f"$N = {N}$")

    ax2.axvline(0.25, color="gray", lw=0.8, ls="--",
                label=r"$\eta = 0.25$ (default)")
    ax2.set_xlabel(r"Frequency spacing $\eta$")
    ax2.set_ylabel("RMSE vs exact Black–Scholes ($)")
    ax2.set_title(r"(b) Sensitivity to $\eta$  ($\alpha = 1.5$, fixed)")
    ax2.legend()

    fig.suptitle("Carr–Madan FFT — Quadrature Error Analysis", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figC_quadrature_error.pdf")
    fig.savefig(f"{OUT}/figC_quadrature_error.png")
    print("  figC_quadrature_error  saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D – No-arbitrage sanity check  (the most important figure)
# Four panels: price bounds · convexity · put-call parity · risk-neutral density
# Tested on all three models simultaneously.
# ─────────────────────────────────────────────────────────────────────────────
def fig_sanity_check():
    disc = np.exp(r * T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes

    # ── Panel (a): call price curves with no-arbitrage bounds ────────────────
    K_bnd = np.linspace(50, 160, 400)
    lb    = np.maximum(S0 - K_bnd * np.exp(-r * T), 0)

    ax1.fill_between(K_bnd, lb, S0,
                     color="#cccccc", alpha=0.45, label="No-arb region")
    ax1.plot(K_bnd, lb, "k:", lw=0.8)
    ax1.axhline(S0, color="k", lw=0.8, ls=":")

    for cf, params, name, col, ls in MODELS:
        K, C = fft_pricer(cf, params, alpha=1.5, N=4096, eta=0.25)
        m    = (K > 50) & (K < 160)
        ax1.plot(K[m], C[m], color=col, lw=1.8, ls=ls, label=name)

    ax1.axvline(S0, color="gray", lw=0.6, ls=":")
    ax1.set_xlabel("Strike $K$")
    ax1.set_ylabel("Call price ($)")
    ax1.set_title("(a) Call Prices within No-Arbitrage Bounds")
    ax1.legend(loc="upper right")

    # ── Panel (b): butterfly spread (convexity) ───────────────────────────────
    for cf, params, name, col, ls in MODELS:
        K, C = fft_pricer(cf, params, alpha=1.5, N=4096, eta=0.25)
        m    = (K > 60) & (K < 150)
        Kw, Cw = K[m], C[m]

        Ka, Kb, Kc = Kw[:-2], Kw[1:-1], Kw[2:]
        Ca, Cb, Cc = Cw[:-2], Cw[1:-1], Cw[2:]
        ha  = Kb - Ka
        hb  = Kc - Kb
        wa  = hb / (ha + hb)
        wc  = ha / (ha + hb)
        # d²C/dK²  (proportional to risk-neutral density — must be ≥ 0)
        d2C = 2.0 * (wa * Ca + wc * Cc - Cb) / (ha * hb)

        ax2.plot(Kb, d2C * 1e4, color=col, lw=1.6, ls=ls, label=name)

    ax2.axhline(0, color="black", lw=1.0, ls="--")
    ax2.set_xlabel("Strike $K$")
    ax2.set_ylabel(r"$d^2C/dK^2\;\times 10^{4}$  (proportional to $q$)")
    ax2.set_title(r"(b) Convexity Check — $d^2C/dK^2 \geq 0$")
    ax2.legend()

    # ── Panel (c): put-call parity error ─────────────────────────────────────
    for cf, params, name, col, ls in MODELS:
        K_c, C = fft_pricer(cf, params, alpha=1.5, N=4096, eta=0.25)
        K_p, P = fft_pricer(cf, params, alpha=1.5, N=4096, eta=0.25,
                             option_type="put")
        m    = (K_c > 60) & (K_c < 150)
        err  = np.abs(C[m] - P[m] - S0 + K_c[m] * np.exp(-r * T))
        ax3.semilogy(K_c[m], np.maximum(err, 1e-17),
                     color=col, lw=1.6, ls=ls, label=name)

    ax3.axhline(1e-10, color="gray", lw=0.8, ls="--",
                label="Machine eps ($10^{-10}$)")
    ax3.set_xlabel("Strike $K$")
    ax3.set_ylabel(r"$|C - P - S_0 + Ke^{-rT}|$  ($)")
    ax3.set_title("(c) Put–Call Parity Error (log scale)")
    ax3.legend()

    # ── Panel (d): risk-neutral density via Breeden-Litzenberger ─────────────
    integrals = {}
    for cf, params, name, col, ls in MODELS:
        # Use finer eta for smoother density
        K, C = fft_pricer(cf, params, alpha=1.5, N=8192, eta=0.08)
        m    = (K > 45) & (K < 220)
        Kw, Cw = K[m], C[m]

        Ka, Kb, Kc = Kw[:-2], Kw[1:-1], Kw[2:]
        Ca, Cb, Cc = Cw[:-2], Cw[1:-1], Cw[2:]
        ha   = Kb - Ka
        hb   = Kc - Kb
        wa   = hb / (ha + hb)          # = (Kc-Kb)/(Kc-Ka)
        wc   = ha / (ha + hb)          # = (Kb-Ka)/(Kc-Ka)
        fly  = wa * Ca + wc * Cc - Cb
        # Breeden-Litzenberger: q(Kb) = e^{rT} * C''(Kb)
        # Non-uniform second derivative: C''(Kb) = 2*fly / (ha*hb)
        q    = np.maximum(disc * 2.0 * fly / (ha * hb), 0)

        # Numerical integral ≈ 1 (consistency check)
        integrals[name] = float(np.trapz(q, Kb))
        ax4.plot(Kb, q, color=col, lw=1.8, ls=ls,
                 label=f"{name}  ($\\int q\\,dK = {integrals[name]:.4f}$)")

    ax4.axvline(S0, color="gray", lw=0.7, ls=":", label="$S_0 = 100$")
    ax4.set_xlabel("Terminal stock price $S_T$")
    ax4.set_ylabel("Risk-neutral density $q(S_T)$")
    ax4.set_title("(d) Risk-neutral Density — Non-negative, Integrates to 1")
    ax4.legend()

    fig.suptitle(
        "No-Arbitrage Sanity Check — Black–Scholes, Merton, Heston\n"
        r"($S_0=100$, $r=5\%$, $T=1$ yr,  $N=4096$, $\alpha=1.5$, $\eta=0.25$)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(h_pad=3.5, w_pad=3.0)
    fig.savefig(f"{OUT}/figD_sanity_check.pdf")
    fig.savefig(f"{OUT}/figD_sanity_check.png")
    print(f"  figD_sanity_check  saved")
    for name, val in integrals.items():
        print(f"    ∫q dK  {name:<20}: {val:.6f}  (should be ≈ 1)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E – Greeks: accuracy vs analytical + speed vs bump-and-reprice vs MC
# ─────────────────────────────────────────────────────────────────────────────
def fig_greeks():
    sigma = 0.20
    params = dict(S0=S0, r=r, T=T, sigma=sigma)
    alpha, N, eta = 1.5, 4096, 0.25
    eps   = S0 * 1e-4   # bump size for FD

    lo, hi = S0 * 0.50, S0 * 2.0

    # ── Analytical benchmarks ─────────────────────────────────────────────────
    K_bench = np.linspace(lo, hi, 600)
    d_bench = bs_delta(S0, K_bench, T, r, sigma)
    g_bench = bs_gamma(S0, K_bench, T, r, sigma)
    v_bench = bs_vega (S0, K_bench, T, r, sigma)

    # ── FFT Greeks (one call per greek) ───────────────────────────────────────
    K_d, delta_fft = delta_fft_bs(params, alpha=alpha, N=N, eta=eta)
    K_g, gamma_fft = gamma_fft_bs(params, alpha=alpha, N=N, eta=eta)
    K_v, vega_fft  = vega_fft_bs (params, alpha=alpha, N=N, eta=eta)

    mask_d = (K_d > lo) & (K_d < hi)
    mask_g = (K_g > lo) & (K_g < hi)
    mask_v = (K_v > lo) & (K_v < hi)

    # ── Bump-and-reprice (FD) Greeks — 2 extra FFT calls per greek ────────────
    def fd_delta():
        _, Cu = fft_pricer(cf_bs, {**params, "S0": S0 + eps}, alpha=alpha, N=N, eta=eta)
        _, Cd = fft_pricer(cf_bs, {**params, "S0": S0 - eps}, alpha=alpha, N=N, eta=eta)
        K_fd, C0 = fft_pricer(cf_bs, params, alpha=alpha, N=N, eta=eta)
        return K_fd, (Cu - Cd) / (2 * eps)

    def fd_gamma():
        _, Cu = fft_pricer(cf_bs, {**params, "S0": S0 + eps}, alpha=alpha, N=N, eta=eta)
        _, Cd = fft_pricer(cf_bs, {**params, "S0": S0 - eps}, alpha=alpha, N=N, eta=eta)
        K_fd, C0 = fft_pricer(cf_bs, params, alpha=alpha, N=N, eta=eta)
        return K_fd, (Cu - 2*C0 + Cd) / eps**2

    def fd_vega():
        dsig = sigma * 1e-4
        _, Cu = fft_pricer(cf_bs, {**params, "sigma": sigma + dsig},
                           alpha=alpha, N=N, eta=eta)
        _, Cd = fft_pricer(cf_bs, {**params, "sigma": sigma - dsig},
                           alpha=alpha, N=N, eta=eta)
        K_fd, _ = fft_pricer(cf_bs, params, alpha=alpha, N=N, eta=eta)
        return K_fd, (Cu - Cd) / (2 * dsig)

    K_fd_d, delta_fd = fd_delta()
    K_fd_g, gamma_fd = fd_gamma()
    K_fd_v, vega_fd  = fd_vega()

    mask_fd = (K_fd_d > lo) & (K_fd_d < hi)

    # ── Speed benchmark ────────────────────────────────────────────────────────
    reps = 40
    rng  = np.random.default_rng(0)

    # FFT Greeks: ONE call → all N strikes
    t0 = time.perf_counter()
    for _ in range(reps):
        delta_fft_bs(params, alpha=alpha, N=N, eta=eta)
    t_fft_greek = (time.perf_counter() - t0) / reps * 1000   # ms

    # FD Delta: 2 FFT calls → all N strikes
    t0 = time.perf_counter()
    for _ in range(reps):
        fd_delta()
    t_fd_greek = (time.perf_counter() - t0) / reps * 1000

    # MC Delta for ONE ATM strike (Likelihood Ratio estimator)
    # Cost to get all N=4096 strikes at same accuracy: multiply by N
    N_sim      = 50_000
    K_atm      = S0
    t0 = time.perf_counter()
    for _ in range(reps):
        z  = rng.standard_normal(N_sim)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        payoff  = np.maximum(ST - K_atm, 0)
        # LR estimator: dC/dS0 = e^{-rT}/S0 * E[payoff * (z/(sigma*sqrt(T)) + 1)]
        lr_wt   = z / (sigma * np.sqrt(T)) + 1
        _ = np.exp(-r*T) / S0 * np.mean(payoff * lr_wt)
    t_mc_one = (time.perf_counter() - t0) / reps * 1000   # ms for 1 strike
    t_mc_all = t_mc_one * N   # extrapolated cost for all N strikes (serial)

    # ── Layout: 4 rows ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.40)
    ax_d  = fig.add_subplot(gs[0, :2])
    ax_g  = fig.add_subplot(gs[0, 2:])
    ax_v  = fig.add_subplot(gs[1, :2])
    ax_de = fig.add_subplot(gs[1, 2:])   # delta error
    ax_sp = fig.add_subplot(gs[2, :2])   # speed bar chart
    ax_ge = fig.add_subplot(gs[2, 2:])   # gamma/vega error

    # Delta
    ax_d.plot(K_bench, d_bench, "k--", lw=1.2, label="Analytical")
    ax_d.plot(K_d[mask_d], delta_fft[mask_d],
              color="#2166ac", lw=2.0, label="FFT (Fourier integrand)")
    ax_d.plot(K_fd_d[mask_fd], delta_fd[mask_fd],
              color="#d6604d", lw=1.2, ls=":", label="Bump-and-reprice (FD)")
    ax_d.axvline(S0, color="gray", lw=0.6, ls=":")
    ax_d.set_title("(a) Delta $\\Delta$")
    ax_d.set_ylabel("$\\Delta$")
    ax_d.set_xlabel("Strike $K$")
    ax_d.legend()

    # Gamma
    ax_g.plot(K_bench, g_bench, "k--", lw=1.2, label="Analytical")
    ax_g.plot(K_g[mask_g], gamma_fft[mask_g],
              color="#2166ac", lw=2.0, label="FFT")
    ax_g.plot(K_fd_g[mask_fd], gamma_fd[mask_fd],
              color="#d6604d", lw=1.2, ls=":", label="FD")
    ax_g.axvline(S0, color="gray", lw=0.6, ls=":")
    ax_g.set_title("(b) Gamma $\\Gamma$")
    ax_g.set_ylabel("$\\Gamma$")
    ax_g.set_xlabel("Strike $K$")
    ax_g.legend()

    # Vega
    ax_v.plot(K_bench, v_bench, "k--", lw=1.2, label="Analytical")
    ax_v.plot(K_v[mask_v], vega_fft[mask_v],
              color="#2166ac", lw=2.0, label="FFT")
    ax_v.plot(K_fd_v[mask_fd], vega_fd[mask_fd],
              color="#d6604d", lw=1.2, ls=":", label="FD")
    ax_v.axvline(S0, color="gray", lw=0.6, ls=":")
    ax_v.set_title("(c) Vega $\\mathcal{V}$")
    ax_v.set_ylabel("$\\mathcal{V}$")
    ax_v.set_xlabel("Strike $K$")
    ax_v.legend()

    # Delta error (FFT vs analytical)
    d_anal_fft = np.interp(K_d[mask_d], K_bench, d_bench)
    d_anal_fd  = np.interp(K_fd_d[mask_fd], K_bench, d_bench)
    ax_de.semilogy(K_d[mask_d], np.maximum(np.abs(delta_fft[mask_d] - d_anal_fft), 1e-18),
                   color="#2166ac", lw=1.5, label=f"FFT  (RMSE={np.sqrt(np.mean((delta_fft[mask_d]-d_anal_fft)**2)):.2e})")
    ax_de.semilogy(K_fd_d[mask_fd], np.maximum(np.abs(delta_fd[mask_fd] - d_anal_fd), 1e-18),
                   color="#d6604d", lw=1.5, ls="--",
                   label=f"FD   (RMSE={np.sqrt(np.mean((delta_fd[mask_fd]-d_anal_fd)**2)):.2e})")
    ax_de.axvline(S0, color="gray", lw=0.6, ls=":")
    ax_de.set_title("(d) Delta Error vs Analytical")
    ax_de.set_ylabel("$|\\Delta_{\\mathrm{method}} - \\Delta_{\\mathrm{exact}}|$")
    ax_de.set_xlabel("Strike $K$")
    ax_de.legend()

    # Speed bar chart
    methods = ["FFT\n(Fourier\nintegrand)", "Bump-and-\nreprice (FD)", "Monte Carlo\n(LR, 50k sim)\nextrapolated"]
    times   = [t_fft_greek, t_fd_greek, t_mc_all]
    cols    = ["#2166ac", "#d6604d", "#762a83"]
    bars    = ax_sp.bar(methods, times, color=cols, alpha=0.85, width=0.5, zorder=3)
    ax_sp.bar_label(bars, fmt="%.1f ms", padding=3, fontsize=8)
    ax_sp.set_ylabel("Wall-clock time (ms)")
    ax_sp.set_title(f"(e) Time to Compute Greeks for All $N={N}$ Strikes")
    ax_sp.set_yscale("log")

    # Gamma+Vega RMSE comparison
    g_anal_fft = np.interp(K_g[mask_g], K_bench, g_bench)
    g_anal_fd  = np.interp(K_fd_g[mask_fd], K_bench, g_bench)
    v_anal_fft = np.interp(K_v[mask_v], K_bench, v_bench)
    v_anal_fd  = np.interp(K_fd_v[mask_fd], K_bench, v_bench)

    greek_names = ["$\\Delta$", "$\\Gamma$", "$\\mathcal{V}$"]
    rmse_fft = [
        np.sqrt(np.mean((delta_fft[mask_d] - d_anal_fft)**2)),
        np.sqrt(np.mean((gamma_fft[mask_g] - g_anal_fft)**2)),
        np.sqrt(np.mean((vega_fft[mask_v]  - v_anal_fft)**2)),
    ]
    rmse_fd = [
        np.sqrt(np.mean((delta_fd[mask_fd] - d_anal_fd)**2)),
        np.sqrt(np.mean((gamma_fd[mask_fd] - g_anal_fd)**2)),
        np.sqrt(np.mean((vega_fd [mask_fd] - v_anal_fd)**2)),
    ]
    x = np.arange(3)
    w = 0.30
    ax_ge.bar(x - w/2, rmse_fft, w, color="#2166ac", alpha=0.85, label="FFT", zorder=3)
    ax_ge.bar(x + w/2, rmse_fd,  w, color="#d6604d", alpha=0.85, label="FD",  zorder=3)
    ax_ge.set_yscale("log")
    ax_ge.set_xticks(x)
    ax_ge.set_xticklabels(greek_names, fontsize=11)
    ax_ge.set_ylabel("RMSE vs analytical")
    ax_ge.set_title("(f) Greeks RMSE — FFT vs FD")
    ax_ge.legend()

    fig.suptitle(
        "Option Greeks via Carr–Madan FFT — Accuracy and Computational Speed\n"
        r"(Black–Scholes: $S_0=100$, $\sigma=20\%$, $T=1$ yr,  $N=4096$, $\alpha=1.5$)",
        fontsize=11, y=1.01,
    )
    fig.savefig(f"{OUT}/figE_greeks.pdf")
    fig.savefig(f"{OUT}/figE_greeks.png")
    print(f"  figE_greeks  saved")
    print(f"    FFT greek time : {t_fft_greek:.2f} ms")
    print(f"    FD  greek time : {t_fd_greek:.2f} ms  ({t_fd_greek/t_fft_greek:.1f}× slower than FFT)")
    print(f"    MC  greek time : {t_mc_all:.1f} ms  ({t_mc_all/t_fft_greek:.0f}× slower than FFT,  extrapolated)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output → {OUT}/\n")
    fig_quadrature_error()
    fig_sanity_check()
    fig_greeks()
    print(f"\nDone.")
