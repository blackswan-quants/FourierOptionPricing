"""
Calibration utilities — IV inversion and multi-model fitting.

Main entry point:
    result = calibrate(model, bounds, spot, r, q, T, strikes, ivs)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm


def ivol_vec(prices: np.ndarray, S: float, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
    """
    Newton-Raphson IV inversion (vectorized).
    Uses OTM switching: ITM Calls → converted to Put via put-call parity before inversion.
    """
    vols = np.full_like(prices, 0.3)
    qd, rd = np.exp(-q * T), np.exp(-r * T)
    for _ in range(40):
        d1 = (np.log(S / K) + (r - q + 0.5 * vols**2) * T) / (vols * np.sqrt(T))
        d2 = d1 - vols * np.sqrt(T)
        mask = K >= S  # OTM for calls
        diff = np.where(
            mask,
            S * qd * norm.cdf(d1) - K * rd * norm.cdf(d2) - prices,
            K * rd * norm.cdf(-d2) - S * qd * norm.cdf(-d1) - (prices - S * qd + K * rd),
        )
        vega = S * qd * np.sqrt(T) * norm.pdf(d1)
        vols -= diff / np.maximum(vega, 1e-8)
        vols = np.clip(vols, 1e-4, 5.0)
    return vols


def calibrate(
    obj_fn: Callable,
    bounds: list[tuple[float, float]],
    args: tuple,
    maxiter: int = 100,
    workers: int = 1,
    label: str = "",
) -> dict:
    """
    Calibrate a model via Differential Evolution.

    Parameters
    ----------
    obj_fn : callable
        Objective function f(x, *args) → float (MSE in IV space).
    bounds : list of (min, max)
        Parameter bounds for the optimizer.
    args : tuple
        Extra arguments passed to obj_fn.
    maxiter : int
        Maximum number of DE generations.
    workers : int
        Number of parallel workers (-1 = all CPUs).
    label : str
        Name shown in progress output.

    Returns
    -------
    dict with keys: params (np.ndarray), rmse (float), success (bool).
    """
    result = differential_evolution(
        obj_fn, bounds, args=args,
        maxiter=maxiter, seed=0, workers=workers,
        tol=1e-7, polish=True,
    )
    return {"params": result.x, "rmse": float(result.fun**0.5), "success": result.success}
