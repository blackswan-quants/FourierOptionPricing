#-------------------------------- Error and Computing Time Evaluations -------------------------------------------------------------

# ---- Standard Libraries:
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Callable
from mpl_toolkits.mplot3d import Axes3D
from src.fft_pricer import fft_pricer
from characteristic_functions import cf_bs

# ---- Purpose(s):
# 1) Implement an error evaluation comparing FFT pricing vs. the Black–Scholes benchmark
# 2) Compute the execution time for each FFT pricing run
# 3) Plot error surfaces over parameter grids


#-------------------------------- Time and Error Functions -------------------------------------------------------------
def fft_runs(alpha_grid: Iterable[float], eta_grid: Iterable[float], n_grid: Iterable[int],
             bs_price: float, fft_pricer: Callable, params: dict[str, float], strike: float) -> pd.DataFrame:

    """
    Runs the FFT pricer over all combinations of alpha, eta, and N,
    recording FFT price, elapsed time, and error vs. Black–Scholes price.

    Args:
        alpha_grid (iterable): Range of damping factors α
        eta_grid (iterable):   Range of frequency spacings η
        n_grid (iterable):     Range of FFT sizes N (typically powers of 2)
        bs_price (float):      Black–Scholes closed-form benchmark price
        fft_pricer (callable): Function implementing FFT pricing with signature:
                               fft_pricer(cf, params, alpha, n, eta)
        params:          Parameters required by fft_pricer (tuple/list/dict)

    Returns:
        pd.DataFrame: One row per run with:
                      ["fft_price", "elapsed_time", "alpha", "eta", "n", "error"]
    """
    
    experiments = []

    for alpha in alpha_grid:
        for eta in eta_grid:
            for n in n_grid:

                # Start timing
                start = time.perf_counter()
                k_temp, fft_prices = fft_pricer(cf_bs, params, alpha, n, eta)
                fft_price = np.interp(strike, k_temp, fft_prices) 
                end = time.perf_counter()

                elapsed_time = end - start
                error = fft_price - bs_price

                experiments.append(
                    [fft_price, elapsed_time, alpha, eta, n, error]
                )
    
    return pd.DataFrame(
        experiments,
        columns=["fft_price", "elapsed_time", "alpha", "eta", "n", "error"]
    )


#-------------------------------- Plotting Function -------------------------------------------------------------
def plot_error_surface(
    exp_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    param_k: str,
    param_k_fix: float,
    plot_type: str = "contourf",
    log_scale_x: bool = False,
    log_scale_y: bool = False
):
    """
    Plots either a 3D surface or a 2D contour map of the FFT pricing error
    as a function of two varying parameters (param_x, param_y) while
    fixing a third parameter (param_k = param_k_fix).

    Args:
        exp_df (DataFrame): Experiment results from fft_runs()
        param_x (str): Parameter for X-axis (e.g., 'alpha')
        param_y (str): Parameter for Y-axis (e.g., 'eta')
        param_k (str): Parameter to keep fixed
        param_k_fix (float): Value of param_k to filter the dataset
        plot_type (str): 'surface3d' or 'contourf'
        log_scale_x (bool): Apply logarithmic X-axis
        log_scale_y (bool): Apply logarithmic Y-axis
    """

    LABEL_ERROR = "FFT Error vs BS"

    # Filter rows where param_k is fixed
    filtered = exp_df[exp_df[param_k] == param_k_fix]

    # Build grid axes in sorted order
    x_values = np.sort(filtered[param_x].unique())
    y_values = np.sort(filtered[param_y].unique())

    X, Y = np.meshgrid(x_values, y_values)

    # Pivot error values to build Z matrix
    pivot_table = filtered.pivot(index=param_y, columns=param_x, values="error")
    Z = pivot_table.loc[y_values, x_values].values

    if np.isnan(Z).any():
        print("Warning: missing values detected in Z surface.")

    # Plotting
    fig = plt.figure(figsize=(10, 7))

    if plot_type == "surface3d":
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        fig.colorbar(surf, shrink=0.5, aspect=10, label=LABEL_ERROR)
        ax.set_zlabel("Error")
    else:
        ax = fig.add_subplot(111)
        cf = ax.contourf(X, Y, Z, levels=15, cmap="viridis")
        fig.colorbar(cf, label=LABEL_ERROR)

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    plt.title(f"FFT Error vs BS — {param_x} & {param_y} (fixed {param_k}={param_k_fix})")
    plt.show()
