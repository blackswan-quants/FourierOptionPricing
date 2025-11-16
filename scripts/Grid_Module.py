#-------------------------------- Grid Generating Functions -------------------------------------------------------------

# ---- Standard Libraries:
import numpy as np


# ---- Purpose(s):
# Provide parameter grids for FFT pricing experiments:
# 1) Damping α grid
# 2) Frequency spacing η grid
# 3) FFT size N grid


#-------------------------------- ALPHA Grid -------------------------------------------------------------
def generate_alpha_grid(alpha_lower_bound, alpha_upper_bound, num_steps=10):
    """
    Generates a grid for the damping coefficient α.

    Args:
        alpha_lower_bound (float): Lower bound (must be > 0)
        alpha_upper_bound (float): Upper bound for α
        num_steps (int): Number of grid points

    Returns:
        np.ndarray: Grid of alpha values
    """
    if alpha_lower_bound <= 0:
        raise ValueError("alpha_lower_bound must be > 0.")
    if alpha_upper_bound <= alpha_lower_bound:
        raise ValueError("alpha_upper_bound must exceed alpha_lower_bound.")

    print(f"Generating alpha grid from {alpha_lower_bound} to {alpha_upper_bound:.4f}.")
    return np.linspace(alpha_lower_bound, alpha_upper_bound, num_steps)


#-------------------------------- ETA Grid -------------------------------------------------------------
def generate_eta_grid(min_eta, max_eta, num_steps=10):
    """
    Generates a grid for the frequency spacing η.
    Smaller η ⇒ finer integration grid.

    Args:
        min_eta (float): Smallest η value
        max_eta (float): Largest η value
        num_steps (int): Number of grid points

    Returns:
        np.ndarray: Grid of eta values
    """
    if min_eta <= 0:
        raise ValueError("min_eta must be > 0.")
    if max_eta <= min_eta:
        raise ValueError("max_eta must exceed min_eta.")

    print(f"Generating eta grid from {min_eta} to {max_eta}.")
    return np.linspace(min_eta, max_eta, num_steps)


#-------------------------------- N Grid -------------------------------------------------------------
def generate_n_grid(min_power=8, max_power=14):
    """
    Generates a grid of FFT sizes N = 2^k.

    Args:
        min_power (int): Smallest exponent (e.g., 8 → 256)
        max_power (int): Largest exponent (e.g., 14 → 16384)

    Returns:
        np.ndarray: Array of N values
    """
    if min_power < 1 or max_power < min_power:
        raise ValueError("min_power must be >= 1 and max_power >= min_power.")

    powers = np.arange(min_power, max_power + 1)
    n_values = 2 ** powers

    print(f"Generating N grid (powers of 2): {n_values}")
    return n_values.astype(int)
