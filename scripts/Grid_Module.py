#-------------------------------- Grid Generating Function (Module) -------------------------------------------------------------

# ---- Modules Used:
import numpy as np

# ---- Functions:

#ALPHA Grid for Exp Correction in the Alternative Call Price BS Function
def generate_alpha_grid(alpha_lower_bound, alpha_upper_bound, num_steps=10):
    """
    Generates a grid for the damping coefficient alpha (alpha).
    
    The grid ranges from the specified lower bound to the theoretical
    upper bound, as derived from the paper.

    Input Args:
        alpha_lower_bound (float):  The small positive value to start the grid from.
                                    The paper requires alpha > 0 .
        alpha_upper_bound (float): The theoretical upper bound for alpha.
        num_steps (int): The number of alpha values to generate.

    Returns:
        np.ndarray: An array of alpha values.
    """
    if alpha_lower_bound <= 0:
        raise ValueError("alpha_lower_bound must be > 0")
        
    if alpha_upper_bound <= alpha_lower_bound:
        raise ValueError("alpha_upper_bound must be greater than alpha_lower_bound")

    print(f"Generating alpha grid from {alpha_lower_bound} to {alpha_upper_bound:.4f} (safe upper bound).")
    return np.linspace(alpha_lower_bound, alpha_upper_bound, num_steps)

#ETA Grid for Inegration Domain Discretization
def generate_eta_grid(max_eta, min_eta, num_steps=10):
    """
    Generates a grid for the frequency spacing eta (η)

    For convergence studies, one would test a range of decreasing 
    step sizes. A smaller eta implies a finer integration grid.

    Args:
        max_eta (float): The largest eta value to test.
        min_eta (float): The smallest eta value to test.
        num_steps (int): The number of eta values to generate.

    Returns:
        eta values grid
    """
    if min_eta <= 0:
        raise ValueError("eta values must be positive.")
    if min_eta >= max_eta:
        raise ValueError("min_eta must be less than max_eta.")
    
    #Generating Grid
    print(f"Generating eta grid from {max_eta} to {min_eta}.")
    return np.linspace(min_eta, max_eta, num_steps)

#N (Points Evaluated and Freq Discretization) for FFT Algorithm:
def generate_n_grid(min_power=8, max_power=14):
    """
    Generates a grid for the number of FFT points, N.

    The paper specifies that N is typically a power of 2 .

    Args:
        min_power (int): The smallest power of 2 (e.g., 8 for 2^8 = 256).
        max_power (int): The largest power of 2 (e.g., 14 for 2^14 = 16384).

    Returns:
        np.ndarray: An array of N values, as integers.
    """
    if min_power < 1 or max_power < min_power:
        raise ValueError("min_power must be >= 1 and max_power must be >= min_power.")
        
    powers = np.arange(min_power, max_power + 1)
    n_values = 2**powers
    
    print(f"Generating N grid (powers of 2): {n_values}")
    return n_values.astype(int)

