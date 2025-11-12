#-------------------------------- Error and Computing Time Evaluations -------------------------------------------------------------

# ---- Standard Libraries:
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ---- Purpose(s):

# // 1) Implementing an Error Evaluation for FFT pricing vs BS Formula (benchmark)
#    2) Calculate Computing Time of the Pricing Process
#    3) Plotting Errors

# ---- Functions:
#-------------------------------- Time and Error Functions -------------------------------------------------------------
def fft_runs(alpha_grid, eta_grid, n_grid, bs_price, fft_pricer, option_param):
    """
    Runs pricing via FFT pricer while computing elapsed time and saving parameters of the calculation
    in a list, generating a results matrix 

    Args:
        alpha_grid: the range of alpha values
        eta_grid: the range of eta values
        n_grid: the range of n values
        bs_price: the price given by the BS formula of the considered option
        FFT_pricer: the function pricing via FFT Method
        option_param (list): parameters of the considered option

    Output: 
        A Matrix containing for each combination of parameter a log of the results
    """

    #// Creating an experiments pandas Dataframe
    experiments = []

    for alpha in alpha_grid:
        for eta in eta_grid:
            for n in n_grid:
                #// Start Timing current run
                start_time = time.perf_counter()
                fft_price = fft_pricer(alpha, eta, n, option_param)
                end_time = time.perf_counter()
                #// End Timing current run

                #// Computing elapsed time and error
                elapsed_time = end_time - start_time
                error = fft_price - bs_price

                #// Adding to experiments log
                run = [fft_price, elapsed_time, alpha, eta, n, error]
                experiments.append(run)
    
    experiments_df = pd.DataFrame(experiments, columns=["fft_price", "elapsed_time", 
                                                "alpha", "eta", "n", "error"])
    return experiments_df

def plot_error_surface(exp_df: pd.DataFrame, param_x: str, param_y: str, param_k: str, param_k_fix: float, plot_type='contourf', 
                        log_scale_x=False, log_scale_y=False):
    """
    Plots a 3D surface or 2D contour map of an error metric versus two parameters

    Args:
        exp_df (df): The df of experiments features (elapsed_time, error, ...)
        param_x (str): The parameter on the X-axis (exe: 'alpha')
        param_y (str): The parameter on the Y-axis (exe: 'eta')
        plot_type (str): 'surface3d' for a 3D plot, 
                        'contourf' for a 2D filled contour plot
        log_scale_x (bool): Whether to use a log scale for the X-axis
        log_scale_y (bool): Whether to use a log scale for the Y-axis
    """
    LABEL_ERROR = "FFT Error on BS"
    #// 1) Create Domain Mesh with respect to param_x and param_y:
    
    #///// Fixing a value for the parameter not choosen (param_k)
    pivot_df = exp_df[exp_df[param_k] == param_k_fix]
            
    #///// Create the X, Y grid coordinates and Z values error
    x_values = pivot_df[param_x].unique()   #first parameter values
    y_values = pivot_df[param_y].unique()  #second parameter values
    X, Y = np.meshgrid(x_values, y_values)

    # Computing Z
    pivot_df = pivot_df.pivot(index=param_y, columns=param_x, values="error")
    Z = pivot_df.values
    #// 2) Plotting Error:

    fig = plt.figure(figsize=(10, 7))
    
    if plot_type == 'surface3d':
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        
        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=10, label=LABEL_ERROR)
        
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_zlabel("FFT Error")
        
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')
        
    else:
        ax = fig.add_subplot(111)
        
        # Plot the filled contour
        cf = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
        
        # Add a color bar
        fig.colorbar(cf, label=LABEL_ERROR)
        
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')

    plt.title(f'"FFT error on BS Formula vs. {param_x} & {param_y}')
    plt.show()