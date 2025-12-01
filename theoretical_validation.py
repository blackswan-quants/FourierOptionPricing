import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from src import fft_pricer, grid_module
from characteristic_functions import cf_bs

#functions
def select_random_parameters_fft(N_grid, eta_grid, alpha_grid):
    """
    Selects one random value from each grid independently.
    Returns: (n_val, eta_val, alpha_val)
    """
    n_val = np.random.choice(N_grid)
    eta_val = np.random.choice(eta_grid)
    alpha_val = np.random.choice(alpha_grid)
    
    return n_val, eta_val, alpha_val

def select_random_parameters_call(T, vol):
    """
    Selects one random value from each grid independently.
    Returns: (n_val, eta_val, alpha_val)
    """
    t_val = np.random.choice(T.flatten())
    vol_val = np.random.choice(vol.flatten())
    
    return t_val, vol_val

#----------------------

#Plot settings
plt.style.use('seaborn-v0_8-darkgrid') 
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'}) # Serif fonts look more academic

#----------------------
# Generate Validation Data:
#// The aim is to validate theoretical monotonicity of the price on some parameters
S0 = 100 #fixed
r = 0.05
T = np.arange(0.5, 1.5, 0.1)
vol = np.arange(0.15, 0.6, 0.05)

params = {"S0": S0,
    "r": r,
    "T": 0.5,
    "sigma": 0.15}

#RMK: the pricer gives in output a list of prices, according to a list of K strike

#// Generate pricer parameters
N = grid_module.generate_n_grid(11, 13)
eta = grid_module.generate_eta_grid(0.2, 0.4)
alpha = grid_module.generate_alpha_grid(1.2, 1.5, 10)

#// IDEA: I will choose randomly a triplet of fft parameters and a couple of fixed Call parameters each timne
#//       This is the best choice because it let me to validate theoretically the engine w/o having to
#//       look at an unmanagable number of plots and running a non sense number of time the same function,
#//       but still mantaining a certain confidence on the results

no_parity = 0 #heck how many times Put - Call parity does not hold
total_runs = 0

for i in range(5):
    n_value, eta_value, alpha_value = select_random_parameters_fft(N, eta, alpha)
    t_value, vol_value = select_random_parameters_call(T, vol)
    
    params["T"] = t_value
    params["sigma"] = vol_value

    #FIxed T, Fixed vol
    k_values, k_results = fft_pricer.fft_pricer(cf_bs, params, alpha_value, n_value, eta_value)
    valid_indices = (k_values > S0 * 0.75) & (k_values < S0 * 1.25)
    k_values, k_results = (k_values[valid_indices], k_results[valid_indices])

    #Put_Call check:
    _, put_prices = fft_pricer.fft_pricer(cf_bs, params, -alpha_value, n_value, eta_value)  #negative alpha for put
    put_prices = put_prices[valid_indices]
    check = k_results - put_prices - S0 + np.exp(-r*t_value)*k_values #must be near 0
    no_parity += np.sum(np.abs(check) > 1e-1)
    
    total_runs += len(k_values)

    #K fixed, T fixed
    vol_results = []
    for volat in vol:
        total_runs += 1

        params["sigma"] = volat

        k_temp, call_price = fft_pricer.fft_pricer(cf_bs, params, alpha_value, n_value, eta_value)

        #fix K = 100
        call_at_100 = np.interp(100, k_temp, call_price)   #choosing the first price corresponding to the first fixed K strike
        vol_results.append(call_at_100)

        #Put_Call check:
        _, put_prices = fft_pricer.fft_pricer(cf_bs, params, -alpha_value, n_value, eta_value)
        put_at_100 = np.interp(100, k_temp, put_prices) 
        check = call_at_100 - put_at_100 - S0 + np.exp(-r*t_value)*100 #must be near 0
        no_parity += np.sum(np.abs(check) > 1e-1)

    #K fixed, vol fixed
    t_results = []
    for t in T:
        total_runs += 1
        params["T"] = t
        params["sigma"] = vol_value
        k_temp, call_price = fft_pricer.fft_pricer(cf_bs, params, alpha_value, n_value, eta_value)

        #fix K = 100
        call_at_100 = np.interp(100, k_temp, call_price)   #choosing the first price corresponding to the first fixed K strike
        t_results.append(call_at_100)

        #Put_Call check:
        _, put_prices = fft_pricer.fft_pricer(cf_bs, params, -alpha_value, n_value, eta_value)
        put_at_100 = np.interp(100, k_temp, put_prices) 
        check = call_at_100 - put_at_100 - S0 + np.exp(-r*t)*100 #must be near 0
        no_parity += np.sum(np.abs(check) > 1e-1)

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'FFT Option Pricing Monotonicity Valiation: $T={t_value:.2f}$ years, $\sigma={vol_value:.2f}$', fontsize=16, fontweight='bold')
    # Plot 1: Filtered Strike Grid
    axis[0].plot(k_values, k_results, color='#1f77b4', linewidth=2.5, label='Call Price')
    axis[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    axis[0].set_title(r'Price vs Strike ($K$)', fontsize=12, weight='bold')
    axis[0].set_xlabel(r'Strike ($K$)')
    axis[0].set_ylabel('Option Price ($)')
    axis[0].legend(loc='upper right', frameon=True)
        
    # Plot 2: Volatility Sensitivity
    axis[1].plot(vol, vol_results, color='#ff7f0e', linewidth=2.5, label='Call Price ATM')
    axis[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    axis[1].set_title(r'Sensitivity to Volatility ($\sigma$)', fontsize=12, weight='bold')
    axis[1].set_xlabel(r'Volatility $\sigma$')
    axis[1].set_ylabel('ATM Price ($)')
    axis[1].legend()

    # Plot 3: Time Sensitivity
    axis[2].plot(T, t_results, color='#2ca02c', linewidth=2.5, label='Call Price ATM')
    axis[2].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    axis[2].set_title(r'Sensitivity to Maturity ($T$)', fontsize=12, weight='bold')
    axis[2].set_xlabel(r'Time to Maturity $T$ (Years)')
    axis[2].set_ylabel('ATM Price ($)')
    axis[2].legend()

    param_text = (
    f"Simulation Parameters:\n"
    f"----------------------\n"
    f"$\\alpha = {alpha_value: .4f}$\n"
    f"$\\eta = {eta_value: .4f}$\n"
    f"$N = {n_value}$"
    )

    # Place text box in the first plot (or wherever fits)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    axis[0].text(0.05, 0.05, param_text, transform=axis[0].transAxes, 
                verticalalignment='bottom', bbox=props, fontsize=9)
    plt.tight_layout()
    plt.show()


print(f"Put - Call Parity did not hold {no_parity/total_runs} percentage of times")
print(no_parity, total_runs)
