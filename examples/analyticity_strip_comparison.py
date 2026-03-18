import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can run this from the examples directory or root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Only importing style/utils if needed, but the user provided standalone math functions.
# We will keep the user's definitions to ensure the specific mathematical behavior 
# (MGF explosion) is demonstrated without interference from library stability fixes.

# Standard Layout
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# --- DEFINIZIONI RAPIDE PER IL PLOT ---
def cf_bs_log(u, params):
    """Log of Black-Scholes Characteristic Function"""
    S0 = params['S0']
    r = params['r']
    T = params['T']
    sigma = params['sigma']
    # Drift risk neutral
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    return 1j * u * mu - 0.5 * u**2 * sigma**2 * T

def cf_heston_log(u, params):
    """Log of Heston Characteristic Function"""
    S0 = params['S0']
    r = params['r']
    T = params['T']
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v'] # xi in notation
    rho = params['rho']
    v0 = params['v0']
    
    # Heston characteristic function components
    # d parameter
    d = np.sqrt((1j * u * rho * sigma_v - kappa)**2 + sigma_v**2 * (1j * u + u**2))
    
    # g parameter
    g_numer = kappa - 1j * u * rho * sigma_v - d
    g_denom = kappa - 1j * u * rho * sigma_v + d
    g = g_numer / g_denom
    
    # C and D terms
    term1 = (kappa * theta / sigma_v**2) * ((g_numer * T) - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    term2 = (v0 / sigma_v**2) * g_numer * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    
    return term1 + term2 + 1j * u * (np.log(S0) + r * T)

# --- ANALISI ---
def main():
    print("Generating Strip of Analyticity Comparison...")

    # Parametri Comparabili
    # BS ha vol = 20%
    # Heston ha vol stocastica che oscilla intorno al 20% ma con code pesanti (rho negativo, vol-of-vol alta)
    params_bs = {'S0': 100, 'r': 0.0, 'T': 1.0, 'sigma': 0.2}
    params_heston = {'S0': 100, 'r': 0.0, 'T': 1.0, 'kappa': 2.0, 'theta': 0.04, 'sigma_v': 1.0, 'rho': -0.7, 'v0': 0.04}
    # sigma_v = 1.0 (molto alto) serve a esasperare le code per far vedere l'esplosione presto

    # Valutiamo la MGF: M(k) = E[e^{k s}] = phi(-i * k)
    # k rappresenta (alpha + 1)
    k_values = np.linspace(0, 45, 500) 
    
    mgf_bs = []
    mgf_heston = []
    
    for k in k_values:
        # Argomento puramente immaginario per ottenere la MGF reale
        u_eval = -1j * k 
        
        # BS
        val_bs = np.exp(cf_bs_log(u_eval, params_bs))
        mgf_bs.append(np.real(val_bs))
        
        # Heston
        # Attenzione: Heston esploderà, usiamo try-except o clipping
        try:
            val_h = np.exp(cf_heston_log(u_eval, params_heston))
            if np.isnan(val_h) or np.isinf(np.real(val_h)) or np.real(val_h) > 1e10:
                mgf_heston.append(np.nan)
            else:
                mgf_heston.append(np.real(val_h))
        except:
            mgf_heston.append(np.nan)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot BS (Scala Log per vedere meglio)
    plt.plot(k_values, np.log(mgf_bs), 's-', label='Black-Scholes (Gaussian)', markevery=20, color='orange', linewidth=2)
    
    # Plot Heston
    plt.plot(k_values, np.log(mgf_heston), 'o-', label='Heston (Heavy Tails)', markevery=20, color='#1f77b4', linewidth=2)
    
    # Trova l'asintoto Heston (dove diventa NaN)
    # L'ultimo valore valido prima del NaN
    idx_nan = np.where(np.isnan(mgf_heston))[0]
    if len(idx_nan) > 0:
        k_crit = k_values[idx_nan[0]-1]
        plt.axvline(x=k_crit, color='red', linestyle='--', label=f'Heston Critical Moment ($k \\approx {k_crit:.1f}$)')
        plt.text(k_crit + 0.5, 5, 'Analyticity Strip Limit!', color='red', rotation=90, fontweight='bold')
    
    plt.title('Domain of Analyticity Comparison: BS vs Heston\n(Log-MGF vs Moment Order k)')
    plt.xlabel('Moment Order $k = \\alpha + 1$')
    plt.ylabel('log($\\mathbb{E}[e^{k S_T}]$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save as SVG (Vector Graphics)
    output_file = 'strip_comparison.svg'
    plt.tight_layout()
    plt.savefig(output_file, format='svg')
    print(f"Plot saved to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    main()
