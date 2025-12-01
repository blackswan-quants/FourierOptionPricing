import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import time

from fft_pricer import fft_pricer
from characteristic_functions import cf_heston

def load_data(filename="data/spy_options_CLEANED_20251201.parquet"):
    return pd.read_parquet(filename)

def heston_loss_function(params_array, market_data):
    """
    Loss function con pesi (Errore Relativo) e penalità.
    """
    kappa, theta, sigma_v, rho, v0 = params_array
    
    # --- 1. Penalità Soft ---
    penalty = 0.0
    
    # Condizione di Feller: 2*kappa*theta > sigma_v^2
    # Se violata, aggiungiamo una penalità enorme
    if 2 * kappa * theta < sigma_v**2:
        penalty += 100.0 

    # Vincoli di dominio "soft" per aiutare l'ottimizzatore se esce dai bounds
    if sigma_v < 0.01 or v0 < 0.0:
        return 1e9

    model_params = {
        "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": v0
    }

    total_weighted_sq_error = 0.0
    n_obs = 0

    grouped = market_data.groupby('T')

    for T, group in grouped:
        r = group['r'].iloc[0]
        S0 = group['S0'].iloc[0]
        model_params["T"] = T
        model_params["r"] = r

        try:
            # N=2**12 (4096) per matchare la precisione del generatore
            k_grid, price_grid = fft_pricer(cf_heston, model_params, S0=S0, N=2**12)
        except Exception:
            return 1e9

        market_strikes = group['K'].values
        market_prices = group['Market_Price'].values
        
        model_prices_interp = np.interp(market_strikes, k_grid, price_grid)
        
        # --- 2. Errore Relativo (Pesi) ---
        # Diamo più importanza alle opzioni OTM (prezzo basso) che altrimenti verrebbero ignorate
        # Peso = 1 / prezzo_mercato
        weights = 1.0 / (market_prices + 0.01) # +0.01 per evitare div by zero
        
        errors = (market_prices - model_prices_interp) * weights
        
        total_weighted_sq_error += np.sum(errors**2)
        n_obs += len(errors)

    rmse = np.sqrt(total_weighted_sq_error / n_obs)
    return rmse + penalty

def calibrate_heston(data):
    print("Inizio calibrazione (Differential Evolution)...")
    print("Potrebbe richiedere 1-2 minuti...")
    start_time = time.time()

    # Bounds (Vincoli)
    bounds = [
        (0.5, 5.0),   # kappa
        (0.01, 0.2),  # theta
        (0.01, 1.0),  # sigma_v
        (-0.9, 0.0),  # rho (solitamente negativo per equity)
        (0.01, 0.2)   # v0
    ]

    # --- USARE DIFFERENTIAL EVOLUTION (Globale) ---
    # Molto più robusto per Heston rispetto a minimize()
    result = differential_evolution(
        func=heston_loss_function,
        bounds=bounds,
        args=(data,),
        strategy='best1bin', # Strategia standard
        maxiter=40,          # Numero di generazioni (aumenta per più precisione)
        popsize=20,          # Popolazione (aumenta per esplorare meglio)
        tol=0.001,
        disp=True,           # Mostra progresso
        workers=-1           # Usa tutti i core della CPU (Parallelizzazione)
    )

    # (Opzionale) Raffinamento finale con L-BFGS-B partendo dal risultato globale
    print("\nRaffinamento locale...")
    result = minimize(
        fun=heston_loss_function,
        x0=result.x,
        args=(data,),
        method='L-BFGS-B',
        bounds=bounds
    )

    end_time = time.time()
    print(f"\nCalibrazione completata in {end_time - start_time:.2f} secondi.")
    return result

if __name__ == "__main__":
    try:
        df = load_data("data/spy_options_CLEANED_20251201.parquet") # Usa il file pulito!
        print(f"Dati caricati: {len(df)} opzioni.")
    except:
        exit()

    best_result = None
    best_rmse = float('inf')
    
    # Eseguiamo 5 tentativi
    N_RUNS = 5
    print(f"\n--- AVVIO MULTI-START ({N_RUNS} runs) ---")
    
    for i in range(N_RUNS):
        print(f"\nRun {i+1}/{N_RUNS}...")
        
        # Lancia la calibrazione
        res = calibrate_heston(df)
        
        print(f" -> RMSE: {res.fun:.6f}")
        
        # Se è il migliore finora, salvatelo
        if res.fun < best_rmse:
            best_rmse = res.fun
            best_result = res
            print(" -> Nuovo Best Found!")

    # --- RISULTATO FINALE (IL MIGLIORE DEI 5) ---
    print("\n" + "="*40)
    print(f"VINCITORE ASSOLUTO (RMSE: {best_rmse:.6f})")
    print("="*40)
    
    param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
    final_params = dict(zip(param_names, best_result.x))
    
    for p, val in final_params.items():
        print(f"{p:<10}: {val:.6f}")
        
    # Salva i parametri su file se vuoi usarli dopo
    # pd.Series(final_params).to_json("calibrated_params.json")