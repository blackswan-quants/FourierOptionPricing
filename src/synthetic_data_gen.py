import numpy as np
import pandas as pd

# TODO: Importa qui il tuo pricer modulare e la funzione caratteristica Heston
from fft_pricer import fft_pricer
from characteristic_functions import cf_heston

def generate_heston_synthetic_data():
    """
    Genera un dataset sintetico per testare la calibrazione.
    Usa parametri noti ('Ground Truth') per calcolare i prezzi, 
    così possiamo verificare se l'ottimizzatore riesce a ritrovarli.
    """
    
    # --- 1. CONFIGURAZIONE DELLO SCENARIO DI MERCATO ---
    S0 = 100.0       # Prezzo Spot iniziale (es. 100$)
    r = 0.03         # Tasso risk-free costante (3%)
    
    # --- 2. PARAMETRI HESTON "VERI" (Ground Truth) ---
    # Questi sono i valori che il tuo codice di calibrazione dovrà "scoprire".
    
    true_params = {
        # v0: Varianza istantanea iniziale. 
        # È il quadrato della volatilità corrente. Se vol=20%, v0=0.04.
        'v0': 0.04,     
        
        # kappa: Velocità di Mean Reversion.
        # Quanto velocemente la volatilità torna alla sua media (theta).
        # > 2.0 = veloce, < 0.5 = lento (persistente).
        'kappa': 2.0,   
        
        # theta: Varianza di lungo periodo (Long-run Variance).
        # Il livello verso cui la varianza tende all'infinito.
        'theta': 0.04,  
        
        # sigma_v: "Vol of Vol" (Volatilità della varianza).
        # Determina quanto è "grassa" la coda della distribuzione (kurtosis).
        # Valori alti aumentano la curvatura del "smile" di volatilità.
        'sigma_v': 0.3, 
        
        # rho: Correlazione (tra prezzo asset e volatilità).
        # Determina l'asimmetria (SKEW) del smile.
        # Per l'Equity è quasi sempre negativo (es. -0.7): se il mercato scende, la paura (vol) sale.
        'rho': -0.7     
    }

    # CHECK DI STABILITÀ (Condizione di Feller)
    # Serve per assicurare che la varianza rimanga sempre positiva matematicamente.
    # Formula: 2 * kappa * theta > sigma_v^2
    feller_stat = 2 * true_params['kappa'] * true_params['theta'] - true_params['sigma_v']**2
    if feller_stat > 0:
        print(f"Condizione di Feller soddisfatta (Stat: {feller_stat:.4f}). Processo stabile.")
    else:
        print(f"ATTENZIONE: Condizione di Feller violata! La varianza potrebbe diventare negativa.")

    # --- 3. CREAZIONE DELLA GRIGLIA ---
    # Creiamo una superficie di opzioni realistica
    
    # Definisci una lista di Maturities (T) in anni. 
    # Es: [0.1, 0.5, 1.0] (che corrispondono a circa 1 mese, 6 mesi, 1 anno)
    maturities = np.linspace(0.1,5,40)

    # Definisci una lista di Strike (K).
    # Suggerimento: usa np.linspace per generare strike attorno a S0.
    # Es: da 80% a 120% di S0.
    strikes = np.linspace(0.6*S0, 1.4*S0, 40)

    synthetic_data = []
    
    print("Inizio generazione prezzi...")

    # --- 4. CORE LOOP ---
    for T in maturities:
        # Aggiorna i parametri con T e r correnti
        true_params["T"] = T
        true_params["r"] = r
        
        # -----------------------------------------------------------
        # Calcola il "Prezzo Modello" usando il tuo FFT Pricer
        # -----------------------------------------------------------
        # Chiamata al pricer con S0
        # Nota: fft_pricer restituisce una griglia di prezzi per molti strike
        strikes_grid, prices_grid = fft_pricer(cf_heston, true_params, S0=S0)
        
        for K in strikes:
            # Interpolazione per trovare il prezzo allo strike K specifico
            true_model_price = np.interp(K, strikes_grid, prices_grid)
            
            # -----------------------------------------------------------
            # Aggiungi Rumore di Mercato
            # -----------------------------------------------------------
            # Aggiungi un piccolo errore casuale (es. distribuzione normale)
            # per simulare lo spread bid-ask.
            # Usiamo un rumore proporzionale al prezzo o fisso?
            # Facciamo un mix: rumore relativo (es. 1%) + rumore assoluto (es. 0.01)
            
            noise = np.random.normal(0, 0.005 * true_model_price + 0.005)
            
            market_price = true_model_price + noise
            
            # Salvataggio riga
            # Scartiamo prezzi negativi o nulli (impossibili per Call)
            if market_price > 0.01:
                synthetic_data.append({
                    'S0': S0,
                    'K': K,
                    'T': T,
                    'r': r,
                    'Market_Price': market_price,
                    'True_Price': true_model_price # Utile per debuggare quanto rumore hai aggiunto
                })

    # --- 5. SALVATAGGIO ---
    df = pd.DataFrame(synthetic_data)
    
    filename = "data/heston_synthetic_data.parquet"
    df.to_parquet(filename)
    
    print(f"\nGenerazione completata.")
    print(f"Salvati {len(df)} prezzi in {filename}")
    print("\nAnteprima dati:")
    print(df.head())

if __name__ == "__main__":
    generate_heston_synthetic_data()