import torch
import torch.fft
import numpy as np
import time

# Verifica disponibilità GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def cf_heston_torch(u, params):
    """
    Funzione Caratteristica Heston riscritta in PyTorch.
    Accetta tensori batch per i parametri.
    """
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho = params['rho']
    v0 = params['v0']
    T = params['T']
    r = params['r']
    S0 = params['S0']
    
    # Gestione numeri complessi in Torch
    i = 1j
    
    # Formule standard Heston vettorizzate
    d = torch.sqrt((rho * sigma_v * u * i - kappa)**2 + sigma_v**2 * (u * i + u**2))
    g = (kappa - rho * sigma_v * u * i - d) / (kappa - rho * sigma_v * u * i + d)
    
    term1 = torch.exp(r * T * u * i)
    term2 = (S0**(u * i))
    term3 = ((1 - g * torch.exp(-d * T)) / (1 - g))**(-2 * kappa * theta / sigma_v**2)
    term4 = torch.exp((kappa * theta / sigma_v**2) * (kappa - rho * sigma_v * u * i - d) * T)
    term5 = torch.exp((v0 / sigma_v**2) * (kappa - rho * sigma_v * u * i - d) * (1 - torch.exp(-d * T)) / (1 - g * torch.exp(-d * T)))
    
    return term1 * term2 * term3 * term4 * term5

def fft_pricer_torch(cf_function, params, alpha=1.5, N=4096, eta=0.25):
    """
    Pricer FFT ottimizzato per GPU.
    Calcola prezzi per N strike simultaneamente.
    """
    # Grid Setup (tensori su GPU)
    N_tens = torch.tensor(N, device=device)
    k_range = torch.arange(N, device=device)
    
    v = k_range * eta
    u = v - (alpha + 1) * 1j
    
    # Chiamata alla CF (che deve supportare tensori)
    phi_vals = cf_function(u, params)
    
    # Carr-Madan Math
    lambd = (2 * torch.pi) / (N * eta)
    b = 0.5 * N * lambd
    ku = -b + lambd * k_range
    
    denom = (alpha + alpha**2 - v**2) + 1j * (2 * alpha + 1) * v
    # Evitiamo divisione per zero a v=0
    denom[0] = 1.0 
    
    damping = torch.exp(-params['r'] * params['T'])
    psi = damping * phi_vals / denom
    
    # Pesi Simpson (opzionale, qui rettangolare per velocità o Simpson vettorizzato)
    # Per semplicità usiamo pesi rettangolari corretti per FFT
    w = torch.ones(N, device=device) * eta
    w[0] *= 0.5
    w[-1] *= 0.5
    
    # FFT
    fft_input = torch.exp(1j * b * v) * psi * w
    fft_out = torch.fft.fft(fft_input)
    
    call_prices = torch.exp(-alpha * ku) * fft_out.real / torch.pi
    strikes = torch.exp(ku)
    
    return strikes, call_prices

# --- LOSS FUNCTION (Batch GPU) ---
def heston_loss_gpu(calibration_params, market_data_tensors, fixed_params):
    """
    Calcola la loss su GPU.
    calibration_params: tensore [5] con (kappa, theta, sigma_v, rho, v0)
    """
    # Unpack parametri ottimizzabili
    # Usiamo torch.clamp per imporre vincoli "soft" durante i calcoli (es. sigma > 0)
    params = {
        'kappa':   torch.abs(calibration_params[0]),
        'theta':   torch.abs(calibration_params[1]),
        'sigma_v': torch.abs(calibration_params[2]),
        'rho':     torch.tanh(calibration_params[3]), # Forza tra -1 e 1
        'v0':      torch.abs(calibration_params[4]),
        
        # Parametri fissi (vettorizzati o scalari)
        'S0': fixed_params['S0'],
        'r':  fixed_params['r'],
        'T':  fixed_params['T']
    }
    
    # 1. Calcolo Modello
    model_strikes, model_prices = fft_pricer_torch(cf_heston_torch, params)
    
    # 2. Interpolazione su GPU (per trovare i prezzi agli strike di mercato)
    # Purtroppo torch non ha un interp1d semplice come numpy.
    # Per velocità, spesso si usa una griglia fissa o un'implementazione custom.
    # Qui usiamo un trucco: assumiamo che market_data sia già allineato o usiamo un Nearest Neighbor semplice
    # PER ORA: Semplificazione -> calcoliamo l'errore solo sugli indici più vicini (o implementiamo interp)
    
    # Implementazione Interpolazione Lineare Torch-friendly
    # Cerchiamo gli indici per ogni market_strike
    market_strikes = market_data_tensors['K']
    market_prices_true = market_data_tensors['Price']
    
    # Cerca indici (searchsorted richiede input ordinati)
    # model_strikes è ordinato per definizione FFT
    idx = torch.searchsorted(model_strikes, market_strikes)
    idx = torch.clamp(idx, 1, len(model_strikes)-1)
    
    x0 = model_strikes[idx-1]
    x1 = model_strikes[idx]
    y0 = model_prices[idx-1]
    y1 = model_prices[idx]
    
    # Formula interpolazione lineare
    slope = (y1 - y0) / (x1 - x0)
    model_prices_interp = y0 + slope * (market_strikes - x0)
    
    # 3. Calcolo MSE
    error = model_prices_interp - market_prices_true
    mse = torch.mean(error**2)
    
    return mse