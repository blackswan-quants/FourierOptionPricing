"""
Example: Advanced Data Pipeline for Heston Calibration (Task 1)

This script demonstrates the complete data pipeline for preparing
option chain data for Heston model calibration.

The pipeline implements:
1. Data Fetching & Loading (from CSV or yfinance API)
2. Liquidity Filters (volume, open interest)
3. Moneyness & Maturity Filters (short maturities, deep OTM/ITM)
4. Arbitrage Checks (no-arbitrage conditions)
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
from fourier_options.data import DataPipeline, OptionDataFilter, ArbitrageChecker
from fourier_options.data.filters import FilterConfig
from fourier_options.data.pipeline import PipelineConfig


def main():
    """Run the data pipeline example."""
    
    print("="*70)
    print("TASK 1: ADVANCED DATA PIPELINE FOR HESTON CALIBRATION")
    print("="*70)
    
    # Path to the option chain data
    data_path = Path(__file__).parent.parent / "option_chain_SPY.csv"
    
    # =========================================================================
    # OPTION 1: Quick Clean (Simple API)
    # =========================================================================
    print("\n" + "-"*70)
    print("OPTION 1: Using quick_clean() helper function")
    print("-"*70)
    
    from fourier_options.data.pipeline import quick_clean
    
    df_quick = quick_clean(
        csv_path=str(data_path),
        ticker="SPY",
        min_volume=50,
        min_open_interest=200,
        min_days=7.0,
        max_days=180.0,
        min_moneyness=0.85,
        max_moneyness=1.15,
        verbose=True
    )
    
    print(f"\nQuick clean result: {len(df_quick)} options ready for calibration")
    
    # =========================================================================
    # OPTION 2: Full Pipeline with Custom Configuration
    # =========================================================================
    print("\n" + "-"*70)
    print("OPTION 2: Using full DataPipeline with custom configuration")
    print("-"*70)
    
    # Configure filters
    filter_config = FilterConfig(
        min_volume=100,              # At least 100 contracts traded
        min_open_interest=500,       # At least 500 open interest
        min_days_to_exp=14.0,        # At least 2 weeks to expiry
        max_days_to_exp=120.0,       # Max 4 months
        min_moneyness=0.90,          # K/S >= 0.90
        max_moneyness=1.10,          # K/S <= 1.10
        max_relative_spread=0.20,    # Max 20% bid-ask spread
        min_bid=0.10,                # Min $0.10 bid
        remove_zero_bid=True,        # Remove zero bids
        remove_itm=False             # Keep ITM options
    )
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        filter_config=filter_config,
        arbitrage_tolerance=0.02,
        arbitrage_relative_tolerance=0.05,
        price_column='mid',
        option_type='call',
        verbose=True
    )
    
    # Create and run pipeline
    pipeline = DataPipeline(
        ticker='SPY',
        risk_free_rate=0.05,
        dividend_yield=0.01,
        config=pipeline_config
    )
    
    df_clean = pipeline.run(csv_path=str(data_path))
    
    # =========================================================================
    # Analyze the clean data
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS OF CLEAN DATA")
    print("-"*70)
    
    if len(df_clean) > 0:
        # Distribution by expiration
        print("\nOptions by expiration:")
        exp_summary = df_clean.groupby('expiration').agg({
            'strike': 'count',
            'moneyness': ['min', 'max'],
            'rel_spread': 'mean',
            'volume': 'sum'
        }).round(4)
        print(exp_summary)
        
        # Moneyness distribution
        print("\nMoneyness distribution:")
        moneyness_bins = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        df_clean['moneyness_bucket'] = pd.cut(df_clean['moneyness'], bins=moneyness_bins)
        print(df_clean['moneyness_bucket'].value_counts().sort_index())
        
        # Data quality metrics
        print("\nData quality metrics:")
        print(f"  Average relative spread: {df_clean['rel_spread'].mean():.2%}")
        print(f"  Average volume: {df_clean['volume'].mean():.0f}")
        print(f"  Average open interest: {df_clean['openInterest'].mean():.0f}")
        print(f"  Average implied volatility: {df_clean['impliedVolatility'].mean():.2%}")
    
    # =========================================================================
    # Prepare data for Heston calibration
    # =========================================================================
    print("\n" + "-"*70)
    print("DATA PREPARED FOR HESTON CALIBRATION")
    print("-"*70)
    
    heston_data = pipeline.prepare_for_heston(df_clean)
    
    print(f"\nSpot price (S0): ${heston_data['S0']:.2f}")
    print(f"Risk-free rate: {heston_data['r']:.2%}")
    print(f"Dividend yield: {heston_data['q']:.2%}")
    print(f"Total options: {heston_data['total_options']}")
    print(f"\nMaturities (in years):")
    
    for T in heston_data['maturities']:
        T_data = heston_data['data_by_T'][T]
        print(f"  T={T:.4f} ({T*365:.0f} days): {T_data['n_options']} options, "
              f"K range [{T_data['K'].min():.2f}, {T_data['K'].max():.2f}]")
    
    # =========================================================================
    # Extract calibration arrays
    # =========================================================================
    calib_data = pipeline.get_calibration_data(df_clean, use_implied_vol=True)
    
    print("\nCalibration arrays ready:")
    print(f"  Strikes (K): shape {calib_data['K'].shape}")
    print(f"  Maturities (T): shape {calib_data['T'].shape}")
    print(f"  Prices: shape {calib_data['prices'].shape}")
    print(f"  Weights: shape {calib_data['weights'].shape}")
    
    # =========================================================================
    # Save clean data
    # =========================================================================
    output_path = Path(__file__).parent.parent / "data" / "spy_clean.csv"
    output_path.parent.mkdir(exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\nClean data saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Pipeline complete! Data is ready for Heston model calibration.")
    print("="*70)
    
    return df_clean, heston_data


def demonstrate_individual_filters():
    """Demonstrate using filters individually."""
    
    print("\n" + "="*70)
    print("INDIVIDUAL FILTER DEMONSTRATION")
    print("="*70)
    
    data_path = Path(__file__).parent.parent / "option_chain_SPY.csv"
    df = pd.read_csv(data_path)
    df = df[df['type'] == 'call']  # Focus on calls
    
    print(f"\nInitial data: {len(df)} options")
    
    # Create filter
    data_filter = OptionDataFilter()
    
    # Apply filters step by step
    print("\nApplying filters step-by-step:")
    
    df1 = data_filter.filter_prices(df)
    print(f"  After price filter: {len(df1)} (removed {len(df) - len(df1)})")
    
    df2 = data_filter.filter_liquidity(df1)
    print(f"  After liquidity filter: {len(df2)} (removed {len(df1) - len(df2)})")
    
    df3 = data_filter.filter_maturity(df2)
    print(f"  After maturity filter: {len(df3)} (removed {len(df2) - len(df3)})")
    
    df4 = data_filter.filter_moneyness(df3)
    print(f"  After moneyness filter: {len(df4)} (removed {len(df3) - len(df4)})")
    
    df5 = data_filter.filter_spread(df4)
    print(f"  After spread filter: {len(df5)} (removed {len(df4) - len(df5)})")
    
    # Arbitrage checks
    print("\nApplying arbitrage checks:")
    arb_checker = ArbitrageChecker()
    
    df6 = arb_checker.apply_all_checks(df5, price_col='mid')
    print(f"  After arbitrage checks: {len(df6)} (removed {len(df5) - len(df6)})")
    
    # Show violations
    violations_df = arb_checker.get_violations_df()
    if len(violations_df) > 0:
        print(f"\nArbitrage violations detected: {len(violations_df)}")
        print(violations_df['violation_type'].value_counts())


if __name__ == "__main__":
    # Run main pipeline
    df_clean, heston_data = main()
    
    # Demonstrate individual filters
    demonstrate_individual_filters()
