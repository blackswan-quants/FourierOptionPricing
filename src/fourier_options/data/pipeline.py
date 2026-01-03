"""
End-to-End Data Pipeline.

Combines data fetching, filtering, and arbitrage checking into a single
pipeline for preparing option chain data for Heston model calibration.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

from .fetcher import OptionChainFetcher
from .filters import OptionDataFilter, FilterConfig
from .arbitrage import ArbitrageChecker


@dataclass
class PipelineConfig:
    """
    Configuration for the complete data pipeline.
    
    Attributes:
        filter_config: Configuration for data filters
        arbitrage_tolerance: Tolerance for arbitrage checks (in $)
        arbitrage_relative_tolerance: Relative tolerance for arbitrage checks
        price_column: Column to use for option prices ('mid', 'bid', 'ask', 'lastPrice')
        option_type: Type of options to include ('call', 'put', 'both')
        verbose: Whether to print progress and statistics
    """
    filter_config: Optional[FilterConfig] = None
    arbitrage_tolerance: float = 0.01
    arbitrage_relative_tolerance: float = 0.05
    price_column: str = 'mid'
    option_type: str = 'call'
    verbose: bool = True


class DataPipeline:
    """
    End-to-end pipeline for preparing option data for calibration.
    
    Combines:
    1. Data fetching (from API or CSV)
    2. Quality filtering (liquidity, maturity, moneyness, spread)
    3. Arbitrage checking (no-arbitrage conditions)
    
    The output is clean, calibration-ready option chain data.
    
    Example:
        >>> pipeline = DataPipeline(ticker='SPY')
        >>> clean_data = pipeline.run()
        >>> 
        >>> # Or from CSV
        >>> pipeline = DataPipeline()
        >>> clean_data = pipeline.run(csv_path='option_chain.csv')
    """
    
    def __init__(
        self,
        ticker: Optional[str] = None,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY'). Required for API fetching.
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual continuous dividend yield
            config: Pipeline configuration. Uses defaults if None.
        """
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.fetcher = None
        if ticker:
            self.fetcher = OptionChainFetcher(
                ticker=ticker,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield
            )
        
        filter_config = self.config.filter_config or FilterConfig()
        self.filter = OptionDataFilter(config=filter_config)
        self.arbitrage_checker = ArbitrageChecker(
            tolerance=self.config.arbitrage_tolerance,
            relative_tolerance=self.config.arbitrage_relative_tolerance
        )
        
        self._pipeline_stats: Dict[str, Any] = {}
        
    @property
    def pipeline_stats(self) -> Dict[str, Any]:
        """Statistics from the last pipeline run."""
        return self._pipeline_stats.copy()
    
    def run(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        expiration: Optional[str] = None,
        include_all_expirations: bool = True,
        raw_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Data is loaded from one of three sources (in priority order):
        1. raw_data: Pre-loaded DataFrame
        2. csv_path: CSV file path
        3. API: Fetch from yfinance using ticker
        
        Args:
            csv_path: Path to CSV file with option chain data
            expiration: Specific expiration date (for API fetching)
            include_all_expirations: Whether to fetch all expirations
            raw_data: Pre-loaded option chain DataFrame
            
        Returns:
            Clean, calibration-ready option chain DataFrame
        """
        self._pipeline_stats = {}
        
        # Step 1: Load data
        if self.config.verbose:
            print("\n" + "="*60)
            print("DATA PIPELINE")
            print("="*60)
        
        df = self._load_data(csv_path, expiration, include_all_expirations, raw_data)
        
        if df.empty:
            print("WARNING: No data loaded!")
            return df
        
        self._pipeline_stats['raw_count'] = len(df)
        
        if self.config.verbose:
            print(f"\n[1/3] Loaded {len(df)} options")
            if 'expiration' in df.columns:
                exps = df['expiration'].nunique()
                print(f"      Expirations: {exps}")
            if 'type' in df.columns:
                types = df['type'].value_counts().to_dict()
                print(f"      Types: {types}")
        
        # Step 2: Apply filters
        if self.config.verbose:
            print(f"\n[2/3] Applying data quality filters...")
        
        df_filtered = self.filter.apply_all_filters(df, verbose=self.config.verbose)
        self._pipeline_stats['filtered_count'] = len(df_filtered)
        self._pipeline_stats['filter_stats'] = self.filter.filter_stats
        
        # Step 3: Arbitrage checks
        if self.config.verbose:
            print(f"\n[3/3] Checking arbitrage conditions...")
        
        df_clean = self.arbitrage_checker.apply_all_checks(
            df_filtered,
            price_col=self.config.price_column,
            verbose=self.config.verbose
        )
        self._pipeline_stats['clean_count'] = len(df_clean)
        self._pipeline_stats['arbitrage_stats'] = self.arbitrage_checker.check_stats
        
        # Final summary
        if self.config.verbose:
            self._print_summary(df, df_clean)
        
        return df_clean
    
    def _load_data(
        self,
        csv_path: Optional[Union[str, Path]],
        expiration: Optional[str],
        include_all_expirations: bool,
        raw_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Load data from the specified source."""
        
        # Priority 1: Raw data
        if raw_data is not None:
            if self.config.verbose:
                print("Loading from provided DataFrame...")
            df = raw_data.copy()
            if self.config.option_type != 'both' and 'type' in df.columns:
                df = df[df['type'] == self.config.option_type]
            return df
        
        # Priority 2: CSV file
        if csv_path is not None:
            if self.config.verbose:
                print(f"Loading from CSV: {csv_path}")
            if self.fetcher is None:
                self.fetcher = OptionChainFetcher(
                    ticker='UNKNOWN',
                    risk_free_rate=self.risk_free_rate,
                    dividend_yield=self.dividend_yield
                )
            df = self.fetcher.fetch_from_csv(
                str(csv_path),
                option_type=self.config.option_type if self.config.option_type != 'both' else None
            )
            return df
        
        # Priority 3: API
        if self.fetcher is not None and self.ticker:
            if self.config.verbose:
                print(f"Fetching from API: {self.ticker}")
            df = self.fetcher.fetch_chain(
                expiration=expiration,
                option_type=self.config.option_type,
                include_all_expirations=include_all_expirations
            )
            return df
        
        raise ValueError(
            "No data source specified. Provide csv_path, raw_data, or initialize with ticker."
        )
    
    def _print_summary(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        """Print pipeline summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Raw options:       {len(df_raw):>6d}")
        print(f"After filtering:   {self._pipeline_stats['filtered_count']:>6d}")
        print(f"After arbitrage:   {len(df_clean):>6d}")
        print(f"Retention rate:    {len(df_clean)/len(df_raw)*100:>6.1f}%")
        print("-"*60)
        
        if len(df_clean) > 0:
            print("\nClean data summary:")
            print(f"  Expirations:     {df_clean['expiration'].nunique() if 'expiration' in df_clean.columns else 'N/A'}")
            print(f"  Moneyness range: [{df_clean['moneyness'].min():.3f}, {df_clean['moneyness'].max():.3f}]")
            print(f"  Maturity range:  [{df_clean['T'].min():.3f}, {df_clean['T'].max():.3f}] years")
            print(f"  Avg rel. spread: {df_clean['rel_spread'].mean():.1%}")
        
        print("="*60 + "\n")
    
    def get_calibration_data(
        self,
        df: pd.DataFrame,
        use_implied_vol: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract arrays needed for calibration from clean DataFrame.
        
        Args:
            df: Clean option chain DataFrame (output of run())
            use_implied_vol: If True, return implied vols instead of prices
            
        Returns:
            Dictionary with:
                - S: Spot price (scalar or array)
                - K: Strike prices
                - T: Times to maturity
                - r: Risk-free rates
                - q: Dividend yields
                - prices: Option prices (mid)
                - weights: Weighting for calibration
                - iv: Implied volatilities (if use_implied_vol=True)
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        result = {
            'S': df['S'].values,
            'K': df['strike'].values,
            'T': df['T'].values,
            'r': df['r'].values if 'r' in df.columns else np.full(len(df), self.risk_free_rate),
            'q': df['q'].values if 'q' in df.columns else np.full(len(df), self.dividend_yield),
            'prices': df[self.config.price_column].values,
            'moneyness': df['moneyness'].values,
            'weights': df['weight'].values if 'weight' in df.columns else np.ones(len(df)),
        }
        
        if use_implied_vol and 'impliedVolatility' in df.columns:
            result['iv'] = df['impliedVolatility'].values
        
        # Add option type indicator
        if 'type' in df.columns:
            result['is_call'] = (df['type'] == 'call').values
        else:
            result['is_call'] = np.ones(len(df), dtype=bool)
        
        return result
    
    def prepare_for_heston(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepare data specifically for Heston model calibration.
        
        Groups options by maturity and returns structured data
        suitable for FFT-based calibration.
        
        Args:
            df: Clean option chain DataFrame
            
        Returns:
            Dictionary with:
                - S0: Spot price
                - r: Risk-free rate
                - q: Dividend yield
                - maturities: Unique maturities
                - data_by_T: Dict mapping T -> {K, prices, weights}
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        S0 = df['S'].iloc[0]
        r = df['r'].iloc[0] if 'r' in df.columns else self.risk_free_rate
        q = df['q'].iloc[0] if 'q' in df.columns else self.dividend_yield
        
        maturities = sorted(df['T'].unique())
        data_by_T = {}
        
        for T in maturities:
            mask = df['T'] == T
            T_data = df[mask].sort_values('strike')
            
            data_by_T[T] = {
                'K': T_data['strike'].values,
                'prices': T_data[self.config.price_column].values,
                'weights': T_data['weight'].values if 'weight' in T_data.columns else np.ones(mask.sum()),
                'moneyness': T_data['moneyness'].values,
                'n_options': mask.sum()
            }
        
        return {
            'S0': S0,
            'r': r,
            'q': q,
            'maturities': maturities,
            'data_by_T': data_by_T,
            'total_options': len(df)
        }


def quick_clean(
    csv_path: str,
    ticker: Optional[str] = None,
    min_volume: int = 10,
    min_open_interest: int = 100,
    min_days: float = 7.0,
    max_days: float = 365.0,
    min_moneyness: float = 0.80,
    max_moneyness: float = 1.20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick helper function to clean option data with common settings.
    
    Args:
        csv_path: Path to option chain CSV
        ticker: Ticker symbol (for display only)
        min_volume: Minimum trading volume
        min_open_interest: Minimum open interest
        min_days: Minimum days to expiration
        max_days: Maximum days to expiration
        min_moneyness: Minimum K/S ratio
        max_moneyness: Maximum K/S ratio
        verbose: Print statistics
        
    Returns:
        Clean option chain DataFrame
    """
    filter_config = FilterConfig(
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        min_days_to_exp=min_days,
        max_days_to_exp=max_days,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness
    )
    
    pipeline_config = PipelineConfig(
        filter_config=filter_config,
        verbose=verbose
    )
    
    pipeline = DataPipeline(ticker=ticker, config=pipeline_config)
    return pipeline.run(csv_path=csv_path)
