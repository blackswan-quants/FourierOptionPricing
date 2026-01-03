"""
Option Chain Data Fetcher.

Fetches option chain data from yfinance API for calibration purposes.
Supports fetching for multiple tickers and expirations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Union
import warnings


class OptionChainFetcher:
    """
    Fetches option chain data from yfinance API.
    
    Provides methods to download and structure option chain data including
    calls and puts with their associated Greeks, volumes, and other metadata.
    
    Attributes:
        ticker (str): The stock ticker symbol
        risk_free_rate (float): Risk-free interest rate for calculations
        dividend_yield (float): Continuous dividend yield
    """
    
    def __init__(
        self,
        ticker: str,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the option chain fetcher.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual continuous dividend yield
        """
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self._yf_ticker = None
        
    def _get_yf_ticker(self):
        """Lazily load yfinance ticker object."""
        if self._yf_ticker is None:
            try:
                import yfinance as yf
                self._yf_ticker = yf.Ticker(self.ticker)
            except ImportError:
                raise ImportError(
                    "yfinance is required for fetching option chains. "
                    "Install with: pip install yfinance"
                )
        return self._yf_ticker
    
    @property
    def spot_price(self) -> float:
        """Get the current spot price of the underlying."""
        ticker = self._get_yf_ticker()
        # Try to get the most recent price
        hist = ticker.history(period="1d")
        if len(hist) > 0:
            return float(hist['Close'].iloc[-1])
        # Fallback to info
        info = ticker.info
        return float(info.get('regularMarketPrice', info.get('previousClose', np.nan)))
    
    @property
    def available_expirations(self) -> tuple:
        """Get available expiration dates for options."""
        ticker = self._get_yf_ticker()
        return ticker.options
    
    def fetch_chain(
        self,
        expiration: Optional[str] = None,
        option_type: str = "call",
        include_all_expirations: bool = False
    ) -> pd.DataFrame:
        """
        Fetch option chain data for specified expiration(s).
        
        Args:
            expiration: Specific expiration date (YYYY-MM-DD format).
                       If None and include_all_expirations=False, uses nearest expiration.
            option_type: 'call', 'put', or 'both'
            include_all_expirations: If True, fetches all available expirations
            
        Returns:
            DataFrame with option chain data including:
                - ticker, type, contractSymbol
                - expiration, days_to_exp, T (time to maturity in years)
                - S (spot price), strike, moneyness
                - bid, ask, mid, spread, rel_spread
                - impliedVolatility, volume, openInterest
                - inTheMoney
        """
        ticker = self._get_yf_ticker()
        spot = self.spot_price
        now = datetime.now(timezone.utc)
        
        # Determine which expirations to fetch
        if include_all_expirations:
            expirations = self.available_expirations
        elif expiration is not None:
            expirations = [expiration]
        else:
            expirations = [self.available_expirations[0]] if self.available_expirations else []
            
        if not expirations:
            warnings.warn(f"No option expirations available for {self.ticker}")
            return pd.DataFrame()
        
        all_chains = []
        
        for exp in expirations:
            try:
                chain = ticker.option_chain(exp)
            except Exception as e:
                warnings.warn(f"Failed to fetch chain for {exp}: {e}")
                continue
                
            # Parse expiration date
            exp_dt = pd.to_datetime(exp).tz_localize(timezone.utc)
            days_to_exp = (exp_dt - now).total_seconds() / (24 * 3600)
            T = days_to_exp / 365.0
            
            # Process calls
            if option_type in ("call", "both") and len(chain.calls) > 0:
                calls_df = self._process_chain_df(
                    chain.calls, spot, exp, exp_dt, days_to_exp, T, "call", now
                )
                all_chains.append(calls_df)
                
            # Process puts
            if option_type in ("put", "both") and len(chain.puts) > 0:
                puts_df = self._process_chain_df(
                    chain.puts, spot, exp, exp_dt, days_to_exp, T, "put", now
                )
                all_chains.append(puts_df)
                
        if not all_chains:
            return pd.DataFrame()
            
        result = pd.concat(all_chains, ignore_index=True)
        return result
    
    def _process_chain_df(
        self,
        df: pd.DataFrame,
        spot: float,
        exp: str,
        exp_dt: datetime,
        days_to_exp: float,
        T: float,
        option_type: str,
        now: datetime
    ) -> pd.DataFrame:
        """Process raw yfinance chain data into standardized format."""
        
        result = pd.DataFrame()
        
        result['ticker'] = self.ticker
        result['type'] = option_type
        result['contractSymbol'] = df['contractSymbol'].values
        result['asof_datetime'] = now
        result['lastTradeDate_dt'] = pd.to_datetime(df['lastTradeDate'])
        result['expiration'] = exp
        result['expiration_dt'] = exp_dt
        result['days_to_exp'] = days_to_exp
        result['T'] = T
        result['S'] = spot
        
        # Forward price (for dividend-adjusted moneyness)
        F = spot * np.exp((self.risk_free_rate - self.dividend_yield) * T)
        result['F'] = F
        result['r'] = self.risk_free_rate
        result['q'] = self.dividend_yield
        
        strike_vals = np.asarray(df['strike'].values, dtype=float)
        bid_vals = np.asarray(df['bid'].values, dtype=float)
        ask_vals = np.asarray(df['ask'].values, dtype=float)
        
        result['strike'] = strike_vals
        result['moneyness'] = strike_vals / spot
        
        # Prices
        result['bid'] = bid_vals
        result['ask'] = ask_vals
        result['mid'] = (bid_vals + ask_vals) / 2
        result['spread'] = ask_vals - bid_vals
        
        # Relative spread (spread / mid)
        mid_vals = np.asarray(result['mid'].values, dtype=float)
        spread_vals = np.asarray(result['spread'].values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            result['rel_spread'] = np.where(
                mid_vals > 0,
                spread_vals / mid_vals,
                np.inf
            )
        
        result['lastPrice'] = df['lastPrice'].values
        result['impliedVolatility'] = df['impliedVolatility'].values
        result['volume'] = df['volume'].fillna(0).values
        result['openInterest'] = df['openInterest'].fillna(0).values
        result['inTheMoney'] = df['inTheMoney'].values
        result['currency'] = 'USD'
        
        # Weight based on inverse relative spread (higher weight = tighter spread)
        rel_spread_vals = np.asarray(result['rel_spread'].values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(
                rel_spread_vals > 0,
                1.0 / rel_spread_vals,
                0.0
            )
            result['weight'] = weights
        
        return result
    
    def fetch_from_csv(
        self,
        filepath: str,
        option_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load option chain data from a CSV file.
        
        Args:
            filepath: Path to CSV file with option chain data
            option_type: Filter by 'call' or 'put', or None for both
            
        Returns:
            DataFrame with option chain data
        """
        df = pd.read_csv(filepath)
        
        if option_type is not None:
            df = df[df['type'] == option_type].copy()
            
        return df
