"""
Option Data Filters.

Provides filtering functions to clean option chain data for calibration:
- Liquidity filters (volume, open interest)
- Moneyness and maturity filters
- Bid-ask spread filters
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    """
    Configuration for option data filters.
    
    Attributes:
        min_volume: Minimum trading volume (filter illiquid options)
        min_open_interest: Minimum open interest
        min_days_to_exp: Minimum days to expiration (avoid gamma instability)
        max_days_to_exp: Maximum days to expiration
        min_moneyness: Minimum moneyness (K/S) - filters deep ITM calls / deep OTM puts
        max_moneyness: Maximum moneyness (K/S) - filters deep OTM calls / deep ITM puts
        max_relative_spread: Maximum bid-ask spread relative to mid price
        min_bid: Minimum bid price (filter near-zero quotes)
        min_mid_price: Minimum mid price
        remove_zero_bid: Whether to remove options with zero bid
        remove_itm: Whether to remove in-the-money options
    """
    min_volume: int = 10
    min_open_interest: int = 100
    min_days_to_exp: float = 7.0
    max_days_to_exp: float = 365.0
    min_moneyness: float = 0.80
    max_moneyness: float = 1.20
    max_relative_spread: float = 0.50
    min_bid: float = 0.05
    min_mid_price: float = 0.10
    remove_zero_bid: bool = True
    remove_itm: bool = False


class OptionDataFilter:
    """
    Filters option chain data based on liquidity, moneyness, and quality criteria.
    
    The filter pipeline removes noisy or unreliable options that could
    destabilize the calibration algorithm. Each filter can be applied
    individually or as part of a complete pipeline.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the filter with configuration.
        
        Args:
            config: FilterConfig object with filter parameters.
                   If None, uses default configuration.
        """
        self.config = config or FilterConfig()
        self._filter_stats: Dict[str, Any] = {}
        
    @property
    def filter_stats(self) -> dict:
        """Statistics from the last filter operation."""
        return self._filter_stats.copy()
    
    def filter_liquidity(
        self,
        df: pd.DataFrame,
        min_volume: Optional[int] = None,
        min_open_interest: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter options by liquidity metrics.
        
        Removes options with low trading volume or open interest,
        as these tend to have unreliable/stale prices.
        
        Args:
            df: Option chain DataFrame
            min_volume: Minimum volume threshold (uses config if None)
            min_open_interest: Minimum open interest threshold
            
        Returns:
            Filtered DataFrame
        """
        min_vol = min_volume if min_volume is not None else self.config.min_volume
        min_oi = min_open_interest if min_open_interest is not None else self.config.min_open_interest
        
        initial_count = len(df)
        
        mask = (df['volume'] >= min_vol) & (df['openInterest'] >= min_oi)
        result = df[mask].copy()
        
        self._filter_stats['liquidity'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result),
            'min_volume': min_vol,
            'min_open_interest': min_oi
        }
        
        return result
    
    def filter_maturity(
        self,
        df: pd.DataFrame,
        min_days: Optional[float] = None,
        max_days: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter options by time to maturity.
        
        Removes very short-dated options (gamma instability) and
        very long-dated options (less relevant for calibration).
        
        Args:
            df: Option chain DataFrame
            min_days: Minimum days to expiration (default: 7)
            max_days: Maximum days to expiration (default: 365)
            
        Returns:
            Filtered DataFrame
        """
        min_d = min_days if min_days is not None else self.config.min_days_to_exp
        max_d = max_days if max_days is not None else self.config.max_days_to_exp
        
        initial_count = len(df)
        
        mask = (df['days_to_exp'] >= min_d) & (df['days_to_exp'] <= max_d)
        result = df[mask].copy()
        
        self._filter_stats['maturity'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result),
            'min_days': min_d,
            'max_days': max_d
        }
        
        return result
    
    def filter_moneyness(
        self,
        df: pd.DataFrame,
        min_moneyness: Optional[float] = None,
        max_moneyness: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter options by moneyness (strike / spot).
        
        Removes deep OTM and deep ITM options where:
        - Prices are very small (OTM) leading to large relative errors
        - Bid-ask spreads are typically wider
        - Market microstructure effects dominate
        
        Args:
            df: Option chain DataFrame
            min_moneyness: Minimum K/S ratio (default: 0.80)
            max_moneyness: Maximum K/S ratio (default: 1.20)
            
        Returns:
            Filtered DataFrame
        """
        min_m = min_moneyness if min_moneyness is not None else self.config.min_moneyness
        max_m = max_moneyness if max_moneyness is not None else self.config.max_moneyness
        
        initial_count = len(df)
        
        mask = (df['moneyness'] >= min_m) & (df['moneyness'] <= max_m)
        result = df[mask].copy()
        
        self._filter_stats['moneyness'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result),
            'min_moneyness': min_m,
            'max_moneyness': max_m
        }
        
        return result
    
    def filter_spread(
        self,
        df: pd.DataFrame,
        max_relative_spread: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter options by bid-ask spread.
        
        Removes options where the spread is too wide relative to the
        option value, indicating poor price discovery or illiquidity.
        
        Args:
            df: Option chain DataFrame
            max_relative_spread: Maximum spread/mid ratio (default: 0.50)
            
        Returns:
            Filtered DataFrame
        """
        max_spread = (max_relative_spread if max_relative_spread is not None 
                      else self.config.max_relative_spread)
        
        initial_count = len(df)
        
        mask = df['rel_spread'] <= max_spread
        result = df[mask].copy()
        
        self._filter_stats['spread'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result),
            'max_relative_spread': max_spread
        }
        
        return result
    
    def filter_prices(
        self,
        df: pd.DataFrame,
        min_bid: Optional[float] = None,
        min_mid: Optional[float] = None,
        remove_zero_bid: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Filter options by price quality.
        
        Removes options with very low or zero bids, which often
        indicate stale quotes or extremely OTM options.
        
        Args:
            df: Option chain DataFrame
            min_bid: Minimum bid price
            min_mid: Minimum mid price
            remove_zero_bid: Whether to remove zero-bid options
            
        Returns:
            Filtered DataFrame
        """
        min_b = min_bid if min_bid is not None else self.config.min_bid
        min_m = min_mid if min_mid is not None else self.config.min_mid_price
        remove_zero = remove_zero_bid if remove_zero_bid is not None else self.config.remove_zero_bid
        
        initial_count = len(df)
        
        mask = (df['mid'] >= min_m)
        
        if remove_zero:
            mask &= (df['bid'] > 0)
        else:
            mask &= (df['bid'] >= min_b)
            
        result = df[mask].copy()
        
        self._filter_stats['prices'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result),
            'min_bid': min_b,
            'min_mid': min_m,
            'remove_zero_bid': remove_zero
        }
        
        return result
    
    def filter_itm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out in-the-money options.
        
        ITM options can be useful but often have wider spreads
        and early exercise considerations (for American options).
        
        Args:
            df: Option chain DataFrame
            
        Returns:
            Filtered DataFrame (OTM options only)
        """
        initial_count = len(df)
        
        if 'inTheMoney' in df.columns:
            result = df[~df['inTheMoney']].copy()
        else:
            # Fallback: use moneyness
            # For calls: ITM when K < S (moneyness < 1)
            # For puts: ITM when K > S (moneyness > 1)
            if 'type' in df.columns:
                mask = (
                    ((df['type'] == 'call') & (df['moneyness'] >= 1.0)) |
                    ((df['type'] == 'put') & (df['moneyness'] <= 1.0))
                )
            else:
                # Assume calls, filter for OTM
                mask = df['moneyness'] >= 1.0
            result = df[mask].copy()
        
        self._filter_stats['itm'] = {
            'initial': initial_count,
            'remaining': len(result),
            'removed': initial_count - len(result)
        }
        
        return result
    
    def apply_all_filters(
        self,
        df: pd.DataFrame,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Apply all filters in sequence.
        
        Filter order:
        1. Price quality (zero bids, min prices)
        2. Liquidity (volume, open interest)
        3. Maturity (days to expiration)
        4. Moneyness (strike / spot range)
        5. Spread (bid-ask quality)
        6. ITM removal (optional)
        
        Args:
            df: Option chain DataFrame
            verbose: If True, print filter statistics
            
        Returns:
            Fully filtered DataFrame
        """
        self._filter_stats = {'initial_count': len(df)}
        
        result = df.copy()
        
        # Apply filters in order
        result = self.filter_prices(result)
        result = self.filter_liquidity(result)
        result = self.filter_maturity(result)
        result = self.filter_moneyness(result)
        result = self.filter_spread(result)
        
        if self.config.remove_itm:
            result = self.filter_itm(result)
        
        self._filter_stats['final_count'] = len(result)
        self._filter_stats['total_removed'] = len(df) - len(result)
        self._filter_stats['retention_rate'] = len(result) / len(df) if len(df) > 0 else 0
        
        if verbose:
            self._print_stats()
            
        return result
    
    def _print_stats(self):
        """Print filter statistics."""
        stats = self._filter_stats
        print("\n" + "="*60)
        print("OPTION DATA FILTER STATISTICS")
        print("="*60)
        print(f"Initial options: {stats['initial_count']}")
        print("-"*60)
        
        for filter_name in ['prices', 'liquidity', 'maturity', 'moneyness', 'spread', 'itm']:
            if filter_name in stats:
                fs = stats[filter_name]
                print(f"{filter_name.upper():12s}: {fs['initial']:5d} -> {fs['remaining']:5d} "
                      f"(removed {fs['removed']:4d})")
        
        print("-"*60)
        print(f"Final options: {stats['final_count']}")
        print(f"Total removed: {stats['total_removed']}")
        print(f"Retention rate: {stats['retention_rate']:.1%}")
        print("="*60 + "\n")
    
    def get_moneyness_distribution(
        self,
        df: pd.DataFrame,
        bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distribution of moneyness in the filtered data.
        
        Args:
            df: Option chain DataFrame
            bins: Number of histogram bins
            
        Returns:
            Tuple of (bin_edges, counts)
        """
        counts, bin_edges = np.histogram(df['moneyness'], bins=bins)
        return bin_edges, counts
    
    def get_maturity_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get statistics by maturity bucket.
        
        Args:
            df: Option chain DataFrame
            
        Returns:
            DataFrame with stats per maturity
        """
        if 'expiration' in df.columns:
            return df.groupby('expiration').agg({
                'strike': 'count',
                'moneyness': ['min', 'max', 'mean'],
                'volume': 'sum',
                'openInterest': 'sum',
                'rel_spread': 'mean'
            }).round(4)
        return pd.DataFrame()
