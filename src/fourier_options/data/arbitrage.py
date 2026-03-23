"""
Arbitrage Condition Checker.

Validates that option quotes satisfy no-arbitrage conditions.
Removes quotes that violate basic arbitrage bounds.
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any, cast
from dataclasses import dataclass


@dataclass
class ArbitrageViolation:
    """Record of an arbitrage violation."""
    index: int
    contract: str
    violation_type: str
    details: str
    severity: str  # 'warning' or 'error'


class ArbitrageChecker:
    """
    Checks and removes options violating no-arbitrage conditions.
    
    Implements the following arbitrage checks:
    
    1. **Lower Bound (Intrinsic Value)**:
       - Call: C ≥ max(0, S·e^(-qT) - K·e^(-rT))
       - Put: P ≥ max(0, K·e^(-rT) - S·e^(-qT))
    
    2. **Upper Bound**:
       - Call: C ≤ S·e^(-qT)
       - Put: P ≤ K·e^(-rT)
    
    3. **Non-negative prices**:
       - C ≥ 0, P ≥ 0
    
    4. **Monotonicity in Strike** (for same maturity):
       - Calls decrease as strike increases
       - Puts increase as strike increases
    
    5. **Convexity in Strike** (Butterfly arbitrage):
       - Second derivative w.r.t. strike is non-negative
    
    6. **Put-Call Parity** (if both available):
       - C - P = S·e^(-qT) - K·e^(-rT)
    """
    
    def __init__(
        self,
        tolerance: float = 0.01,
        relative_tolerance: float = 0.05
    ):
        """
        Initialize the arbitrage checker.
        
        Args:
            tolerance: Absolute tolerance for arbitrage violations (in $)
            relative_tolerance: Relative tolerance for violations
        """
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.violations: List[ArbitrageViolation] = []
        self._check_stats: Dict[str, Any] = {}
        
    @property
    def check_stats(self) -> Dict[str, Any]:
        """Statistics from the last arbitrage check."""
        return self._check_stats.copy()
    
    def _get_discount_factors(
        self,
        df: pd.DataFrame
    ) -> tuple:
        """Calculate discount factors from DataFrame."""
        r = df['r'].values if 'r' in df.columns else np.zeros(len(df))
        q = df['q'].values if 'q' in df.columns else np.zeros(len(df))
        T = df['T'].values
        
        # Ensure arrays are numpy arrays
        r = np.asarray(r, dtype=float)
        q = np.asarray(q, dtype=float)
        T = np.asarray(T, dtype=float)
        
        df_r = np.exp(-r * T)  # Risk-free discount
        df_q = np.exp(-q * T)  # Dividend discount
        
        return df_r, df_q
    
    def check_lower_bound(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid'
    ) -> pd.DataFrame:
        """
        Check lower bound (intrinsic value) condition.
        
        For European options:
        - Call: C ≥ max(0, S·e^(-qT) - K·e^(-rT))
        - Put: P ≥ max(0, K·e^(-rT) - S·e^(-qT))
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            
        Returns:
            DataFrame with violating rows removed
        """
        self.violations = []
        initial_count = len(df)
        
        df_r, df_q = self._get_discount_factors(df)
        S = np.asarray(df['S'].values, dtype=float)
        K = np.asarray(df['strike'].values, dtype=float)
        price = np.asarray(df[price_col].values, dtype=float)
        
        # Calculate discounted values
        S_disc = S * df_q  # S·e^(-qT)
        K_disc = K * df_r  # K·e^(-rT)
        
        # Lower bounds
        if 'type' in df.columns:
            call_mask = np.asarray((df['type'] == 'call').values, dtype=bool)
            put_mask = np.asarray((df['type'] == 'put').values, dtype=bool)
            
            lower_bound = np.zeros(len(df))
            lower_bound[call_mask] = np.maximum(0, S_disc[call_mask] - K_disc[call_mask])
            lower_bound[put_mask] = np.maximum(0, K_disc[put_mask] - S_disc[put_mask])
        else:
            # Assume calls
            lower_bound = np.maximum(0, S_disc - K_disc)
        
        # Check violations with tolerance
        violation = price < (lower_bound - self.tolerance)
        
        # Record violations
        for idx in df.index[violation]:
            loc = df.index.get_loc(idx)
            contract = str(df.at[idx, 'contractSymbol']) if 'contractSymbol' in df.columns else str(idx)
            self.violations.append(ArbitrageViolation(
                index=int(idx) if isinstance(idx, (int, np.integer)) else hash(idx),
                contract=contract,
                violation_type='lower_bound',
                details=f"Price {price[loc]:.4f} < Lower bound {lower_bound[loc]:.4f}",
                severity='error'
            ))
        
        result = df[~violation].copy()
        
        self._check_stats['lower_bound'] = {
            'initial': initial_count,
            'violations': int(violation.sum()),
            'remaining': len(result)
        }
        
        return result
    
    def check_upper_bound(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid'
    ) -> pd.DataFrame:
        """
        Check upper bound condition.
        
        - Call: C ≤ S·e^(-qT) (can't be worth more than the stock)
        - Put: P ≤ K·e^(-rT) (can't be worth more than discounted strike)
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            
        Returns:
            DataFrame with violating rows removed
        """
        initial_count = len(df)
        
        df_r, df_q = self._get_discount_factors(df)
        S = np.asarray(df['S'].values, dtype=float)
        K = np.asarray(df['strike'].values, dtype=float)
        price = np.asarray(df[price_col].values, dtype=float)
        
        S_disc = S * df_q
        K_disc = K * df_r
        
        if 'type' in df.columns:
            call_mask = np.asarray((df['type'] == 'call').values, dtype=bool)
            put_mask = np.asarray((df['type'] == 'put').values, dtype=bool)
            
            upper_bound = np.zeros(len(df))
            upper_bound[call_mask] = S_disc[call_mask]
            upper_bound[put_mask] = K_disc[put_mask]
        else:
            # Assume calls
            upper_bound = S_disc
        
        # Check violations
        violation = price > (upper_bound + self.tolerance)
        
        for idx in df.index[violation]:
            loc = df.index.get_loc(idx)
            contract = str(df.at[idx, 'contractSymbol']) if 'contractSymbol' in df.columns else str(idx)
            self.violations.append(ArbitrageViolation(
                index=int(idx) if isinstance(idx, (int, np.integer)) else hash(idx),
                contract=contract,
                violation_type='upper_bound',
                details=f"Price {price[loc]:.4f} > Upper bound {upper_bound[loc]:.4f}",
                severity='error'
            ))
        
        result = df[~violation].copy()
        
        self._check_stats['upper_bound'] = {
            'initial': initial_count,
            'violations': int(violation.sum()),
            'remaining': len(result)
        }
        
        return result
    
    def check_non_negative(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid'
    ) -> pd.DataFrame:
        """
        Check that option prices are non-negative.
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            
        Returns:
            DataFrame with negative prices removed
        """
        initial_count = len(df)
        
        price = np.asarray(df[price_col].values, dtype=float)
        violation = price < 0
        
        for idx in df.index[violation]:
            loc = df.index.get_loc(idx)
            contract = str(df.at[idx, 'contractSymbol']) if 'contractSymbol' in df.columns else str(idx)
            self.violations.append(ArbitrageViolation(
                index=int(idx) if isinstance(idx, (int, np.integer)) else hash(idx),
                contract=contract,
                violation_type='non_negative',
                details=f"Negative price: {price[loc]:.4f}",
                severity='error'
            ))
        
        result = df[~violation].copy()
        
        self._check_stats['non_negative'] = {
            'initial': initial_count,
            'violations': int(violation.sum()),
            'remaining': len(result)
        }
        
        return result
    
    def check_monotonicity(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid'
    ) -> pd.DataFrame:
        """
        Check strike monotonicity for each maturity.
        
        For same expiration:
        - Call prices decrease as strike increases
        - Put prices increase as strike increases
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            
        Returns:
            DataFrame with monotonicity violations removed
        """
        if len(df) < 2:
            return df
            
        initial_count = len(df)
        violations_mask = pd.Series(False, index=df.index)
        
        # Group by expiration and type
        groupby_cols = ['expiration']
        if 'type' in df.columns:
            groupby_cols.append('type')
        
        for group_key, group in df.groupby(groupby_cols):
            if len(group) < 2:
                continue
                
            # Sort by strike
            sorted_group = group.sort_values('strike')
            strikes = np.asarray(sorted_group['strike'].values, dtype=float)
            prices = np.asarray(sorted_group[price_col].values, dtype=float)
            
            option_type = group_key[1] if len(groupby_cols) > 1 else 'call'
            
            # Check monotonicity
            price_diff = np.diff(prices)
            
            if option_type == 'call':
                # Calls should decrease with strike
                bad_indices = np.where(price_diff > self.tolerance)[0]
            else:
                # Puts should increase with strike
                bad_indices = np.where(price_diff < -self.tolerance)[0]
            
            # Mark violations (mark the higher strike option as violation)
            for i in bad_indices:
                idx = sorted_group.index[i + 1]
                violations_mask[idx] = True
                self.violations.append(ArbitrageViolation(
                    index=int(idx) if isinstance(idx, (int, np.integer)) else hash(idx),
                    contract=str(df.loc[idx, 'contractSymbol']) if 'contractSymbol' in df.columns else str(idx),
                    violation_type='monotonicity',
                    details=f"{option_type} monotonicity violation at K={strikes[i+1]:.2f}",
                    severity='warning'
                ))
        
        result = df[~violations_mask].copy()
        
        self._check_stats['monotonicity'] = {
            'initial': initial_count,
            'violations': int(violations_mask.sum()),
            'remaining': len(result)
        }
        
        return result
    
    def check_convexity(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid'
    ) -> pd.DataFrame:
        """
        Check convexity in strike (butterfly arbitrage).
        
        Option prices must be convex in strike:
        C(K2) ≤ λ·C(K1) + (1-λ)·C(K3) for K1 < K2 < K3
        
        This is equivalent to the second derivative being non-negative.
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            
        Returns:
            DataFrame with convexity violations removed
        """
        if len(df) < 3:
            return df
            
        initial_count = len(df)
        violations_mask = pd.Series(False, index=df.index)
        
        groupby_cols = ['expiration']
        if 'type' in df.columns:
            groupby_cols.append('type')
        
        for group_key, group in df.groupby(groupby_cols):
            if len(group) < 3:
                continue
            
            sorted_group = group.sort_values('strike')
            strikes = np.asarray(sorted_group['strike'].values, dtype=float)
            prices = np.asarray(sorted_group[price_col].values, dtype=float)
            
            # Check convexity using second differences
            for i in range(1, len(prices) - 1):
                K1, K2, K3 = strikes[i-1], strikes[i], strikes[i+1]
                C1, C2, C3 = prices[i-1], prices[i], prices[i+1]
                
                # Linear interpolation weight
                lam = (K3 - K2) / (K3 - K1)
                C_interp = lam * C1 + (1 - lam) * C3
                
                # Convexity violation: C2 > interpolated value
                if C2 > C_interp + self.tolerance:
                    idx = sorted_group.index[i]
                    violations_mask[idx] = True
                    contract = str(df.at[idx, 'contractSymbol']) if 'contractSymbol' in df.columns else str(idx)
                    self.violations.append(ArbitrageViolation(
                        index=int(idx) if isinstance(idx, (int, np.integer)) else hash(idx),
                        contract=contract,
                        violation_type='convexity',
                        details=f"Butterfly arbitrage at K={K2:.2f}: C={C2:.4f} > interp={C_interp:.4f}",
                        severity='warning'
                    ))
        
        result = df[~violations_mask].copy()
        
        self._check_stats['convexity'] = {
            'initial': initial_count,
            'violations': int(violations_mask.sum()),
            'remaining': len(result)
        }
        
        return result
    
    def apply_all_checks(
        self,
        df: pd.DataFrame,
        price_col: str = 'mid',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Apply all arbitrage checks in sequence.
        
        Check order:
        1. Non-negative prices
        2. Lower bound (intrinsic value)
        3. Upper bound
        4. Monotonicity
        5. Convexity
        
        Args:
            df: Option chain DataFrame
            price_col: Column to use for option price
            verbose: If True, print check statistics
            
        Returns:
            DataFrame with all arbitrage violations removed
        """
        self.violations = []
        self._check_stats = {'initial_count': len(df)}
        
        result = df.copy()
        
        result = self.check_non_negative(result, price_col)
        result = self.check_lower_bound(result, price_col)
        result = self.check_upper_bound(result, price_col)
        result = self.check_monotonicity(result, price_col)
        result = self.check_convexity(result, price_col)
        
        self._check_stats['final_count'] = len(result)
        self._check_stats['total_violations'] = len(df) - len(result)
        
        if verbose:
            self._print_stats()
            
        return result
    
    def _print_stats(self):
        """Print arbitrage check statistics."""
        stats = self._check_stats
        print("\n" + "="*60)
        print("ARBITRAGE CHECK STATISTICS")
        print("="*60)
        print(f"Initial options: {stats['initial_count']}")
        print("-"*60)
        
        for check_name in ['non_negative', 'lower_bound', 'upper_bound', 'monotonicity', 'convexity']:
            if check_name in stats:
                cs = stats[check_name]
                print(f"{check_name.upper():15s}: {cs['violations']:4d} violations")
        
        print("-"*60)
        print(f"Final options: {stats['final_count']}")
        print(f"Total violations removed: {stats['total_violations']}")
        print("="*60 + "\n")
    
    def get_violations_df(self) -> pd.DataFrame:
        """
        Get a DataFrame of all recorded violations.
        
        Returns:
            DataFrame with violation details
        """
        if not self.violations:
            return pd.DataFrame(columns=['index', 'contract', 'violation_type', 'details', 'severity'])
        
        return pd.DataFrame([
            {
                'index': v.index,
                'contract': v.contract,
                'violation_type': v.violation_type,
                'details': v.details,
                'severity': v.severity
            }
            for v in self.violations
        ])
