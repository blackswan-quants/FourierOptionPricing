import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """Split configuration."""
    test_size: float = 0.2
    min_strikes_per_maturity: int = 5
    random_state: Optional[int] = None


class OptionDataSplitter:
    """
    Train-test split utilities for processed option data.

    Expected columns:
    - QuoteDate
    - Maturity
    - Strike
    - OptionType
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()

    def cross_sectional_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split each (QuoteDate, Maturity) group by strike.

        Train: ~80% of strikes
        Test: ~20% of hidden strikes
        """
        required_cols = {"QuoteDate", "Maturity", "Strike"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rng = np.random.default_rng(self.config.random_state)

        train_idx = []
        test_idx = []
        skipped_groups = 0

        for _, group in df.groupby(["QuoteDate", "Maturity"]):
            strikes = np.sort(group["Strike"].dropna().unique())

            if len(strikes) < self.config.min_strikes_per_maturity:
                train_idx.extend(group.index.tolist())
                skipped_groups += 1
                continue

            # Prefer internal strikes to reduce boundary extrapolation.
            internal_strikes = strikes[1:-1] if len(strikes) >= 5 else strikes

            if len(internal_strikes) == 0:
                train_idx.extend(group.index.tolist())
                skipped_groups += 1
                continue

            n_test = max(1, int(round(len(strikes) * self.config.test_size)))
            n_test = min(n_test, len(internal_strikes), len(strikes) - 1)

            test_strikes = rng.choice(internal_strikes, size=n_test, replace=False)

            mask_test = group["Strike"].isin(test_strikes)
            test_idx.extend(group[mask_test].index.tolist())
            train_idx.extend(group[~mask_test].index.tolist())

        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        stats = {
            "strategy": "cross_sectional",
            "train_size": len(train_df),
            "test_size": len(test_df),
            "test_ratio_rows": len(test_df) / len(df) if len(df) > 0 else 0.0,
            "skipped_small_groups": skipped_groups,
        }

        return train_df, test_df, stats

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_t: Optional[str] = None,
        date_t1: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split data by two quote dates.

        Train: data at time t
        Test: data at time t+1

        If dates are not provided, the last two available quote dates are used.
        """
        required_cols = {"QuoteDate", "Maturity", "Strike"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data = df.copy()
        data["QuoteDate"] = pd.to_datetime(data["QuoteDate"])
        data["Maturity"] = pd.to_datetime(data["Maturity"])

        available_dates = np.sort(data["QuoteDate"].dropna().unique())
        if len(available_dates) < 2:
            raise ValueError("Need at least two quote dates for temporal split.")

        if date_t is None or date_t1 is None:
            date_t = pd.Timestamp(available_dates[-2])
            date_t1 = pd.Timestamp(available_dates[-1])
        else:
            date_t = pd.Timestamp(date_t)
            date_t1 = pd.Timestamp(date_t1)

        train_df = data[data["QuoteDate"] == date_t].copy()
        test_df = data[data["QuoteDate"] == date_t1].copy()

        align_cols = ["Maturity", "Strike"]
        if "OptionType" in data.columns:
            align_cols.append("OptionType")

        # Keep only contracts available on both dates.
        train_keys = train_df[align_cols].drop_duplicates()
        test_keys = test_df[align_cols].drop_duplicates()
        common_keys = train_keys.merge(test_keys, on=align_cols, how="inner")

        train_df = train_df.merge(common_keys, on=align_cols, how="inner")
        test_df = test_df.merge(common_keys, on=align_cols, how="inner")

        stats = {
            "strategy": "temporal",
            "train_date": str(date_t.date()),
            "test_date": str(date_t1.date()),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "aligned_on": align_cols,
        }

        return train_df, test_df, stats
    