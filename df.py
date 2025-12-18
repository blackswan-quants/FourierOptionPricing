#!/usr/bin/env python3
"""
Download & clean option chains from Yahoo Finance (via yfinance) and save to CSV.

Example:
  python download_yahoo_options.py --ticker SPY --out option_chain_SPY.csv

Notes:
- Yahoo option data is fine for prototyping; it is not institutional-grade.
- SPX chains are often incomplete/unreliable on Yahoo; SPY or large caps are safer.
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Filters:
    # maturities
    min_days: int = 7
    max_days: int = 365 * 2

    # moneyness K/S
    moneyness_min: float = 0.80
    moneyness_max: float = 1.20

    # liquidity
    min_volume: int = 1
    min_open_interest: int = 1
    liquidity_rule: str = "or"  # "or" or "and"

    # microstructure / sanity
    require_bid_ask: bool = True
    drop_zero_bid: bool = True
    max_rel_spread: float = 0.25      # (ask-bid)/mid
    min_mid: float = 0.01             # avoid tiny mids that explode rel_spread

    # stale-trade filter
    max_stale_days: int | None = 7    # set None to disable

    # rates placeholders (can be overwritten later)
    r: float = 0.0
    q: float = 0.0


def _get_spot_proxy(tk: yf.Ticker) -> float:
    """
    Best-effort spot proxy from yfinance without requiring intraday history.
    """
    # fast_info exists in recent yfinance versions and is usually faster
    try:
        fi = getattr(tk, "fast_info", None)
        if fi is not None:
            for key in ("last_price", "lastPrice", "regular_market_price", "regularMarketPrice"):
                v = fi.get(key, None) if hasattr(fi, "get") else getattr(fi, key, None)
                if v is not None and np.isfinite(v) and v > 0:
                    return float(v)
    except Exception:
        pass

    # fallback to info (slower; sometimes throttled)
    try:
        info = tk.info
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            v = info.get(key, None)
            if v is not None and np.isfinite(v) and v > 0:
                return float(v)
    except Exception:
        pass

    # final fallback: last close from recent history
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError("Unable to fetch spot proxy from yfinance.")
    return float(hist["Close"].iloc[-1])


def fetch_all_chains(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        raise RuntimeError(
            f"No option expirations found for ticker '{ticker}'. "
            f"(Yahoo may not provide options for this symbol.)"
        )

    asof_dt = pd.Timestamp.now(tz="UTC")
    S = _get_spot_proxy(tk)

    frames = []
    for exp in expirations:
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"
        calls["expiration"] = exp
        puts["expiration"] = exp

        df = pd.concat([calls, puts], ignore_index=True)
        df["ticker"] = ticker
        df["asof_datetime"] = pd.Timestamp.now(tz="UTC")
        df["S"] = S
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def clean_and_filter(df: pd.DataFrame, flt: Filters) -> pd.DataFrame:
    df = df.copy()

    # Standardize columns
    needed = ["bid", "ask", "strike", "volume", "openInterest", "impliedVolatility", "lastPrice", "lastTradeDate"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure asof_datetime exists and is tz-aware UTC
    if "asof_datetime" not in df.columns:
        df["asof_datetime"] = pd.Timestamp.utcnow().tz_localize("UTC")
    df["asof_datetime"] = pd.to_datetime(df["asof_datetime"], utc=True, errors="coerce")

    # Parse expiration to UTC midnight (date-based)
    df["expiration_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)

    # Time to maturity in ACT/365 using asof_datetime
    # Use exact seconds rather than day count to avoid off-by-one.
    dt_seconds = (df["expiration_dt"] - df["asof_datetime"]).dt.total_seconds()
    df["T"] = dt_seconds / (365.0 * 24.0 * 3600.0)
    df["days_to_exp"] = dt_seconds / (24.0 * 3600.0)

    # Basic validity
    base_drop = ["strike", "T", "S", "expiration_dt", "asof_datetime"]
    if flt.require_bid_ask:
        base_drop += ["bid", "ask"]
    df = df.dropna(subset=base_drop)

    df = df[(df["T"] > 0) & (df["S"] > 0)]
    df = df[df["strike"] > 0]

    # Parse lastTradeDate (optional)
    df["lastTradeDate_dt"] = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")

    # Stale filter (optional): remove contracts that haven't traded recently
    if flt.max_stale_days is not None:
        # If lastTradeDate is missing, keep (Yahoo sometimes omits it),
        # but you can flip this logic if you prefer stricter filtering.
        stale_cut = df["asof_datetime"] - pd.Timedelta(days=int(flt.max_stale_days))
        df = df[(df["lastTradeDate_dt"].isna()) | (df["lastTradeDate_dt"] >= stale_cut)]

    # Microstructure sanity: enforce positive ask and strict ask > bid (avoid crossed/locked)
    df = df[(df["ask"] > 0) & (df["bid"] >= 0)]
    df = df[df["ask"] > df["bid"]]

    if flt.drop_zero_bid:
        df = df[df["bid"] > 0]

    # Mid & spread (do NOT clip; crossed markets already removed)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]

    df = df[df["mid"] >= flt.min_mid]

    # Maturity filters
    df = df[(df["days_to_exp"] >= flt.min_days) & (df["days_to_exp"] <= flt.max_days)]

    # Moneyness filters
    df["moneyness"] = df["strike"] / df["S"]
    df = df[(df["moneyness"] >= flt.moneyness_min) & (df["moneyness"] <= flt.moneyness_max)]

    # Liquidity filters
    vol = df["volume"].fillna(0)
    oi = df["openInterest"].fillna(0)
    if flt.liquidity_rule.lower() == "and":
        df = df[(vol >= flt.min_volume) & (oi >= flt.min_open_interest)]
    else:
        df = df[(vol >= flt.min_volume) | (oi >= flt.min_open_interest)]

    # Relative spread filter
    df["rel_spread"] = (df["spread"] / df["mid"]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["rel_spread"])
    df = df[df["rel_spread"] <= flt.max_rel_spread]

    # Rates placeholders + forward
    df["r"] = float(flt.r)
    df["q"] = float(flt.q)
    df["F"] = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])

    # Static no-arbitrage bounds (discounted intrinsic + simple upper bounds)
    # Lower bounds:
    #   C >= max(S e^{-qT} - K e^{-rT}, 0)
    #   P >= max(K e^{-rT} - S e^{-qT}, 0)
    disc_S = df["S"] * np.exp(-df["q"] * df["T"])
    disc_K = df["strike"] * np.exp(-df["r"] * df["T"])

    is_call = df["type"].astype(str).str.lower().eq("call").to_numpy()
    is_put = ~is_call

    mid = df["mid"].to_numpy()
    lb_call = np.maximum(disc_S.to_numpy() - disc_K.to_numpy(), 0.0)
    lb_put = np.maximum(disc_K.to_numpy() - disc_S.to_numpy(), 0.0)

    # Upper bounds:
    #   C <= S e^{-qT}
    #   P <= K e^{-rT}
    ub_call = disc_S.to_numpy()
    ub_put = disc_K.to_numpy()

    ok = np.ones(len(df), dtype=bool)
    eps = 1e-10
    ok[is_call] &= (mid[is_call] + eps >= lb_call[is_call]) & (mid[is_call] <= ub_call[is_call] + 1e-6)
    ok[is_put]  &= (mid[is_put]  + eps >= lb_put[is_put])  & (mid[is_put]  <= ub_put[is_put]  + 1e-6)
    df = df.loc[ok].copy()

    # Calibration weight suggestion (optional but useful)
    # Inverse-spread weighting: higher weight for tighter quotes.
    df["weight"] = 1.0 / (df["spread"] + 1e-6)

    # Final sort
    df = df.sort_values(["expiration_dt", "type", "strike"]).reset_index(drop=True)
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Download Yahoo Finance option chains and save to CSV.")
    p.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., SPY, AAPL).")
    p.add_argument("--out", type=str, default=None, help="Output CSV path.")

    p.add_argument("--min-days", type=int, default=7)
    p.add_argument("--max-days", type=int, default=365 * 2)

    p.add_argument("--m-min", type=float, default=0.80, help="Min moneyness K/S.")
    p.add_argument("--m-max", type=float, default=1.20, help="Max moneyness K/S.")

    p.add_argument("--min-vol", type=int, default=1)
    p.add_argument("--min-oi", type=int, default=1)
    p.add_argument("--liquidity-rule", type=str, default="or", choices=["or", "and"])

    p.add_argument("--max-rel-spread", type=float, default=0.25, help="Max (ask-bid)/mid.")
    p.add_argument("--min-mid", type=float, default=0.01)
    p.add_argument("--keep-zero-bid", action="store_true", help="Do not drop bid==0 quotes.")

    p.add_argument("--max-stale-days", type=int, default=7, help="Drop quotes with lastTradeDate older than this many days. Use -1 to disable.")

    # placeholders for rates (can be improved later)
    p.add_argument("--r", type=float, default=0.0, help="Risk-free rate (flat, placeholder).")
    p.add_argument("--q", type=float, default=0.0, help="Dividend yield (flat, placeholder).")

    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper().strip()
    out = args.out or f"option_chain_{ticker}.csv"

    max_stale_days = None if args.max_stale_days < 0 else int(args.max_stale_days)

    flt = Filters(
        min_days=args.min_days,
        max_days=args.max_days,
        moneyness_min=args.m_min,
        moneyness_max=args.m_max,
        min_volume=args.min_vol,
        min_open_interest=args.min_oi,
        liquidity_rule=args.liquidity_rule,
        max_rel_spread=args.max_rel_spread,
        min_mid=args.min_mid,
        drop_zero_bid=not args.keep_zero_bid,
        max_stale_days=max_stale_days,
        r=args.r,
        q=args.q,
        require_bid_ask=True,
    )

    try:
        raw = fetch_all_chains(ticker)
        clean = clean_and_filter(raw, flt)

        cols = [
            "ticker", "type", "contractSymbol",
            "asof_datetime", "lastTradeDate_dt",
            "expiration", "expiration_dt", "days_to_exp", "T",
            "S", "F", "r", "q",
            "strike", "moneyness",
            "bid", "ask", "mid", "spread", "rel_spread",
            "lastPrice", "impliedVolatility",
            "volume", "openInterest",
            "inTheMoney", "currency",
            "weight",
        ]
        cols = [c for c in cols if c in clean.columns]
        clean[cols].to_csv(out, index=False)

        print(f"Saved: {out}")
        print(
            f"Rows: {len(clean)} | Expirations: {clean['expiration'].nunique()} "
            f"| Spot(S): {clean['S'].iloc[0]:.4f} | AsOf: {clean['asof_datetime'].iloc[0]}"
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
