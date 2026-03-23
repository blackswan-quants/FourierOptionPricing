from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spy_2020_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLEAN_OUTPUT_PATH = OUTPUT_DIR / "processed_spy_option_chain.csv"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "empirical_data_quality_report.json"

REQUIRED_COLUMNS = [
    "quote_date",
    "expire_date",
    "dte",
    "underlying_last",
    "strike",
    "c_bid",
    "c_ask",
    "c_volume",
    "c_last",
    "p_bid",
    "p_ask",
    "p_volume",
    "p_last",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize raw column names.
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.lower()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_data(path: Path) -> pd.DataFrame:
    # Load raw CSV.
    logger.info("Loading input from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Raw shape=%s", df.shape)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    # Check required fields.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    # Cast dates and numeric fields.
    df = df.copy()

    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df["expire_date"] = pd.to_datetime(df["expire_date"], errors="coerce")

    numeric_cols = [
        "dte",
        "underlying_last",
        "strike",
        "c_bid",
        "c_ask",
        "c_volume",
        "c_last",
        "p_bid",
        "p_ask",
        "p_volume",
        "p_last",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def reshape_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Convert wide call/put rows into a standard long format.
    base = df[["quote_date", "expire_date", "dte", "underlying_last", "strike"]].copy()

    calls = base.copy()
    calls["option_type"] = "call"
    calls["bid"] = df["c_bid"]
    calls["ask"] = df["c_ask"]
    calls["volume"] = df["c_volume"]
    calls["last"] = df["c_last"]

    puts = base.copy()
    puts["option_type"] = "put"
    puts["bid"] = df["p_bid"]
    puts["ask"] = df["p_ask"]
    puts["volume"] = df["p_volume"]
    puts["last"] = df["p_last"]

    long_df = pd.concat([calls, puts], ignore_index=True)
    long_df["mid_price"] = (long_df["bid"] + long_df["ask"]) / 2.0
    long_df["ttm_years"] = long_df["dte"] / 365.0

    return long_df


def build_quality_report(df: pd.DataFrame) -> dict:
    # Compute basic quality metrics and structural alerts.
    report: dict = {}

    report["row_count"] = int(len(df))
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["null_counts"] = {
        col: int(df[col].isna().sum())
        for col in ["quote_date", "expire_date", "dte", "underlying_last", "strike", "bid", "ask", "volume", "last"]
    }

    report["non_positive_strike_rows"] = int((df["strike"] <= 0).sum())
    report["negative_dte_rows"] = int((df["dte"] < 0).sum())
    report["non_positive_spot_rows"] = int((df["underlying_last"] <= 0).sum())
    report["negative_bid_rows"] = int((df["bid"] < 0).sum())
    report["negative_ask_rows"] = int((df["ask"] < 0).sum())
    report["ask_below_bid_rows"] = int((df["ask"] < df["bid"]).sum())
    report["non_positive_mid_rows"] = int((df["mid_price"] <= 0).sum())

    strike_density = df.groupby(["quote_date", "expire_date"])["strike"].nunique()
    maturity_density = df.groupby("quote_date")["expire_date"].nunique()

    report["unique_quote_dates"] = int(df["quote_date"].nunique(dropna=True))
    report["unique_expirations"] = int(df["expire_date"].nunique(dropna=True))
    report["unique_strikes"] = int(df["strike"].nunique(dropna=True))

    report["strike_density_summary"] = {
        k: float(v) for k, v in strike_density.describe().fillna(0).to_dict().items()
    }
    report["maturity_density_summary"] = {
        k: float(v) for k, v in maturity_density.describe().fillna(0).to_dict().items()
    }

    report["zero_volume_ratio"] = float((df["volume"].fillna(0) <= 0).mean())
    rel_spread = (df["ask"] - df["bid"]) / df["mid_price"]
    report["wide_spread_ratio"] = float((rel_spread > 0.50).fillna(False).mean())

    alerts = []

    if report["duplicate_rows"] > 0:
        alerts.append(f"Found {report['duplicate_rows']} duplicate rows.")

    if report["ask_below_bid_rows"] > 0:
        alerts.append(f"Found {report['ask_below_bid_rows']} rows with ask < bid.")

    if report["negative_dte_rows"] > 0:
        alerts.append(f"Found {report['negative_dte_rows']} rows with negative DTE.")

    if report["strike_density_summary"]["25%"] < 10:
        alerts.append("Low strike density detected for a non-trivial part of the surface.")

    if report["maturity_density_summary"]["25%"] < 3:
        alerts.append("Low maturity density detected for a non-trivial part of quote dates.")

    if report["zero_volume_ratio"] > 0.50:
        alerts.append("More than 50% of contracts have zero or missing volume.")

    if report["wide_spread_ratio"] > 0.25:
        alerts.append("More than 25% of contracts have relative spread above 50%.")

    report["alerts"] = alerts
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Apply basic consistency and liquidity filters.
    logger.info("Starting data cleaning")

    clean = df.copy()
    initial_rows = len(clean)

    clean = clean.drop_duplicates()

    clean = clean.dropna(
        subset=[
            "quote_date",
            "expire_date",
            "dte",
            "underlying_last",
            "strike",
            "bid",
            "ask",
        ]
    )

    clean = clean[clean["strike"] > 0]
    clean = clean[clean["underlying_last"] > 0]
    clean = clean[clean["dte"] >= 0]
    clean = clean[clean["bid"] > 0]
    clean = clean[clean["ask"] > 0]
    clean = clean[clean["ask"] >= clean["bid"]]

    clean["mid_price"] = (clean["bid"] + clean["ask"]) / 2.0
    clean = clean[clean["mid_price"] > 0]

    clean["relative_spread"] = (clean["ask"] - clean["bid"]) / clean["mid_price"]
    clean["volume"] = clean["volume"].fillna(0)
    clean["last"] = clean["last"].fillna(0)

    clean = clean[(clean["volume"] > 0) | (clean["last"] > 0)]
    clean = clean[clean["relative_spread"] <= 0.50]

    clean["ttm_years"] = clean["dte"] / 365.0

    clean = clean.sort_values(
        ["quote_date", "expire_date", "strike", "option_type"]
    ).reset_index(drop=True)

    logger.info("Rows before cleaning=%d", initial_rows)
    logger.info("Rows after cleaning=%d", len(clean))
    logger.info("Dropped rows=%d", initial_rows - len(clean))

    return clean


def extract_clean_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Build the processed output.
    out = df[
        [
            "quote_date",
            "expire_date",
            "dte",
            "ttm_years",
            "strike",
            "option_type",
            "underlying_last",
            "mid_price",
        ]
    ].copy()

    out = out.rename(
        columns={
            "quote_date": "QuoteDate",
            "expire_date": "Maturity",
            "dte": "TTM_Days",
            "ttm_years": "TTM_Years",
            "strike": "Strike",
            "option_type": "OptionType",
            "underlying_last": "Spot",
            "mid_price": "MidPrice",
        }
    )

    return out


def save_outputs(processed_df: pd.DataFrame, report: dict) -> None:
    # Save processed dataframe and quality report.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(CLEAN_OUTPUT_PATH, index=False)

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Saved processed dataframe to %s", CLEAN_OUTPUT_PATH)
    logger.info("Saved quality report to %s", REPORT_OUTPUT_PATH)


def main() -> None:
    df_raw = load_data(DATA_PATH)
    df_raw = normalize_columns(df_raw)

    logger.info("Normalized columns=%s", list(df_raw.columns))

    validate_columns(df_raw)
    df_std = standardize_types(df_raw)
    df_long = reshape_to_long(df_std)

    logger.info("Long format shape=%s", df_long.shape)

    report = build_quality_report(df_long)

    if report["alerts"]:
        logger.warning("Structural alerts detected:")
        for alert in report["alerts"]:
            logger.warning(" - %s", alert)
    else:
        logger.info("No structural alerts detected")

    df_clean = clean_data(df_long)
    processed_df = extract_clean_matrix(df_clean)

    logger.info("Final processed shape=%s", processed_df.shape)
    logger.info("Unique quote dates after cleaning=%d", processed_df["QuoteDate"].nunique())
    logger.info("Unique maturities after cleaning=%d", processed_df["Maturity"].nunique())
    logger.info("Unique strikes after cleaning=%d", processed_df["Strike"].nunique())

    save_outputs(processed_df, report)


if __name__ == "__main__":
    main()