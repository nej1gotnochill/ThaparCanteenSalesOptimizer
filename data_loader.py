"""
data_loader.py
==============
Responsible for two tasks:

1. ``generate_raw_data`` – synthesises a realistic 90-day transaction log
   and writes it to ``data/canteen_sales.csv`` (runs once).
2. ``load_data`` – reads the CSV and returns a clean, typed DataFrame.

No model logic or visualisation belongs here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CANTEEN_CLOSE_HOUR,
    CANTEEN_OPEN_HOUR,
    Cols,
    MENU_PRICES,
    OFF_PEAK_TRANSACTIONS,
    PEAK_HOURS,
    PEAK_TRANSACTIONS,
    RANDOM_SEED,
    RAW_CSV_PATH,
    SIMULATION_DAYS,
    STUDENT_TYPES,
)

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────

def generate_raw_data(
    days: int = SIMULATION_DAYS,
    output_path: Path = RAW_CSV_PATH,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic canteen transaction log and persist it as CSV.

    Creates one row per individual transaction (item purchase) over
    ``days`` calendar days. Temperature follows a slow seasonal sine wave
    so the data has mild non-linearity for the model to learn.

    Args:
        days: Number of calendar days to simulate.
        output_path: Destination path for the CSV file.
        seed: NumPy random seed for reproducibility.

    Returns:
        Raw transaction-level DataFrame (before feature engineering).
    """
    np.random.seed(seed)

    items = list(MENU_PRICES.keys())
    start_date = datetime.now() - timedelta(days=days)
    rows: list[dict] = []

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        # Smooth temperature that rises and falls over weeks
        temperature = round(
            25 + 10 * np.sin(day_offset / 30) + np.random.normal(0, 2), 1
        )

        for hour in range(CANTEEN_OPEN_HOUR, CANTEEN_CLOSE_HOUR):
            n_transactions = (
                PEAK_TRANSACTIONS if hour in PEAK_HOURS else OFF_PEAK_TRANSACTIONS
            )

            for _ in range(n_transactions):
                item = np.random.choice(items)
                quantity = np.random.randint(1, 4)
                price = MENU_PRICES[item]

                rows.append(
                    {
                        Cols.TIMESTAMP:    current_date.replace(
                            hour=hour, minute=np.random.randint(0, 60)
                        ),
                        Cols.DATE:         current_date.date(),
                        Cols.HOUR:         hour,
                        Cols.DAY_OF_WEEK:  current_date.weekday(),  # 0=Mon … 6=Sun
                        Cols.ITEM:         item,
                        Cols.QUANTITY:     quantity,
                        Cols.PRICE:        price,
                        Cols.REVENUE:      price * quantity,
                        Cols.STUDENT_TYPE: np.random.choice(STUDENT_TYPES),
                        Cols.TEMPERATURE:  temperature,
                    }
                )

    raw_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_path, index=False)
    logger.info("Saved %d transaction rows to %s", len(raw_df), output_path)
    return raw_df


def load_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the canteen CSV and return a clean, correctly typed DataFrame.

    If the CSV does not exist, ``generate_raw_data`` is called first so
    the project always runs end-to-end out of the box.

    Args:
        csv_path: Path to the canteen CSV file.

    Returns:
        Cleaned transaction-level DataFrame with a proper datetime index.
    """
    if not csv_path.exists():
        logger.warning("CSV not found at %s — generating synthetic data.", csv_path)
        generate_raw_data(output_path=csv_path)

    df = pd.read_csv(csv_path)
    if _needs_regeneration(df):
        logger.warning(
            "CSV items do not match the current menu — regenerating %s.",
            csv_path,
        )
        generate_raw_data(output_path=csv_path)
        df = pd.read_csv(csv_path)
    df = _clean(df)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


# ── Private helpers ───────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dtype corrections and drop obviously corrupt rows.

    Args:
        df: Raw DataFrame straight from CSV.

    Returns:
        Cleaned DataFrame.
    """
    # Parse timestamp and drop rows where it failed
    df[Cols.TIMESTAMP] = pd.to_datetime(df[Cols.TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[Cols.TIMESTAMP]).copy()

    # Ensure numeric columns are numeric (guards against CSV corruption)
    for col in (Cols.QUANTITY, Cols.PRICE, Cols.REVENUE, Cols.TEMPERATURE):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=[Cols.REVENUE, Cols.QUANTITY])

    # Sort chronologically — important for time-aware train/test split
    df = df.sort_values(Cols.TIMESTAMP).reset_index(drop=True)
    return df


def _needs_regeneration(df: pd.DataFrame) -> bool:
    """Return True when the cached CSV no longer matches the configured menu."""
    if Cols.ITEM not in df.columns:
        return True

    current_items = set(df[Cols.ITEM].dropna().astype(str).unique())
    expected_items = set(MENU_PRICES.keys())
    return current_items != expected_items
