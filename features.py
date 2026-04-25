"""
features.py
===========
All feature-engineering logic lives here.

The public entry point is ``build_daily_features``, which collapses the
raw transaction log into one row per calendar day and attaches time-based
features that the regression model can learn from.

Keeping this module separate means you can add new features (e.g. holiday
flags, rolling averages) in one place without touching the model or loader.
"""

from __future__ import annotations

import pandas as pd

from config import Cols


# ── Public API ────────────────────────────────────────────────────────

def build_daily_features(transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to daily level and attach engineered features.

    Steps performed:
      1. Sum quantity (→ DEM) and revenue (→ sales) per calendar day.
      2. Attach the first temperature reading of each day.
      3. Derive time-based flags: day-of-week, month, is_weekend, is_peak_day.
      4. Compute 7-day rolling average of sales (trend signal).
      5. Compute 1-day lag of DEM (yesterday's demand as a predictor).

    Args:
        transaction_df: Cleaned transaction-level DataFrame from
            ``data_loader.load_data``.

    Returns:
        Daily-level DataFrame with DEM, sales, and all engineered features.
        Rows with NaN introduced by lag/rolling are dropped.
    """
    daily_df = _aggregate_to_daily(transaction_df)
    daily_df = _add_time_features(daily_df)
    daily_df = _add_rolling_features(daily_df)
    daily_df = daily_df.dropna().reset_index(drop=True)
    return daily_df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used by the model.

    Centralising this list means ``model.py`` and ``main.py`` never hard-code
    column names — swap or add features here and the rest updates automatically.

    Returns:
        List of feature column name strings.
    """
    return [
        Cols.DEM,
        Cols.TEMPERATURE,
        Cols.DAY_OF_WEEK,
        Cols.MONTH,
        Cols.IS_WEEKEND,
        Cols.IS_PEAK_HOUR,   # re-used as "is_high_demand_day" at daily level
        "dem_lag_1",
        "sales_rolling_7",
    ]


# ── Private helpers ───────────────────────────────────────────────────

def _aggregate_to_daily(transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transaction rows into one row per calendar date.

    Args:
        transaction_df: Raw transaction-level DataFrame.

    Returns:
        Daily-level DataFrame with columns: date, DEM, sales, temperature.
    """
    daily_df = (
        transaction_df
        .groupby(Cols.DATE)
        .agg(
            DEM=(Cols.QUANTITY, "sum"),
            sales=(Cols.REVENUE, "sum"),
            temperature=(Cols.TEMPERATURE, "first"),  # same for whole day
            day_of_week=(Cols.DAY_OF_WEEK, "first"),
        )
        .reset_index()
    )
    # Make date a proper datetime so we can extract month, etc.
    daily_df[Cols.DATE] = pd.to_datetime(daily_df[Cols.DATE])
    return daily_df


def _add_time_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar-based features from the date column.

    Args:
        daily_df: Daily-level DataFrame with a datetime ``date`` column.

    Returns:
        DataFrame with additional time feature columns.
    """
    daily_df = daily_df.copy()
    daily_df[Cols.MONTH]        = daily_df[Cols.DATE].dt.month
    # is_weekend: 1 if Saturday (5) or Sunday (6), else 0
    daily_df[Cols.IS_WEEKEND]   = (daily_df[Cols.DAY_OF_WEEK] >= 5).astype(int)
    # is_peak_hour repurposed at daily level: 1 if a weekday (Mon–Fri)
    daily_df[Cols.IS_PEAK_HOUR] = (daily_df[Cols.DAY_OF_WEEK] < 5).astype(int)
    return daily_df


def _add_rolling_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-window features for trend and momentum signals.

    Args:
        daily_df: Daily-level DataFrame sorted chronologically.

    Returns:
        DataFrame with ``dem_lag_1`` and ``sales_rolling_7`` columns.
        Rows where these features are NaN (start of series) are left in
        so the caller can decide whether to drop them.
    """
    daily_df = daily_df.copy()
    # Yesterday's total demand — a strong predictor of today's demand
    daily_df["dem_lag_1"] = daily_df[Cols.DEM].shift(1)
    # 7-day rolling mean of sales — captures weekly trend
    daily_df["sales_rolling_7"] = (
        daily_df[Cols.SALES].shift(1).rolling(window=7).mean()
    )
    return daily_df
