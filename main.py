"""
main.py
=======
End-to-end pipeline for the Thapar Canteen Sales Optimizer.

Pipeline steps
--------------
1. LOAD     – generate synthetic CSV (first run) then load and clean it.
2. EDA      – print summary statistics and save a 2×2 dashboard chart.
3. FEATURES – engineer daily-level features (lag, rolling mean, time flags).
4. SPLIT    – time-aware chronological train / test split (no data leakage).
5. TRAIN    – fit a GradientBoostingRegressor (swap easily in config/model.py).
6. EVALUATE – compute MAE, MSE, RMSE, R² on the held-out test set.
7. VISUALISE– save actual-vs-predicted, residuals, and feature-importance plots.
8. REPORT   – print an executive summary to the console.

Run from the repo root with:
    python main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Project modules ───────────────────────────────────────────────────
from config import Cols, OUTPUT_DIR
from data_loader import load_data
from features import build_daily_features
from model import SalesPredictor, time_aware_split
from visualization import (
    plot_actual_vs_predicted,
    plot_eda_dashboard,
    plot_feature_importances,
    plot_residuals,
    plot_sales_over_time,
)

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ── Pipeline steps ────────────────────────────────────────────────────

def step_load() -> pd.DataFrame:
    """Step 1 — Load and clean the raw transaction data.

    Returns:
        Cleaned transaction-level DataFrame.
    """
    logger.info("━━  STEP 1 / LOAD  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    transaction_df = load_data()
    total_rev = transaction_df[Cols.REVENUE].sum()
    logger.info(
        "Dataset: %d rows | ₹%s total revenue | %d unique items",
        len(transaction_df),
        f"{total_rev:,.0f}",
        transaction_df[Cols.ITEM].nunique(),
    )
    return transaction_df


def step_eda(transaction_df: pd.DataFrame) -> None:
    """Step 2 — Print quick stats and save EDA dashboard.

    Args:
        transaction_df: Cleaned transaction-level DataFrame.
    """
    logger.info("━━  STEP 2 / EDA   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\nTop 5 items by revenue:")
    top5 = (
        transaction_df.groupby(Cols.ITEM)[Cols.REVENUE]
        .sum()
        .nlargest(5)
    )
    for item, rev in top5.items():
        print(f"  {item:<15} ₹{rev:,.2f}")

    print("\nPeak hours by revenue:")
    peak = (
        transaction_df.groupby(Cols.HOUR)[Cols.REVENUE]
        .sum()
        .nlargest(5)
    )
    for hr, rev in peak.items():
        print(f"  {hr:02d}:00  ₹{rev:,.2f}")

    plot_eda_dashboard(transaction_df)


def step_features(transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — Aggregate to daily level and engineer features.

    Args:
        transaction_df: Cleaned transaction-level DataFrame.

    Returns:
        Daily-level DataFrame with engineered feature columns.
    """
    logger.info("━━  STEP 3 / FEATURES  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    daily_df = build_daily_features(transaction_df)
    logger.info("Daily feature matrix: %d rows × %d columns", *daily_df.shape)
    return daily_df


def step_split(daily_df: pd.DataFrame):
    """Step 4 — Chronological train / test split.

    Args:
        daily_df: Daily feature DataFrame.

    Returns:
        ``SplitData`` namedtuple with X_train, X_test, y_train, y_test.
    """
    logger.info("━━  STEP 4 / SPLIT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return time_aware_split(daily_df)


def step_train(split) -> SalesPredictor:
    """Step 5 — Fit the regression model.

    To use a different algorithm, change the constructor argument here:
        SalesPredictor(RandomForestRegressor(n_estimators=200))

    Args:
        split: Output of ``time_aware_split``.

    Returns:
        Fitted ``SalesPredictor`` instance.
    """
    logger.info("━━  STEP 5 / TRAIN  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    predictor = SalesPredictor()          # defaults to GradientBoostingRegressor
    predictor.fit(split)
    return predictor


def step_evaluate(predictor: SalesPredictor, split) -> None:
    """Step 6 — Evaluate the model on the held-out test set.

    Args:
        predictor: Fitted ``SalesPredictor``.
        split:     Output of ``time_aware_split``.
    """
    logger.info("━━  STEP 6 / EVALUATE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    metrics = predictor.evaluate(split.X_test, split.y_test)
    print(f"\n  Model  : {type(predictor.regressor).__name__}")
    print(f"  MAE    : ₹{metrics.mae:,.2f}")
    print(f"  MSE    : ₹{metrics.mse:,.2f}")
    print(f"  RMSE   : ₹{metrics.rmse:,.2f}")
    print(f"  R²     : {metrics.r2:.4f}  (1.0 = perfect)\n")
    return metrics


def step_visualise(
    predictor: SalesPredictor,
    daily_df: pd.DataFrame,
    split,
) -> None:
    """Step 7 — Save all model-evaluation and insight charts.

    Args:
        predictor: Fitted ``SalesPredictor``.
        daily_df:  Full daily feature DataFrame (used for the time-series plot).
        split:     Output of ``time_aware_split``.
    """
    logger.info("━━  STEP 7 / VISUALISE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    y_pred = predictor.predict(split.X_test)

    # Recover test-period dates for the time-aligned line chart
    n_test      = len(split.X_test)
    test_dates  = daily_df[Cols.DATE].iloc[-n_test:].reset_index(drop=True)

    plot_sales_over_time(daily_df)
    plot_actual_vs_predicted(split.y_test, y_pred, test_dates)
    plot_residuals(split.y_test, y_pred)
    plot_feature_importances(predictor.feature_importances())


def step_report(metrics, output_dir: Path = OUTPUT_DIR) -> None:
    """Step 8 — Print executive summary.

    Args:
        metrics:    ``ModelMetrics`` returned by ``step_evaluate``.
        output_dir: Folder where all charts were saved.
    """
    logger.info("━━  STEP 8 / REPORT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("=" * 60)
    print("  EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"  R²   = {metrics.r2:.4f}  (higher is better, max 1.0)")
    print(f"  RMSE = ₹{metrics.rmse:,.2f}  (average prediction error)")
    print(f"  MAE  = ₹{metrics.mae:,.2f}  (average absolute error)")
    print(f"  All charts saved in: {output_dir}/")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Run the full canteen sales optimiser pipeline."""
    print("=" * 60)
    print("  THAPAR CANTEEN SALES OPTIMIZER")
    print("=" * 60)

    transaction_df = step_load()
    step_eda(transaction_df)
    daily_df       = step_features(transaction_df)
    split          = step_split(daily_df)
    predictor      = step_train(split)
    metrics        = step_evaluate(predictor, split)
    step_visualise(predictor, daily_df, split)
    step_report(metrics)


if __name__ == "__main__":
    main()
