"""
visualization.py
================
Every plot the pipeline produces lives here.

Rules enforced in this module:
* No model training or data loading — only charting.
* Every function saves its figure to ``OUTPUT_DIR`` and returns the
  ``Path`` it wrote to, so callers can log or display the location.
* ``plt.close()`` is called after every save to avoid memory leaks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import Cols, FIGURE_DPI, OUTPUT_DIR, STYLE

logger = logging.getLogger(__name__)
sns.set_style(STYLE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── EDA plots ─────────────────────────────────────────────────────────

def plot_eda_dashboard(transaction_df: pd.DataFrame) -> Path:
    """Save a 2 × 2 EDA overview dashboard.

    Panels:
      - Top-left:  Revenue heat-map (hour × weekday).
      - Top-right: Top 8 items by total revenue (horizontal bar).
      - Bot-left:  Revenue distribution by student type (box plot).
      - Bot-right: Total revenue by hour of day (bar chart).

    Args:
        transaction_df: Cleaned transaction-level DataFrame.

    Returns:
        Path to the saved PNG file.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Thapar Canteen — EDA Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Revenue heat-map
    pivot = transaction_df.pivot_table(
        values=Cols.REVENUE,
        index=Cols.HOUR,
        columns=Cols.DAY_OF_WEEK,
        aggfunc="sum",
    )
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        ax=axes[0, 0], cbar_kws={"label": "Revenue (₹)"},
    )
    axes[0, 0].set_title("Revenue Heatmap: Hour × Day of Week", fontweight="bold")
    axes[0, 0].set_xlabel("Day of Week (0=Mon, 6=Sun)")
    axes[0, 0].set_ylabel("Hour of Day")

    # Panel 2: Top 8 items by revenue
    top8_items = (
        transaction_df.groupby(Cols.ITEM)[Cols.REVENUE]
        .sum()
        .nlargest(8)
        .sort_values()
    )
    top8_items.plot(kind="barh", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Top 8 Items by Total Revenue", fontweight="bold")
    axes[0, 1].set_xlabel("Revenue (₹)")

    # Panel 3: Revenue by student type (box plot)
    student_groups = [
        transaction_df.loc[
            transaction_df[Cols.STUDENT_TYPE] == stype, Cols.REVENUE
        ].values
        for stype in transaction_df[Cols.STUDENT_TYPE].unique()
    ]
    axes[1, 0].boxplot(student_groups,
                       labels=transaction_df[Cols.STUDENT_TYPE].unique())
    axes[1, 0].set_title("Revenue Distribution by Student Type", fontweight="bold")
    axes[1, 0].set_xlabel("Student Type")
    axes[1, 0].set_ylabel("Revenue per Transaction (₹)")

    # Panel 4: Hourly revenue
    hourly_rev = transaction_df.groupby(Cols.HOUR)[Cols.REVENUE].sum()
    hourly_rev.plot(kind="bar", ax=axes[1, 1], color="coral")
    axes[1, 1].set_title("Total Revenue by Hour of Day", fontweight="bold")
    axes[1, 1].set_xlabel("Hour")
    axes[1, 1].set_ylabel("Revenue (₹)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_dashboard.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EDA dashboard → %s", out_path)
    return out_path


def plot_sales_over_time(daily_df: pd.DataFrame) -> Path:
    """Line chart of daily sales across the full date range.

    Args:
        daily_df: Daily-level DataFrame with ``date`` and ``sales`` columns.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_df[Cols.DATE], daily_df[Cols.SALES],
            color="steelblue", linewidth=1.5, label="Daily Sales")

    # 7-day rolling average for trend visibility
    rolling_avg = daily_df[Cols.SALES].rolling(7).mean()
    ax.plot(daily_df[Cols.DATE], rolling_avg,
            color="darkorange", linewidth=2.5, linestyle="--",
            label="7-Day Rolling Average")

    ax.set_title("Canteen Daily Sales Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "sales_over_time.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sales-over-time chart → %s", out_path)
    return out_path


# ── Model evaluation plots ────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    test_dates: pd.Series | None = None,
) -> Path:
    """Two-panel chart: scatter + optional time-aligned line comparison.

    Left panel: scatter of actual vs predicted (closer to diagonal = better).
    Right panel: line chart comparing actual and predicted over test dates
    (only rendered when ``test_dates`` is supplied).

    Args:
        y_true:     Ground-truth sales values (test partition).
        y_pred:     Model predictions aligned with ``y_true``.
        test_dates: Optional date index for the right-panel line chart.

    Returns:
        Path to the saved PNG file.
    """
    has_dates = test_dates is not None
    n_panels  = 2 if has_dates else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]   # make iterable

    # --- Scatter panel ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, color="purple", alpha=0.65, s=55,
               label="Test samples", zorder=3)
    # Perfect-prediction diagonal
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--",
            linewidth=2, label="Perfect prediction")
    ax.set_title("Actual vs Predicted Sales", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Sales (₹)")
    ax.set_ylabel("Predicted Sales (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Line chart panel (optional) ---
    if has_dates:
        ax2 = axes[1]
        ax2.plot(test_dates.values, y_true.values,
                 color="steelblue", label="Actual", linewidth=2)
        ax2.plot(test_dates.values, y_pred,
                 color="darkorange", linestyle="--", label="Predicted", linewidth=2)
        ax2.set_title("Sales: Actual vs Predicted (Test Period)",
                      fontsize=13, fontweight="bold")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Daily Revenue (₹)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "actual_vs_predicted.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved actual-vs-predicted chart → %s", out_path)
    return out_path


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """Two-panel residuals diagnostic chart.

    Left panel: histogram + KDE of residuals (should be centred near 0).
    Right panel: residuals vs predicted values (should show no pattern).

    Args:
        y_true: Ground-truth sales values.
        y_pred: Model predictions aligned with ``y_true``.

    Returns:
        Path to the saved PNG file.
    """
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residuals Diagnostics", fontsize=14, fontweight="bold")

    # Left: distribution of residuals
    sns.histplot(residuals, kde=True, ax=axes[0], color="steelblue", bins=20)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title("Residuals Distribution", fontweight="bold")
    axes[0].set_xlabel("Residual (Actual − Predicted)")
    axes[0].set_ylabel("Count")

    # Right: residuals vs predicted (look for heteroscedasticity)
    axes[1].scatter(y_pred, residuals, color="purple", alpha=0.65, s=55)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("Residuals vs Predicted", fontweight="bold")
    axes[1].set_xlabel("Predicted Sales (₹)")
    axes[1].set_ylabel("Residual (₹)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "residuals.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residuals chart → %s", out_path)
    return out_path


def plot_feature_importances(importances: pd.Series) -> Path:
    """Horizontal bar chart of model feature importances.

    Args:
        importances: Sorted Series returned by ``SalesPredictor.feature_importances()``.

    Returns:
        Path to the saved PNG file.  Returns early if Series is empty.
    """
    if importances.empty:
        logger.warning("No feature importances available — skipping plot.")
        return OUTPUT_DIR / "feature_importances_skipped.txt"

    fig, ax = plt.subplots(figsize=(10, 5))
    importances.sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "feature_importances.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importances chart → %s", out_path)
    return out_path
