"""
model.py
========
Model training, time-aware splitting, and evaluation.

Design goals
------------
* ``SalesPredictor`` wraps *any* scikit-learn–compatible regressor so you
  can swap GradientBoostingRegressor for RandomForest, Ridge, XGBoost, etc.
  by changing one line in ``main.py``.
* ``evaluate`` returns a plain dict of metrics so callers can log or display
  them without coupling to this module.
* ``time_aware_split`` ensures no future data leaks into training — critical
  for time-series forecasting correctness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import GBR_PARAMS, SPLIT_SEED, TEST_SIZE, Cols
from features import get_feature_columns

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────

@dataclass
class SplitData:
    """Holds the four arrays produced by ``time_aware_split``.

    Attributes:
        X_train: Feature matrix for training.
        X_test:  Feature matrix for testing.
        y_train: Target vector for training.
        y_test:  Target vector for testing.
    """
    X_train: pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_test:  pd.Series


@dataclass
class ModelMetrics:
    """Evaluation metrics for one model run.

    Attributes:
        mae:  Mean Absolute Error.
        mse:  Mean Squared Error.
        rmse: Root Mean Squared Error.
        r2:   Coefficient of determination (R²).
    """
    mae:  float
    mse:  float
    rmse: float
    r2:   float

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:,.2f}  MSE={self.mse:,.2f}  "
            f"RMSE={self.rmse:,.2f}  R²={self.r2:.4f}"
        )


# ── Core class ────────────────────────────────────────────────────────

class SalesPredictor:
    """Wraps a scikit-learn regressor with a canteen-specific interface.

    Swap the underlying algorithm by passing any sklearn-compatible estimator
    to the constructor.  Everything else — split, fit, predict, evaluate —
    stays the same.

    Example::

        from sklearn.ensemble import RandomForestRegressor
        model = SalesPredictor(RandomForestRegressor(n_estimators=200))
        model.fit(split)
        metrics = model.evaluate(split.X_test, split.y_test)

    Attributes:
        regressor: The underlying sklearn estimator.
        feature_cols: Ordered list of feature column names used during fit.
        is_fitted: True after ``fit`` has been called.
    """

    def __init__(self, regressor: BaseEstimator | None = None) -> None:
        """Initialise with an optional regressor.

        Args:
            regressor: Any sklearn-compatible regressor.  Defaults to
                ``GradientBoostingRegressor`` with parameters from ``config``.
        """
        self.regressor: BaseEstimator = (
            regressor if regressor is not None
            else GradientBoostingRegressor(**GBR_PARAMS)
        )
        self.feature_cols: list[str] = get_feature_columns()
        self.is_fitted: bool = False

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, split: SplitData) -> "SalesPredictor":
        """Train the regressor on the training partition.

        Args:
            split: Output of ``time_aware_split`` containing X_train / y_train.

        Returns:
            Self (for method chaining).
        """
        logger.info(
            "Training %s on %d samples …",
            type(self.regressor).__name__,
            len(split.X_train),
        )
        self.regressor.fit(split.X_train[self.feature_cols], split.y_train)
        self.is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for a feature matrix.

        Args:
            X: DataFrame containing at least the columns in ``feature_cols``.

        Returns:
            1-D NumPy array of predicted sales values.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.regressor.predict(X[self.feature_cols])

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> ModelMetrics:
        """Compute regression metrics on a held-out partition.

        Args:
            X:      Feature matrix (test partition).
            y_true: Ground-truth sales values.

        Returns:
            ``ModelMetrics`` dataclass with MAE, MSE, RMSE, and R².
        """
        y_pred = self.predict(X)
        mse  = mean_squared_error(y_true, y_pred)
        metrics = ModelMetrics(
            mae  = mean_absolute_error(y_true, y_pred),
            mse  = mse,
            rmse = float(np.sqrt(mse)),
            r2   = r2_score(y_true, y_pred),
        )
        logger.info("Test metrics → %s", metrics)
        return metrics

    # ── Feature importance ────────────────────────────────────────────

    def feature_importances(self) -> pd.Series:
        """Return feature importances as a labelled Series (if supported).

        Args:
            None.

        Returns:
            Sorted pandas Series (descending) of feature importances.
            Returns an empty Series if the estimator does not expose
            ``feature_importances_``.
        """
        if not hasattr(self.regressor, "feature_importances_"):
            logger.warning("%s does not expose feature_importances_.",
                           type(self.regressor).__name__)
            return pd.Series(dtype=float)

        return (
            pd.Series(
                self.regressor.feature_importances_,
                index=self.feature_cols,
                name="importance",
            )
            .sort_values(ascending=False)
        )


# ── Splitting helper ──────────────────────────────────────────────────

def time_aware_split(
    daily_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> SplitData:
    """Split daily data chronologically — no random shuffling.

    Using a chronological split prevents data leakage: the model never
    sees future information during training, which would inflate metrics.

    Args:
        daily_df: Daily-level DataFrame returned by ``build_daily_features``.
        test_size: Fraction of rows to allocate to the test set (default 0.20).

    Returns:
        ``SplitData`` with X_train, X_test, y_train, y_test.
    """
    feature_cols = get_feature_columns()
    n_total      = len(daily_df)
    n_test       = max(1, int(n_total * test_size))
    n_train      = n_total - n_test

    train_df = daily_df.iloc[:n_train]
    test_df  = daily_df.iloc[n_train:]

    logger.info(
        "Time-aware split: %d train days / %d test days", n_train, n_test
    )
    return SplitData(
        X_train = train_df[feature_cols],
        X_test  = test_df[feature_cols],
        y_train = train_df[Cols.SALES],
        y_test  = test_df[Cols.SALES],
    )
