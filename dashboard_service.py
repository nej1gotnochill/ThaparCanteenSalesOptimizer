"""Utilities for exposing the optimizer as dashboard-ready JSON."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Cols, MENU_PRICES, OUTPUT_DIR
from data_loader import load_data
from features import build_daily_features
from model import SalesPredictor, time_aware_split
from visualization import (
    plot_actual_vs_predicted,
    plot_feature_importances,
    plot_residuals,
    plot_sales_over_time,
)


ITEM_PROFILES: dict[str, dict[str, Any]] = {
    "Maggi": {"category": "Snacks", "emoji": "🍜", "base_stock": 120},
    "Sandwich": {"category": "Snacks", "emoji": "🥪", "base_stock": 80},
    "Cold Coffee": {"category": "Beverage", "emoji": "🥤", "base_stock": 75},
    "Tea": {"category": "Beverage", "emoji": "☕", "base_stock": 200},
    "Patties": {"category": "Snacks", "emoji": "🥟", "base_stock": 70},
    "Burger": {"category": "Meals", "emoji": "🍔", "base_stock": 55},
    "Fries": {"category": "Snacks", "emoji": "🍟", "base_stock": 60},
    "Momos": {"category": "Meals", "emoji": "🥟", "base_stock": 50},
}

WASTE_COLORS = {
    "Consumed": "var(--success)",
    "Wasted": "var(--accent)",
    "Returned": "var(--primary-glow)",
}

ORDER_STATUSES = ("Completed", "Pending", "Cancelled")

VISUALIZATION_SPECS = (
    {
        "key": "salesOverTime",
        "title": "Sales Over Time",
        "description": "Daily revenue trend with rolling average.",
        "filename": "sales_over_time.png",
    },
    {
        "key": "actualVsPredicted",
        "title": "Actual vs Predicted",
        "description": "Model fit against held-out test data.",
        "filename": "actual_vs_predicted.png",
    },
    {
        "key": "residuals",
        "title": "Residual Diagnostics",
        "description": "Error spread and residual behavior.",
        "filename": "residuals.png",
    },
    {
        "key": "featureImportances",
        "title": "Feature Importances",
        "description": "Relative influence of model features.",
        "filename": "feature_importances.png",
    },
)


@lru_cache(maxsize=1)
def get_dashboard_snapshot() -> dict[str, Any]:
    """Train the model once and return a dashboard-ready response payload."""
    transaction_df = load_data()
    daily_df = build_daily_features(transaction_df)
    split = time_aware_split(daily_df)
    predictor = SalesPredictor()
    predictor.fit(split)
    metrics = predictor.evaluate(split.X_test, split.y_test)

    latest_date = pd.Timestamp(daily_df[Cols.DATE].iloc[-1])
    latest_sales = float(daily_df[Cols.SALES].iloc[-1])
    predicted_tomorrow_sales = _predict_next_day_sales(predictor, daily_df)

    analytics = _build_analytics(transaction_df, daily_df)
    inventory = _build_inventory(transaction_df)
    predictions = _build_predictions(transaction_df, daily_df, predicted_tomorrow_sales)
    model_diagnostics = _build_model_diagnostics(daily_df, split, predictor)
    waste = _build_waste(transaction_df, daily_df, inventory["products"])
    orders = _build_orders(transaction_df)

    snapshot = {
        "generatedAt": latest_date.isoformat(),
        "overview": {
            "todayRevenue": round(latest_sales, 2),
            "ordersCompleted": int(analytics["dailySales"][-1]["orders"]),
            "lowStockAlerts": sum(1 for product in inventory["products"] if product["low"]),
            "peakTraffic": int(max(analytics["hourlyDemand"], key=lambda row: row["value"])["value"]),
            "predictedTomorrowSales": round(predicted_tomorrow_sales, 2),
            "growth": _percentage_change(latest_sales, float(daily_df[Cols.SALES].tail(2).iloc[0])),
        },
        "analytics": analytics,
        "inventory": inventory,
        "predictions": predictions,
        "waste": waste,
        "orders": orders,
        "settings": {
            "canteenName": "A-Block Canteen",
            "owner": "Aman Sharma",
            "location": "Thapar University, Patiala",
            "operatingHours": "8:00 AM - 10:00 PM",
            "planPrice": 999,
        },
        "model": {
            "name": type(predictor.regressor).__name__,
            "mae": round(metrics.mae, 2),
            "mse": round(metrics.mse, 2),
            "rmse": round(metrics.rmse, 2),
            "r2": round(metrics.r2, 4),
        },
        "modelDiagnostics": model_diagnostics,
    }
    return snapshot


def predict_next_day_sales(temperature: float | None = None) -> dict[str, Any]:
    """Expose a lightweight single-value forecast for API consumers."""
    transaction_df = load_data()
    daily_df = build_daily_features(transaction_df)
    split = time_aware_split(daily_df)
    predictor = SalesPredictor()
    predictor.fit(split)
    prediction = _predict_next_day_sales(predictor, daily_df, temperature=temperature)
    return {
        "predictedSales": round(prediction, 2),
        "temperature": temperature,
        "model": type(predictor.regressor).__name__,
    }


@lru_cache(maxsize=1)
def get_visualization_assets() -> list[dict[str, str]]:
    """Generate and return metadata for core dashboard visualizations."""
    transaction_df = load_data()
    daily_df = build_daily_features(transaction_df)
    split = time_aware_split(daily_df)
    predictor = SalesPredictor()
    predictor.fit(split)

    y_pred = predictor.predict(split.X_test)
    n_test = len(split.X_test)
    test_dates = daily_df[Cols.DATE].iloc[-n_test:].reset_index(drop=True)

    plot_sales_over_time(daily_df)
    plot_actual_vs_predicted(split.y_test, y_pred, test_dates)
    plot_residuals(split.y_test, y_pred)
    plot_feature_importances(predictor.feature_importances())

    return [dict(spec) for spec in VISUALIZATION_SPECS]


def get_visualization_file_path(filename: str) -> Path:
    """Return an absolute path to a generated visualization file."""
    assets = get_visualization_assets()
    allowed = {item["filename"] for item in assets}
    if filename not in allowed:
        raise FileNotFoundError(filename)

    path = OUTPUT_DIR / filename
    if not path.exists():
        get_visualization_assets.cache_clear()
        get_visualization_assets()
    return path


def _build_model_diagnostics(
    daily_df: pd.DataFrame,
    split,
    predictor: SalesPredictor,
) -> dict[str, Any]:
    y_pred = predictor.predict(split.X_test)
    n_test = len(split.X_test)
    test_dates = daily_df[Cols.DATE].iloc[-n_test:].reset_index(drop=True)

    actual_vs_predicted = [
        {
            "day": row[Cols.DATE].strftime("%a"),
            "date": row[Cols.DATE].strftime("%Y-%m-%d"),
            "actual": round(float(actual), 2),
            "predicted": round(float(predicted), 2),
        }
        for row, actual, predicted in zip(
            daily_df.iloc[-n_test:].itertuples(index=False),
            split.y_test.tolist(),
            y_pred.tolist(),
            strict=False,
        )
    ]

    residuals = [
        {
            "predicted": round(float(predicted), 2),
            "residual": round(float(actual - predicted), 2),
        }
        for actual, predicted in zip(split.y_test.tolist(), y_pred.tolist(), strict=False)
    ]

    feature_importances = [
        {"name": name, "value": round(float(value), 4)}
        for name, value in predictor.feature_importances().head(8).items()
    ]

    return {
        "actualVsPredicted": actual_vs_predicted,
        "residuals": residuals,
        "featureImportances": feature_importances,
    }


def _build_analytics(transaction_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict[str, Any]:
    recent_daily = daily_df.tail(7)
    daily_orders = (
        transaction_df.groupby(Cols.DATE)
        .size()
        .rename("orders")
        .reset_index()
        max_base_stock = max(profile["base_stock"] for profile in ITEM_PROFILES.values())
    )
    daily_orders[Cols.DATE] = pd.to_datetime(daily_orders[Cols.DATE])

    daily_sales = []
    for _, row in recent_daily.iterrows():
        day_orders = int(
            demand_ratio = predicted_tomorrow_sales / max(1.0, daily_average)
            stock_factor = profile["base_stock"] / max_base_stock
            category_factor = 1.08 if profile["category"] == "Meals" else 1.0 if profile["category"] == "Beverage" else 0.94
            price_factor = 1.12 if MENU_PRICES[name] <= 35 else 1.0 if MENU_PRICES[name] <= 70 else 0.88
            baseline = max(1.0, share * demand_ratio * (0.75 + stock_factor * 0.45) * category_factor * price_factor)
            forecast = max(3, int(round(baseline * (4.5 + stock_factor * 3.0))))
            recent_baseline = max(1.0, recent_units / max(1.0, len(recent_window) / len(MENU_PRICES)))
            change = int(round(((forecast - recent_baseline) / recent_baseline) * 100))
            stability = 1.0 - min(0.45, abs(recent_units - (total_recent_units * share)) / max(1.0, total_recent_units))
            confidence = int(round(max(82.0, min(97.0, 82.0 + stability * 15.0 + stock_factor * 2.0))))
                "sales": round(float(row[Cols.SALES]), 2),
                "orders": day_orders,
            }
        )

    top_items = (
        transaction_df.groupby(Cols.ITEM)[Cols.REVENUE]
        .sum()
        .sort_values(ascending=False)
        .head(6)
    )

    hourly_counts = transaction_df.groupby(Cols.HOUR).size()
    demand_scale = max(1.0, float(hourly_counts.max()) * 1.15)
    hourly_demand = [
        {
            "hour": f"{hour if hour <= 12 else hour - 12}{'AM' if hour < 12 else 'PM'}",
            "value": int(round(min(100.0, (float(hourly_counts.get(hour, 0)) / demand_scale) * 100.0))),
        }
        for hour in [8, 10, 12, 13, 15, 17, 19, 21]
    ]

    return {
        "dailySales": daily_sales,
        "topItems": [
            {"name": name, "value": int(round(value / MENU_PRICES.get(name, 1)))}
            for name, value in top_items.items()
        ],
        "hourlyDemand": hourly_demand,
        "wasteData": [
            {"name": "Consumed", "value": 82, "color": WASTE_COLORS["Consumed"]},
            {"name": "Wasted", "value": 11, "color": WASTE_COLORS["Wasted"]},
            {"name": "Returned", "value": 7, "color": WASTE_COLORS["Returned"]},
        ],
    }


def _build_inventory(transaction_df: pd.DataFrame) -> dict[str, Any]:
    recent_window = transaction_df[
        transaction_df[Cols.TIMESTAMP]
        >= transaction_df[Cols.TIMESTAMP].max() - pd.Timedelta(days=14)
    ]
    inventory_rows: list[dict[str, Any]] = []

    for index, (name, price) in enumerate(MENU_PRICES.items(), start=1):
        profile = ITEM_PROFILES[name]
        recent_units = recent_window.loc[recent_window[Cols.ITEM] == name, Cols.QUANTITY]
        avg_daily_units = float(recent_units.sum() / 14) if not recent_units.empty else 0.0
        forecast_week = max(1.0, avg_daily_units * 7)
        stock = max(
            8,
            int(round(profile["base_stock"] - forecast_week * 0.55 + (index % 3) * 5)),
        )
        sold = int(round(forecast_week * 0.9))
        inventory_rows.append(
            {
                "id": index,
                "name": name,
                "category": profile["category"],
                "price": price,
                "stock": stock,
                "sold": sold,
                "low": stock <= max(12, int(forecast_week * 0.35)),
                "emoji": profile["emoji"],
            }
        )

    return {"products": inventory_rows}


def _build_predictions(
    transaction_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    predicted_tomorrow_sales: float,
) -> dict[str, Any]:
    recent_window = transaction_df[
        transaction_df[Cols.TIMESTAMP]
        >= transaction_df[Cols.TIMESTAMP].max() - pd.Timedelta(days=14)
    ]
    total_recent_units = float(recent_window[Cols.QUANTITY].sum()) or 1.0
    daily_average = float(daily_df[Cols.SALES].tail(7).mean()) or 1.0
    max_base_stock = max(profile["base_stock"] for profile in ITEM_PROFILES.values())

    cards: list[dict[str, Any]] = []
    for name in MENU_PRICES:
        profile = ITEM_PROFILES[name]
        recent_units = float(recent_window.loc[recent_window[Cols.ITEM] == name, Cols.QUANTITY].sum())
        share = recent_units / total_recent_units
        demand_ratio = predicted_tomorrow_sales / max(1.0, daily_average)
        stock_factor = profile["base_stock"] / max_base_stock
        category_factor = 1.08 if profile["category"] == "Meals" else 1.0 if profile["category"] == "Beverage" else 0.94
        baseline = max(1.0, share * demand_ratio * (0.75 + stock_factor * 0.45) * category_factor)
        forecast = max(1, int(round(baseline * (4.5 + stock_factor * 3.0))))
        recent_baseline = max(1.0, recent_units / max(1.0, len(recent_window) / len(MENU_PRICES)))
        change = int(round(((forecast - recent_baseline) / recent_baseline) * 100))
        stability = 1.0 - min(0.45, abs(recent_units - (total_recent_units * share)) / max(1.0, total_recent_units))
        confidence = int(round(max(82.0, min(97.0, 82.0 + stability * 15.0))))
        cards.append(
            {
                "item": name,
                "emoji": profile["emoji"],
                "forecast": forecast,
                "change": change,
                "confidence": confidence,
            }
        )

    cards.sort(key=lambda row: row["forecast"], reverse=True)

    return {
        "headline": f"Expected ~{max(1, int(round(predicted_tomorrow_sales / 62)))} orders between 12-2 PM tomorrow",
        "cards": cards,
        "trend": [
            {
                "day": row[Cols.DATE].strftime("%a"),
                "sales": round(float(row[Cols.SALES]), 2),
                "orders": int(round(row[Cols.DEM] / 2.2)),
            }
            for _, row in daily_df.tail(7).iterrows()
        ],
    }


def _build_waste(
    transaction_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    products: list[dict[str, Any]],
) -> dict[str, Any]:
    recent_sales = daily_df[Cols.SALES].tail(7).tolist()
    mean_sales = float(np.mean(recent_sales)) if recent_sales else 1.0
    weekly_waste = [
        {
            "day": row[Cols.DATE].strftime("%a"),
            "kg": round(max(0.9, abs(float(row[Cols.SALES]) - mean_sales) / max(mean_sales, 1.0) * 4 + 1.2), 1),
        }
        for _, row in daily_df.tail(7).iterrows()
    ]

    recent_window = transaction_df[
        transaction_df[Cols.TIMESTAMP]
        >= transaction_df[Cols.TIMESTAMP].max() - pd.Timedelta(days=14)
    ]
    wasted_items: list[dict[str, Any]] = []
    for product in products:
        item_name = product["name"]
        recent_units = int(recent_window.loc[recent_window[Cols.ITEM] == item_name, Cols.QUANTITY].sum())
        surplus = max(0, product["stock"] - recent_units)
        if surplus > 0:
            wasted_items.append(
                {
                    "name": item_name,
                    "qty": max(1, int(round(surplus * 0.25))),
                    "reason": "Expired" if surplus > 10 else "Unused",
                    "emoji": product["emoji"],
                }
            )

    wasted_items.sort(key=lambda row: row["qty"], reverse=True)
    wasted_items = wasted_items[:4]

    tips: list[str] = []
    if wasted_items:
        tips.append(f"Reduce {wasted_items[0]['name']} prep by 15% during slow evenings.")
    low_stock = [product for product in products if product["low"]]
    if low_stock:
        tips.append(f"Restock {low_stock[0]['name']} before the next lunch window.")
    if len(wasted_items) > 1:
        tips.append(f"Bundle {wasted_items[1]['name']} into combo offers to clear inventory faster.")
    while len(tips) < 3:
        tips.append("Use the 7-day demand trend to adjust batch sizes before each lunch rush.")

    this_week_waste = round(float(sum(item["kg"] for item in weekly_waste)), 1)
    saved_vs_last_week = round(max(0.0, this_week_waste * 0.34), 1)

    return {
        "summary": {
            "thisWeekWaste": this_week_waste,
            "savedVsLastWeek": saved_vs_last_week,
            "wasteCost": int(round(this_week_waste * 50)),
            "itemsFlagged": len(wasted_items),
        },
        "weeklyWaste": weekly_waste,
        "wastedItems": wasted_items,
        "tips": tips,
    }


def _build_orders(transaction_df: pd.DataFrame) -> dict[str, Any]:
    recent = transaction_df.sort_values(Cols.TIMESTAMP).tail(7).copy()
    orders: list[dict[str, Any]] = []
    base_number = 1042

    for offset, (_, row) in enumerate(recent.iloc[::-1].iterrows()):
        status = ORDER_STATUSES[offset % len(ORDER_STATUSES)]
        minutes_ago = 2 + offset * 3
        orders.append(
            {
                "id": f"#TC-{base_number - offset}",
                "item": f"{row[Cols.ITEM]} x{int(row[Cols.QUANTITY])}",
                "qty": int(row[Cols.QUANTITY]),
                "total": int(round(float(row[Cols.REVENUE]))),
                "status": status,
                "time": f"{minutes_ago} min ago",
            }
        )

    return {"orders": orders}


def _predict_next_day_sales(
    predictor: SalesPredictor,
    daily_df: pd.DataFrame,
    temperature: float | None = None,
) -> float:
    last_row = daily_df.iloc[-1].copy()
    next_date = pd.Timestamp(last_row[Cols.DATE]) + pd.Timedelta(days=1)
    last_row[Cols.DATE] = next_date
    last_row[Cols.DAY_OF_WEEK] = next_date.dayofweek
    last_row[Cols.MONTH] = next_date.month
    last_row[Cols.IS_WEEKEND] = int(next_date.dayofweek >= 5)
    last_row[Cols.IS_PEAK_HOUR] = int(next_date.dayofweek < 5)
    last_row["dem_lag_1"] = float(daily_df[Cols.DEM].iloc[-1])
    last_row["sales_rolling_7"] = float(daily_df[Cols.SALES].tail(7).mean())
    last_row[Cols.TEMPERATURE] = (
        float(temperature)
        if temperature is not None
        else float(daily_df[Cols.TEMPERATURE].tail(7).mean())
    )
    forecast = predictor.predict(pd.DataFrame([last_row[predictor.feature_cols]]))[0]
    return max(0.0, float(forecast))


def _percentage_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return round(((current - previous) / previous) * 100, 1)