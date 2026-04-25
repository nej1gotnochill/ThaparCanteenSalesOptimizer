# Thapar Canteen Sales Optimizer

A clean, modular Python pipeline that analyses campus canteen sales data,
engineers time-based features, trains a **Gradient Boosting Regressor** to
predict daily revenue, and produces publication-quality diagnostic charts.

---

## What the project does

| Step | What happens |
|------|-------------|
| **Load** | Generates a 90-day synthetic transaction log (CSV) on first run; loads and cleans it on every subsequent run |
| **EDA** | Prints top items / peak hours and saves a 2×2 dashboard chart |
| **Features** | Collapses transactions to one row per day; adds lag, rolling mean, and calendar features |
| **Split** | Chronological 80/20 train–test split (no data leakage) |
| **Train** | Fits a `GradientBoostingRegressor` (swap for any sklearn model in one line) |
| **Evaluate** | Reports MAE, MSE, RMSE, and R² on the held-out test set |
| **Visualise** | Saves sales-over-time, actual-vs-predicted, residuals, and feature-importance charts |

---

## Project structure

```
ThaparCanteenSalesOptimizer/
├── main.py            # Orchestrates the full pipeline
├── config.py          # All constants (paths, column names, model params)
├── data_loader.py     # Data generation and CSV loading
├── features.py        # Feature engineering (lag, rolling, calendar flags)
├── model.py           # SalesPredictor class + time-aware split
├── visualization.py   # All chart functions
├── requirements.txt
├── README.md
├── data/
│   └── canteen_sales.csv   # Auto-generated on first run
└── outputs/                # All saved charts land here
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/nej1gotnochill/ThaparCanteenSalesOptimizer.git
cd ThaparCanteenSalesOptimizer

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the pipeline

```bash
python main.py
```

On the very first run the script generates `data/canteen_sales.csv`
automatically — no manual data preparation needed.

All charts are saved to the `outputs/` folder.

## Running the backend API

```bash
python api_server.py
```

Health endpoint: http://127.0.0.1:8000/api/health

Dashboard endpoint: http://127.0.0.1:8000/api/dashboard

---

## Frontend dashboard

The frontend for this optimizer is Canteen Vision, and it lives in the workspace here:

- Frontend folder: [canteen-vision](../canteen-vision)
- App router: [src/router.tsx](../canteen-vision/src/router.tsx)
- Landing page: [src/routes/index.tsx](../canteen-vision/src/routes/index.tsx)
- Dashboard overview: [src/routes/dashboard.index.tsx](../canteen-vision/src/routes/dashboard.index.tsx)
- Analytics page: [src/routes/dashboard.analytics.tsx](../canteen-vision/src/routes/dashboard.analytics.tsx)
- Predictions page: [src/routes/dashboard.predictions.tsx](../canteen-vision/src/routes/dashboard.predictions.tsx)
- Root layout: [src/routes/__root.tsx](../canteen-vision/src/routes/__root.tsx)

### Run the frontend locally

```bash
cd ../canteen-vision
npm install
npm run dev
```

Default local URL (Vite): http://localhost:8080

The frontend reads the backend URL from `VITE_CANTEEN_API_URL` and defaults to `http://127.0.0.1:8000`.

---

## One-command local run

From this repository root, run:

```powershell
./scripts/start-integrated.ps1
```

This starts the Python API and the React dashboard together.

---

## Vercel deployment

Deploy only the `canteen-vision` frontend repo on Vercel.

The Python backend in `ThaparCanteenSalesOptimizer` should be deployed on a Python-friendly host such as Render, Railway, Fly.io, or a VPS, then point `VITE_CANTEEN_API_URL` at that hosted backend.

If you want to jump straight to the app files in this workspace, open [canteen-vision](../canteen-vision) and start with [src/router.tsx](../canteen-vision/src/router.tsx).

---

## Swapping the model

Open `main.py`, find `step_train`, and replace the constructor argument:

```python
# Default — Gradient Boosting
predictor = SalesPredictor()

# Switch to Random Forest (example)
from sklearn.ensemble import RandomForestRegressor
predictor = SalesPredictor(RandomForestRegressor(n_estimators=200))

# Switch to Ridge Regression (example)
from sklearn.linear_model import Ridge
predictor = SalesPredictor(Ridge(alpha=1.0))
```

Everything else — split, evaluate, charts — works unchanged.

---

## Output charts

| File | Description |
|------|-------------|
| `eda_dashboard.png` | 2×2 EDA overview (heatmap, top items, box plot, hourly bar) |
| `sales_over_time.png` | Daily revenue line chart with 7-day rolling average |
| `actual_vs_predicted.png` | Scatter + time-aligned line comparison on test set |
| `residuals.png` | Residuals histogram and residuals-vs-predicted scatter |
| `feature_importances.png` | Bar chart of GBR feature importances |

---

## Model details

**Gradient Boosting Regressor** (`sklearn.ensemble.GradientBoostingRegressor`)

- Builds 100 shallow decision trees sequentially; each tree corrects the
  errors of the previous one.
- Can capture non-linear relationships between demand and revenue — unlike
  plain Linear Regression.
- Key hyperparameters (all configurable in `config.py`):

  | Parameter | Value | Effect |
  |-----------|-------|--------|
  | `n_estimators` | 100 | Number of trees |
  | `learning_rate` | 0.1 | Step size per tree (smaller = more careful) |
  | `max_depth` | 3 | Max depth per tree (shallow = less overfitting) |

---

## Features used

| Feature | Description |
|---------|-------------|
| `DEM` | Total units sold that day |
| `temperature` | Daily temperature (°C) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `month` | Calendar month (1–12) |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_peak_hour` | 1 if a weekday (Mon–Fri) |
| `dem_lag_1` | Yesterday's demand |
| `sales_rolling_7` | 7-day rolling mean of sales (trend) |

---

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib, seaborn
