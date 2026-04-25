"""
config.py
=========
Central configuration for the Thapar Canteen Sales Optimizer.

Change values here — nowhere else — to reconfigure the pipeline.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"

RAW_CSV_PATH: Path = DATA_DIR / "canteen_sales.csv"

# ── Column names ──────────────────────────────────────────────────────
class Cols:
    TIMESTAMP    = "timestamp"
    DATE         = "date"
    HOUR         = "hour"
    DAY_OF_WEEK  = "day_of_week"
    ITEM         = "item"
    QUANTITY     = "quantity"
    PRICE        = "price"
    REVENUE      = "revenue"
    STUDENT_TYPE = "student_type"
    TEMPERATURE  = "temperature"
    # Engineered columns
    IS_WEEKEND   = "is_weekend"
    IS_PEAK_HOUR = "is_peak_hour"
    MONTH        = "month"
    # Daily aggregates used by the model
    DEM          = "DEM"
    SALES        = "sales"

# ── Data-generation settings ──────────────────────────────────────────
SIMULATION_DAYS: int = 90
RANDOM_SEED: int = 42

MENU_PRICES: dict[str, int] = {
    "Samosa":      20,
    "Chai":        10,
    "Paratha":     30,
    "Sandwich":    50,
    "Paneer Roll": 40,
    "Cold Coffee": 25,
    "Juice":       15,
    "Patties":     30,
    "Energy Drink":20,
}

STUDENT_TYPES: list[str] = ["Undergrad", "Graduate"]

# Working hours (canteen is open 8 am–4 pm)
CANTEEN_OPEN_HOUR:  int = 8
CANTEEN_CLOSE_HOUR: int = 17   # exclusive upper bound for range()

PEAK_HOURS: list[int] = [9, 10, 12, 13]
PEAK_TRANSACTIONS:     int = 15
OFF_PEAK_TRANSACTIONS: int = 5

# ── Train / test split ────────────────────────────────────────────────
TEST_SIZE:    float = 0.20   # 20 % of daily rows go to the test set
SPLIT_SEED:   int   = 42

# ── Model hyperparameters (Gradient Boosting Regressor) ───────────────
GBR_PARAMS: dict = {
    "n_estimators":  100,
    "learning_rate": 0.1,
    "max_depth":     3,
    "random_state":  RANDOM_SEED,
}

# ── Visualisation ─────────────────────────────────────────────────────
FIGURE_DPI:  int = 150
STYLE:       str = "whitegrid"
