"""FastAPI server for the Thapar canteen dashboard."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard_service import get_dashboard_snapshot, predict_next_day_sales

app = FastAPI(title="Thapar Canteen Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    return get_dashboard_snapshot()


@app.get("/api/predict")
def predict(temperature: float | None = None) -> dict[str, object]:
    return predict_next_day_sales(temperature=temperature)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)