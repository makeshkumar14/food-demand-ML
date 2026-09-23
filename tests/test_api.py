"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_contract():
    """Verify POST /predict returns valid response matching schema."""
    payload = {
        "item": "Chicken Biryani",
        "meal_id": 1885,
        "center_id": 55,
        "date": "2026-10-01",
        "current_stock": 20.0,
        "checkout_price": 280.0,
        "base_price": 310.0,
        "emailer_for_promotion": 0,
        "homepage_featured": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_demand" in res
    assert "recommended_quantity" in res
    assert "safety_stock" in res
    assert res["recommended_quantity"] >= 0.0


def test_forecast_horizon_endpoint():
    """Verify POST /forecast returns 7-day breakdown."""
    payload = {
        "meal_id": 1885,
        "center_id": 55,
        "horizon_days": 7,
        "current_stock": 30.0
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert len(res["daily_forecasts"]) == 7
    assert res["total_projected_demand"] > 0.0


def test_items_list():
    """Verify GET /items returns non-empty food catalog."""
    response = client.get("/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
