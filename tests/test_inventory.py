"""
Unit tests for Inventory Recommendation & Safety Stock Logic.
"""

import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.inventory.recommendation import InventoryOptimizer


def test_safety_stock_calculation():
    """Verify safety stock formula: Z * sigma * sqrt(L)."""
    optimizer = InventoryOptimizer(service_level_z=1.65)
    # sigma = 10, L = 1 -> 1.65 * 10 * 1 = 16.5
    safety_stock = optimizer.calculate_safety_stock(demand_std=10.0, lead_time_days=1)
    assert pytest.approx(safety_stock, 0.1) == 16.5


def test_recommendation_non_negative_when_current_stock_is_excess():
    """Ensure recommended quantity NEVER becomes negative (Requirement #27)."""
    optimizer = InventoryOptimizer(service_level_z=1.65)
    # Predicted = 50, Safety = 10, Current Stock = 100 -> 50 + 10 - 100 = -40 -> Should clamp to 0.0
    res = optimizer.generate_recommendation(
        predicted_demand=50.0,
        current_stock=100.0,
        demand_std=5.0,
    )
    assert res["recommended_quantity"] == 0.0
    assert res["potential_surplus"] == 50.0
    assert res["stockout_risk"] == "LOW"


def test_recommendation_positive_shortage():
    """Verify positive procurement recommendation during inventory shortage."""
    optimizer = InventoryOptimizer(service_level_z=1.65)
    # Predicted = 85, Current = 20, Safety ~ 10 -> Expected ~ 75
    res = optimizer.generate_recommendation(
        predicted_demand=85.0,
        current_stock=20.0,
        demand_std=6.06,  # 1.65 * 6.06 ~ 10.0
    )
    assert res["recommended_quantity"] >= 74.0
    assert res["potential_shortage"] == 65.0
    assert res["stockout_risk"] in ["HIGH", "MODERATE"]
