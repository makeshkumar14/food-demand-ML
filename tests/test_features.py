"""
Unit tests for Time-Series Preprocessing & Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features.preprocessor import clean_and_derive_features
from src.features.engineering import create_time_series_features


def test_pricing_features_calculation():
    """Verify discount_amount and discount_percent logic."""
    raw_sample = pd.DataFrame([{
        "week": 1,
        "center_id": 10,
        "meal_id": 1001,
        "checkout_price": 250.0,
        "base_price": 300.0,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
        "category": "Biryani",
        "cuisine": "Indian",
        "center_type": "TYPE_A"
    }])
    cleaned = clean_and_derive_features(raw_sample)
    assert cleaned.loc[0, "discount_amount"] == 50.0
    assert pytest.approx(cleaned.loc[0, "discount_percent"], 0.1) == 16.67
    assert cleaned.loc[0, "is_discounted"] == 1


def test_lag_features_strict_anti_leakage():
    """Verify lag_1 is strictly t-1 demand and not current week t demand."""
    sample = pd.DataFrame([
        {"week": 1, "center_id": 1, "meal_id": 101, "num_orders": 100},
        {"week": 2, "center_id": 1, "meal_id": 101, "num_orders": 120},
        {"week": 3, "center_id": 1, "meal_id": 101, "num_orders": 150},
        {"week": 4, "center_id": 1, "meal_id": 101, "num_orders": 180},
        {"week": 5, "center_id": 1, "meal_id": 101, "num_orders": 200},
    ])
    fe = create_time_series_features(sample)
    # Week 5 row should have lag_1 = 180 (demand at week 4), NOT 200
    w5 = fe[fe["week"] == 5].iloc[0]
    assert w5["lag_1"] == 180.0
    assert w5["lag_2"] == 150.0
    assert w5["lag_3"] == 120.0
    assert w5["lag_4"] == 100.0
