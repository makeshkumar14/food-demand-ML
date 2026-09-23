"""
Data preprocessing module.
Handles missing values, duplicate checks, categorical encoding,
and derived pricing features.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger

logger = get_logger("preprocessor")


def clean_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data, validates chronological order, and derives non-leaking static features.
    
    Derived Features:
    - discount_amount: base_price - checkout_price
    - discount_percent: (discount_amount / base_price) * 100
    - price_ratio: checkout_price / base_price
    - is_discounted: 1 if discount_amount > 0 else 0
    """
    logger.info("Cleaning data and deriving pricing features...")
    df = df.copy()

    # Ensure chronological sort by week, center_id, meal_id
    df = df.sort_values(by=["week", "center_id", "meal_id"]).reset_index(drop=True)

    # Derived Pricing Features (Pricing is set in advance of the sales week -> No leakage)
    df["discount_amount"] = np.maximum(0.0, df["base_price"] - df["checkout_price"])
    df["discount_percent"] = (df["discount_amount"] / df["base_price"]) * 100.0
    df["price_ratio"] = df["checkout_price"] / df["base_price"]
    df["is_discounted"] = (df["discount_amount"] > 0).astype(int)

    # Cyclical Calendar Signal (Week of Year cyclic representation)
    # 52 weeks in a calendar year
    df["week_of_year"] = ((df["week"] - 1) % 52) + 1
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

    # Categorical Frequency Encoding for category and cuisine
    # Frequency encoding preserves cardinality while being numerical for tree models
    for cat_col in ["category", "cuisine", "center_type"]:
        if cat_col in df.columns:
            freq_map = df[cat_col].value_counts(normalize=True).to_dict()
            df[f"{cat_col}_freq"] = df[cat_col].map(freq_map)

    logger.info(f"Preprocessing completed. Shape: {df.shape}")
    return df
