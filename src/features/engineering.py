"""
Time-Series Feature Engineering Module.
Constructs lag features and rolling statistics with strict anti-leakage constraints.
Performs chronological train/validation/test splitting.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_WEEK_END, VAL_WEEK_END, TEST_WEEK_END,
    PROCESSED_DATA_DIR, FEATURE_DATA_PATH
)
from src.features.preprocessor import clean_and_derive_features
from src.data.loader import load_raw_datasets, merge_datasets
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs lag and rolling window features per (center_id, meal_id) group.
    
    ANTI-LEAKAGE GUARANTEE:
    All rolling windows operate on the SHIFTED target: orders.shift(1).
    This ensures that target at time t is never present in features computed for time t.
    """
    logger.info("Generating time-series lag and rolling features...")
    df = df.copy()

    # Sort strictly chronologically
    df = df.sort_values(by=["center_id", "meal_id", "week"]).reset_index(drop=True)

    # Grouping key
    group = df.groupby(["center_id", "meal_id"])["num_orders"]

    # 1. Lag Features (Historical demand at t-1, t-2, t-3, t-4)
    logger.info("Computing lag features: lag_1, lag_2, lag_3, lag_4...")
    df["lag_1"] = group.shift(1)
    df["lag_2"] = group.shift(2)
    df["lag_3"] = group.shift(3)
    df["lag_4"] = group.shift(4)

    # 2. Rolling Window Statistics (Strictly shifted to prevent lookahead leakage)
    logger.info("Computing rolling statistics (mean, std) with closed='left' logic...")
    # Shifted series for rolling calculations
    shifted_target = group.shift(1)
    
    df["rolling_mean_3"] = (
        df.groupby(["center_id", "meal_id"])["num_orders"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["rolling_mean_7"] = (
        df.groupby(["center_id", "meal_id"])["num_orders"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )
    df["rolling_std_3"] = (
        df.groupby(["center_id", "meal_id"])["num_orders"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).std())
    ).fillna(0.0)

    # 3. Demand Momentum: Ratio of recent demand (lag_1) to rolling average
    df["demand_momentum"] = (df["lag_1"] + 1.0) / (df["rolling_mean_3"] + 1.0)

    # Drop the initial warmup rows where lag_4 is NaN (first 4 weeks per center-meal pair)
    initial_len = len(df)
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4"]).reset_index(drop=True)
    logger.info(f"Dropped warmup rows with undefined lags: {initial_len - len(df):,} rows removed.")
    logger.info(f"Remaining training matrix shape: {df.shape}")

    return df


def split_chronological(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset strictly chronologically into Train, Validation, and Test sets.
    
    Why Chronological Splitting?
    Random train_test_split leaks future demand data into training, which results
    in artificially inflated validation accuracy and disastrous real-world failure.
    Chronological splitting simulates the exact operational environment.
    """
    logger.info("Splitting dataset chronologically...")
    train_mask = df["week"] <= TRAIN_WEEK_END
    val_mask = (df["week"] > TRAIN_WEEK_END) & (df["week"] <= VAL_WEEK_END)
    test_mask = (df["week"] > VAL_WEEK_END) & (df["week"] <= TEST_WEEK_END)

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    logger.info(f"Train set: Weeks {train_df['week'].min()}-{train_df['week'].max()} ({len(train_df):,} rows)")
    logger.info(f"Val set:   Weeks {val_df['week'].min()}-{val_df['week'].max()} ({len(val_df):,} rows)")
    logger.info(f"Test set:  Weeks {test_df['week'].min()}-{test_df['week'].max()} ({len(test_df):,} rows)")

    return train_df, val_df, test_df


def run_feature_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the end-to-end data preparation and feature engineering pipeline.
    Saves the processed feature table to disk.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_raw, meal_raw, center_raw = load_raw_datasets()
    merged = merge_datasets(train_raw, meal_raw, center_raw)
    cleaned = clean_and_derive_features(merged)
    features_df = create_time_series_features(cleaned)

    # Save to parquet for fast downstream model training
    logger.info(f"Saving processed feature matrix to {FEATURE_DATA_PATH}...")
    features_df.to_parquet(FEATURE_DATA_PATH, index=False)
    logger.info("Feature matrix successfully saved.")

    train_df, val_df, test_df = split_chronological(features_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = run_feature_pipeline()
    print("\n--- FEATURE PIPELINE COMPLETE ---")
    print(f"Features: {list(train_df.columns)}")
