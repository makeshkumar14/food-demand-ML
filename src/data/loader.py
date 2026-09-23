"""
Data loader module for Food Demand Forecasting.
Loads raw files and performs consistent joining of meal and center metadata.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_CSV, MEAL_INFO_CSV, FULFILMENT_CENTER_CSV
from src.utils.logger import get_logger

logger = get_logger("data_loader")


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV files from data/raw/.
    
    Returns:
        Tuple containing (train_df, meal_info_df, center_info_df)
    """
    logger.info("Loading raw datasets from disk...")
    train_df = pd.read_csv(TRAIN_CSV)
    meal_df = pd.read_csv(MEAL_INFO_CSV)
    center_df = pd.read_csv(FULFILMENT_CENTER_CSV)
    
    logger.info(f"Loaded train.csv with shape: {train_df.shape}")
    logger.info(f"Loaded meal_info.csv with shape: {meal_df.shape}")
    logger.info(f"Loaded fulfilment_center_info.csv with shape: {center_df.shape}")
    return train_df, meal_df, center_df


def merge_datasets(
    train_df: pd.DataFrame,
    meal_df: pd.DataFrame,
    center_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges train orders with meal metadata and fulfillment center metadata.
    Uses LEFT JOIN so that no order records are inadvertently dropped.
    
    Returns:
        Merged unified DataFrame.
    """
    logger.info("Merging datasets on 'meal_id' and 'center_id'...")
    # Merge meal details
    merged_df = train_df.merge(meal_df, on="meal_id", how="left")
    # Merge fulfillment center details
    merged_df = merged_df.merge(center_df, on="center_id", how="left")
    
    logger.info(f"Successfully merged datasets. Total rows: {len(merged_df):,}, columns: {len(merged_df.columns)}")
    return merged_df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key statistical and data quality metrics for reporting/EDA.
    """
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "num_duplicates": int(df.duplicated().sum()),
        "num_unique_meals": int(df["meal_id"].nunique()) if "meal_id" in df.columns else 0,
        "num_unique_centers": int(df["center_id"].nunique()) if "center_id" in df.columns else 0,
        "min_week": int(df["week"].min()) if "week" in df.columns else 0,
        "max_week": int(df["week"].max()) if "week" in df.columns else 0,
        "total_orders": int(df["num_orders"].sum()) if "num_orders" in df.columns else 0,
    }
    return summary


if __name__ == "__main__":
    train_df, meal_df, center_df = load_raw_datasets()
    merged_df = merge_datasets(train_df, meal_df, center_df)
    summary = get_dataset_summary(merged_df)
    print("\n--- DATASET OVERVIEW SUMMARY ---")
    for k, v in summary.items():
        if k != "missing_values":
            print(f"{k}: {v}")
        else:
            total_missing = sum(v.values())
            print(f"total_missing_values: {total_missing}")
