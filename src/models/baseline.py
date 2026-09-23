"""
Baseline Forecasting Model Module.
Implements a 7-week Moving Average / Last Observed Value baseline forecaster.
Serves as the benchmark to determine if advanced ML models add real predictive value.
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger

logger = get_logger("baseline_model")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and MAPE on original demand scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))  # Demand cannot be negative

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Avoid zero division in MAPE with epsilon
    epsilon = 1.0
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


def evaluate_baseline(val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Evaluates baseline predictions:
    1. Last-week demand (lag_1)
    2. 7-week Moving Average (rolling_mean_7)
    """
    logger.info("Evaluating Moving Average Baseline model on Validation and Test sets...")
    
    # 7-week moving average prediction
    val_pred_ma = val_df["rolling_mean_7"].values
    test_pred_ma = test_df["rolling_mean_7"].values

    val_metrics = compute_metrics(val_df["num_orders"].values, val_pred_ma)
    test_metrics = compute_metrics(test_df["num_orders"].values, test_pred_ma)

    logger.info(f"Baseline Validation Metrics (7-week MA): MAE={val_metrics['MAE']}, RMSE={val_metrics['RMSE']}, MAPE={val_metrics['MAPE']}%")
    logger.info(f"Baseline Test Metrics (7-week MA):       MAE={test_metrics['MAE']}, RMSE={test_metrics['RMSE']}, MAPE={test_metrics['MAPE']}%")

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }
