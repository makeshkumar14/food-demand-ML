"""
Configuration module for Food Demand Forecasting & Inventory Optimization System.
Centralizes paths, hyperparameters, column names, and environment variables.
"""

from pathlib import Path
import os

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# File paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
MEAL_INFO_CSV = RAW_DATA_DIR / "meal_info.csv"
FULFILMENT_CENTER_CSV = RAW_DATA_DIR / "fulfilment_center_info.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.parquet"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "features_data.parquet"

MODEL_FILE = MODELS_DIR / "xgboost_demand_model.joblib"
BASELINE_METRICS_FILE = MODELS_DIR / "baseline_metrics.json"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/food_demand.db")

# Machine Learning Target and Grouping
TARGET_COL = "num_orders"
TIME_COL = "week"
ITEM_COL = "meal_id"
CENTER_COL = "center_id"

# Time-series Split Configuration
# Analytics Vidhya dataset spans 145 weeks.
# Train: Weeks 1 to 125
# Validation: Weeks 126 to 135
# Test: Weeks 136 to 145
TRAIN_WEEK_END = 125
VAL_WEEK_END = 135
TEST_WEEK_END = 145

# Business Inventory Policy Defaults
SERVICE_LEVEL_Z = 1.65  # 95% service level for normal distribution safety stock
DEFAULT_LEAD_TIME_WEEKS = 1
