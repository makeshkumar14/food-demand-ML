"""
Generates notebooks/02_model_training_and_evaluation.ipynb.
"""

import json
from pathlib import Path

notebook_path = Path("notebooks/02_model_training_and_evaluation.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 2 & 3: Time-Series Feature Engineering & XGBoost Modeling\n",
            "## Food Demand Forecasting & Inventory Optimization System\n",
            "\n",
            "**Academic Context:** B.Tech CSE Project Evaluation  \n",
            "**Goal:** Build lag features, enforce strict anti-leakage guards, train XGBoost regressor, benchmark against a 7-week moving average, and calculate safety stock."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "sys.path.append('..')\n",
            "from src.config import FEATURE_DATA_PATH, MODEL_FILE\n",
            "from src.features.engineering import split_chronological\n",
            "from src.models.train import train_and_evaluate_models\n",
            "from src.inventory.recommendation import InventoryOptimizer\n",
            "\n",
            "print('Libraries and modules loaded!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Load Processed Feature Matrix & Inspect Chronological Split\n",
            "Notice how Train, Validation, and Test sets are partitioned strictly by week number."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_parquet(FEATURE_DATA_PATH)\n",
            "train_df, val_df, test_df = split_chronological(df)\n",
            "print(f'Train Set: Weeks {train_df[\"week\"].min()} to {train_df[\"week\"].max()} ({len(train_df):,} rows)')\n",
            "print(f'Val Set:   Weeks {val_df[\"week\"].min()} to {val_df[\"week\"].max()} ({len(val_df):,} rows)')\n",
            "print(f'Test Set:  Weeks {test_df[\"week\"].min()} to {test_df[\"week\"].max()} ({len(test_df):,} rows)')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Benchmark Comparison: Moving Average Baseline vs. XGBoost Regressor"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results = train_and_evaluate_models()\n",
            "comp_df = pd.DataFrame([\n",
            "    {'Model': '7-Week Moving Average Baseline', **results['baseline_test']},\n",
            "    {'Model': 'XGBoost Regressor (Engineered Lags)', **results['xgboost_test']},\n",
            "])\n",
            "comp_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Inventory Recommendation & Dynamic Safety Stock Calculation\n",
            "Demonstrating the procurement formula: $\\max(0, \\text{Predicted} + \\text{Safety Stock} - \\text{Current Stock})$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "optimizer = InventoryOptimizer(service_level_z=1.65)\n",
            "rec = optimizer.generate_recommendation(\n",
            "    predicted_demand=85.0,\n",
            "    current_stock=20.0,\n",
            "    demand_std=12.0,\n",
            ")\n",
            "print('Inventory Recommendation Result:')\n",
            "for k, v in rec.items():\n",
            "    print(f'  {k}: {v}')"
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Generated notebook: {notebook_path}")
