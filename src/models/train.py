"""
Model Training and Evaluation Pipeline.
Trains a LightGBM regressor with log-transformed targets,
benchmarks against the 7-week moving average baseline,
computes production metrics, and persists artifacts.
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    FEATURE_DATA_PATH, MODEL_FILE, MODEL_METRICS_FILE,
    FIGURES_DIR, MODELS_DIR
)
from src.features.engineering import split_chronological, run_feature_pipeline
from src.models.baseline import evaluate_baseline, compute_metrics
from src.utils.logger import get_logger

logger = get_logger("model_training")

# Core predictive features (Strictly available prior to the order period)
FEATURE_COLUMNS = [
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
    "op_area",
    "discount_amount",
    "discount_percent",
    "price_ratio",
    "is_discounted",
    "week_of_year",
    "sin_week",
    "cos_week",
    "category_freq",
    "cuisine_freq",
    "center_type_freq",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_3",
    "demand_momentum",
]


def train_and_evaluate_models() -> dict:
    """
    Executes the training workflow:
    1. Loads features
    2. Runs Baseline Benchmark
    3. Trains LightGBM Regressor
    4. Computes test metrics & comparison table
    5. Saves model and visualizations
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load feature data
    if FEATURE_DATA_PATH.exists():
        logger.info(f"Loading existing features from {FEATURE_DATA_PATH}...")
        df = pd.read_parquet(FEATURE_DATA_PATH)
        train_df, val_df, test_df = split_chronological(df)
    else:
        logger.info("Feature dataset not found on disk. Running feature pipeline...")
        train_df, val_df, test_df = run_feature_pipeline()

    # 2. Evaluate Baseline
    baseline_results = evaluate_baseline(val_df, test_df)

    # 3. Prepare Feature Matrices and Log-Transformed Targets
    logger.info("Preparing feature matrices for LightGBM...")
    X_train = train_df[FEATURE_COLUMNS]
    y_train_log = np.log1p(train_df["num_orders"].values)

    X_val = val_df[FEATURE_COLUMNS]
    y_val_log = np.log1p(val_df["num_orders"].values)

    X_test = test_df[FEATURE_COLUMNS]
    y_test_actual = test_df["num_orders"].values

        # 4. Instantiate and Train LightGBM Regressor
    logger.info("Training LightGBM Regressor with early stopping...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=25, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info(f"Training completed. Best iteration: {model.best_iteration_}")

    # 5. Invert Predictions to Original Unit Scale
    logger.info("Generating predictions and converting back to natural demand scale...")
    val_pred_log = model.predict(X_val)
    val_pred = np.maximum(0.0, np.expm1(val_pred_log))
    val_metrics = compute_metrics(val_df["num_orders"].values, val_pred)

    test_pred_log = model.predict(X_test)
    test_pred = np.maximum(0.0, np.expm1(test_pred_log))
    test_metrics = compute_metrics(y_test_actual, test_pred)

    logger.info(f"LightGBM Test Metrics: MAE={test_metrics['MAE']}, RMSE={test_metrics['RMSE']}, MAPE={test_metrics['MAPE']}%")

    # 6. Save Trained Model Artifact
    logger.info(f"Persisting model artifact to {MODEL_FILE}...")
    model_payload = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "metrics": test_metrics,
        "best_iteration": int(model.best_iteration_),
    }
    joblib.dump(model_payload, MODEL_FILE)
    logger.info("Model artifact successfully saved.")

    # 7. Comparison Summary Table
    comparison = {
        "model_features": FEATURE_COLUMNS,
        "baseline_test": baseline_results["test"],
        "xgboost_test": test_metrics,
        "improvement_pct": {
            "MAE_reduction": round(((baseline_results["test"]["MAE"] - test_metrics["MAE"]) / baseline_results["test"]["MAE"]) * 100, 2),
            "RMSE_reduction": round(((baseline_results["test"]["RMSE"] - test_metrics["RMSE"]) / baseline_results["test"]["RMSE"]) * 100, 2),
            "MAPE_reduction": round(((baseline_results["test"]["MAPE"] - test_metrics["MAPE"]) / baseline_results["test"]["MAPE"]) * 100, 2),
        }
    }

    with open(MODEL_METRICS_FILE, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved evaluation metrics to {MODEL_METRICS_FILE}")

    # 8. Export Visualizations: Feature Importance & Actual vs Predicted
    logger.info("Exporting feature importance plot...")
    plt.figure(figsize=(10, 7), dpi=150)
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", palette="mako")
    plt.title("Top 15 Feature Importances (LightGBM)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Relative Feature Importance (Gain)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "7_feature_importance.png")
    plt.close()

    logger.info("Exporting Actual vs Predicted plot...")
    plt.figure(figsize=(10, 5), dpi=150)
    sample_indices = np.random.RandomState(42).choice(len(y_test_actual), size=min(100, len(y_test_actual)), replace=False)
    sample_indices = np.sort(sample_indices)
    plt.plot(range(len(sample_indices)), y_test_actual[sample_indices], label="Actual Demand", color="#1f77b4", marker="o", markersize=4, lw=1.5)
    plt.plot(range(len(sample_indices)), test_pred[sample_indices], label="LightGBM Predicted", color="#ff7f0e", marker="x", markersize=4, linestyle="--", lw=1.5)
    plt.title("Actual vs. Predicted Demand (Held-Out Test Sample)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("Food Orders", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "8_actual_vs_predicted.png")
    plt.close()

    logger.info("Model pipeline execution finished successfully.")
    return comparison


if __name__ == "__main__":
    results = train_and_evaluate_models()
    print("\n==========================================")
    print("      FINAL MODEL COMPARISON RESULTS      ")
    print("==========================================")
    print(f"Baseline Test MAE:  {results['baseline_test']['MAE']}")
    print(f"LightGBM Test MAE:   {results['xgboost_test']['MAE']} (Reduced by {results['improvement_pct']['MAE_reduction']}%)")
    print(f"Baseline Test RMSE:  {results['baseline_test']['RMSE']}")
    print(f"LightGBM Test RMSE:  {results['xgboost_test']['RMSE']} (Reduced by {results['improvement_pct']['RMSE_reduction']}%)")
    print(f"Baseline Test MAPE:  {results['baseline_test']['MAPE']}%")
    print(f"LightGBM Test MAPE:  {results['xgboost_test']['MAPE']}% (Reduced by {results['improvement_pct']['MAPE_reduction']}%)")
    print("==========================================")
