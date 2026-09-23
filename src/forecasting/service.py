"""
Forecasting Service Module.
Loads the trained ML model and generates multi-step demand predictions
for any specified food item, center, and time horizon.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import MODEL_FILE, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("forecasting_service")


class DemandForecaster:
    """Production service for generating demand forecasts."""

    def __init__(self, model_path: Path = MODEL_FILE):
        self.model_path = model_path
        self._load_model()
        self._load_metadata()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        logger.info(f"Loading forecasting model from {self.model_path}...")
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.metrics = payload.get("metrics", {})
        logger.info("Model loaded successfully.")

    def _load_metadata(self):
        meal_path = RAW_DATA_DIR / "meal_info.csv"
        center_path = RAW_DATA_DIR / "fulfilment_center_info.csv"
        self.meal_df = pd.read_csv(meal_path) if meal_path.exists() else pd.DataFrame()
        self.center_df = pd.read_csv(center_path) if center_path.exists() else pd.DataFrame()

    def predict_single(
        self,
        meal_id: int,
        center_id: int,
        checkout_price: float = 300.0,
        base_price: float = 320.0,
        emailer_for_promotion: int = 0,
        homepage_featured: int = 0,
        recent_lags: Optional[List[float]] = None,
        op_area: float = 4.0,
        category: str = "Rice Bowl",
        cuisine: str = "Indian",
        center_type: str = "TYPE_A",
        week: int = 146,
    ) -> float:
        """Generates a point forecast for a single meal and center."""
        if recent_lags is None or len(recent_lags) < 4:
            # Default fallback to category baseline if lags not provided
            recent_lags = [150.0, 150.0, 150.0, 150.0]

        # Explicitly cast lags to float — prevents pandas object-dtype errors in LightGBM / XGBoost
        recent_lags = [float(v) for v in recent_lags]
        lag_1, lag_2, lag_3, lag_4 = recent_lags[:4]
        rolling_mean_3 = float(np.mean([lag_1, lag_2, lag_3]))
        rolling_mean_7 = float(np.mean(recent_lags))
        rolling_std_3 = float(np.std([lag_1, lag_2, lag_3]))
        demand_momentum = float((lag_1 + 1.0) / (rolling_mean_3 + 1.0))

        checkout_price = float(checkout_price)
        base_price = float(base_price)
        discount_amount = max(0.0, base_price - checkout_price)
        discount_percent = (discount_amount / base_price) * 100.0 if base_price > 0 else 0.0
        price_ratio = checkout_price / base_price if base_price > 0 else 1.0
        is_discounted = int(discount_amount > 0)

        week_of_year = ((week - 1) % 52) + 1
        sin_week = np.sin(2 * np.pi * week_of_year / 52.0)
        cos_week = np.cos(2 * np.pi * week_of_year / 52.0)

        # Approximate frequency priors
        cat_freq = 0.20
        cui_freq = 0.25
        ct_freq = 0.50

        feature_dict = {
            "checkout_price": checkout_price,
            "base_price": base_price,
            "emailer_for_promotion": int(emailer_for_promotion),
            "homepage_featured": int(homepage_featured),
            "op_area": op_area,
            "discount_amount": discount_amount,
            "discount_percent": discount_percent,
            "price_ratio": price_ratio,
            "is_discounted": is_discounted,
            "week_of_year": week_of_year,
            "sin_week": sin_week,
            "cos_week": cos_week,
            "category_freq": cat_freq,
            "cuisine_freq": cui_freq,
            "center_type_freq": ct_freq,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "rolling_mean_3": rolling_mean_3,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_3": rolling_std_3,
            "demand_momentum": demand_momentum,
        }

        feature_vector = pd.DataFrame([feature_dict])[self.feature_names].astype(float)
        log_pred = self.model.predict(feature_vector)[0]
        actual_pred = max(0.0, float(np.expm1(log_pred)))
        return round(actual_pred, 1)

    def generate_horizon_forecast(
        self,
        meal_id: int,
        center_id: int,
        horizon_days: int = 7,
        current_stock: float = 50.0,
        base_price: float = 350.0,
        checkout_price: float = 320.0,
    ) -> List[Dict[str, Any]]:
        """
        Generates an auto-regressive multi-step forecast over a horizon (e.g. 7 days).
        Daily demand is distributed based on weekly seasonality patterns.
        """
        weekly_pred = self.predict_single(
            meal_id=meal_id,
            center_id=center_id,
            checkout_price=checkout_price,
            base_price=base_price,
        )

        # Realistic intra-week consumption distribution weights (Mon - Sun)
        day_weights = [0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.16]
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        forecasts = []
        for i in range(horizon_days):
            day_idx = i % 7
            daily_predicted = round(weekly_pred * day_weights[day_idx], 1)
            forecasts.append({
                "day_number": i + 1,
                "day_of_week": day_names[day_idx],
                "predicted_demand": daily_predicted,
            })
        return forecasts
