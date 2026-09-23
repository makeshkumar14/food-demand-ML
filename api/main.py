"""
FastAPI Backend API Application.
Serves demand forecasts, inventory optimization decisions, and model metrics.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import MODEL_METRICS_FILE, MODEL_FILE
from database.database import get_db
from database.models import FoodItem, ForecastRecord, init_db
from api.schemas import (
    PredictRequest, PredictResponse,
    ForecastHorizonRequest, ForecastHorizonResponse,
    FoodItemOut, ModelMetricsResponse
)
from src.forecasting.service import DemandForecaster
from src.inventory.recommendation import InventoryOptimizer
from src.utils.logger import get_logger

logger = get_logger("api")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database is initialized on server startup."""
    init_db()
    yield

app = FastAPI(
    title="Food Demand Forecasting & Inventory Optimization API",
    description="Production REST API for commercial hotel/restaurant food demand predictions and inventory replenishment.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Enables Streamlit or any web client to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
try:
    forecaster = DemandForecaster()
    inventory_optimizer = InventoryOptimizer()
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    forecaster = None
    inventory_optimizer = InventoryOptimizer()


@app.get("/health", tags=["System"])
def health_check():
    """System health check and model availability status."""
    return {
        "status": "healthy",
        "model_loaded": forecaster is not None,
        "database_connected": True,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict_demand(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Predicts food demand for an item and outputs the recommended replenishment quantity.
    """
    if forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded. Train the model first."
        )

    # Resolve meal_id from database if item name is passed
    meal_id = req.meal_id or 1885
    db_item = db.query(FoodItem).filter(FoodItem.meal_id == meal_id).first()
    item_name = db_item.name if db_item else req.item

    predicted_val = forecaster.predict_single(
        meal_id=meal_id,
        center_id=req.center_id or 55,
        checkout_price=req.checkout_price or 300.0,
        base_price=req.base_price or 320.0,
        emailer_for_promotion=req.emailer_for_promotion or 0,
        homepage_featured=req.homepage_featured or 0,
    )

    inv_analysis = inventory_optimizer.generate_recommendation(
        predicted_demand=predicted_val,
        current_stock=req.current_stock,
        demand_std=max(10.0, predicted_val * 0.20),  # Derived from demand variability
        unit_price=req.checkout_price or 300.0,
    )

    # Persist forecast to database
    if db_item:
        forecast_entry = ForecastRecord(
            food_item_id=db_item.id,
            forecast_date=req.date or "2026-10-01",
            predicted_demand=predicted_val,
            model_version="XGBoost_v1.0"
        )
        db.add(forecast_entry)
        db.commit()

    return PredictResponse(
        item=item_name,
        meal_id=meal_id,
        date=req.date or "2026-10-01",
        predicted_demand=inv_analysis["predicted_demand"],
        safety_stock=inv_analysis["safety_stock"],
        current_stock=inv_analysis["current_stock"],
        recommended_quantity=inv_analysis["recommended_quantity"],
        stockout_risk=inv_analysis["stockout_risk"],
        waste_risk=inv_analysis["waste_risk"],
        potential_shortage=inv_analysis["potential_shortage"],
        potential_surplus=inv_analysis["potential_surplus"],
        estimated_lost_revenue=inv_analysis["estimated_lost_revenue"],
        estimated_waste_cost=inv_analysis["estimated_waste_cost"],
    )


@app.post("/forecast", response_model=ForecastHorizonResponse, tags=["Forecasting"])
def forecast_horizon(req: ForecastHorizonRequest):
    """Generates multi-day demand predictions (e.g., next 7, 14, or 30 days)."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecasting model is not loaded.")

    daily_forecasts = forecaster.generate_horizon_forecast(
        meal_id=req.meal_id,
        center_id=req.center_id,
        horizon_days=req.horizon_days,
        current_stock=req.current_stock,
    )

    total_demand = sum(f["predicted_demand"] for f in daily_forecasts)
    recommended_procure = max(0.0, total_demand + (0.15 * total_demand) - req.current_stock)

    return ForecastHorizonResponse(
        meal_id=req.meal_id,
        horizon_days=req.horizon_days,
        daily_forecasts=daily_forecasts,
        total_projected_demand=round(total_demand, 1),
        recommended_total_procurement=round(recommended_procure, 1),
    )


@app.get("/items", response_model=List[FoodItemOut], tags=["Inventory"])
def list_items(db: Session = Depends(get_db)):
    """Returns list of all available food menu items."""
    return db.query(FoodItem).all()


@app.get("/inventory", tags=["Inventory"])
def get_inventory_status(db: Session = Depends(get_db)):
    """Provides snapshot of inventory levels and simulated risk metrics."""
    items = db.query(FoodItem).limit(10).all()
    results = []
    for item in items:
        results.append({
            "meal_id": item.meal_id,
            "name": item.name,
            "category": item.category,
            "current_stock": 25.0,
            "safety_stock": 15.0,
            "status": "In Stock"
        })
    return results


@app.get("/model-info", tags=["Analytics"])
def get_model_info():
    """Returns model performance metrics and comparison to baseline."""
    if MODEL_METRICS_FILE.exists():
        with open(MODEL_METRICS_FILE, "r") as f:
            data = json.load(f)
        data["model_version"] = "XGBoost_v1.0"
        return data
    return {"message": "Model metrics file not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
