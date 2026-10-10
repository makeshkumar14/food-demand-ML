"""
Pydantic Request & Response Data Contracts for FastAPI Backend.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """Input payload for single point demand forecasting."""
    item: str = Field(default="Thai Beverages #1885", description="Food item name or identifier")
    meal_id: Optional[int] = Field(default=1885, description="Meal ID")
    center_id: Optional[int] = Field(default=55, description="Fulfillment Center ID")
    date: Optional[str] = Field(default="2026-10-01", description="Prediction target date")
    current_stock: float = Field(default=20.0, ge=0.0, description="Current physical inventory in portions")
    checkout_price: Optional[float] = Field(default=300.0, gt=0.0, description="Customer selling price")
    base_price: Optional[float] = Field(default=320.0, gt=0.0, description="Catalog base price")
    emailer_for_promotion: Optional[int] = Field(default=0, ge=0, le=1)
    homepage_featured: Optional[int] = Field(default=0, ge=0, le=1)


class PredictResponse(BaseModel):
    """Output payload for demand forecast and inventory recommendation."""
    item: str
    meal_id: int
    date: str
    predicted_demand: float
    safety_stock: float
    current_stock: float
    recommended_quantity: float
    stockout_risk: str
    waste_risk: str
    potential_shortage: float
    potential_surplus: float
    estimated_lost_revenue: float
    estimated_waste_cost: float


class ForecastHorizonRequest(BaseModel):
    """Request for multi-day horizon forecast (e.g. 7 days)."""
    meal_id: int = Field(default=1885)
    center_id: int = Field(default=55)
    horizon_days: int = Field(default=7, ge=1, le=30)
    current_stock: float = Field(default=50.0, ge=0.0)


class DailyForecastItem(BaseModel):
    day_number: int
    day_of_week: str
    predicted_demand: float


class ForecastHorizonResponse(BaseModel):
    meal_id: int
    horizon_days: int
    daily_forecasts: List[DailyForecastItem]
    total_projected_demand: float
    recommended_total_procurement: float


class FoodItemOut(BaseModel):
    id: int
    meal_id: int
    name: str
    category: str
    cuisine: str
    base_price: float

    model_config = ConfigDict(from_attributes=True)


class ModelMetricsResponse(BaseModel):
    baseline_test: Dict[str, float]
    lightgbm_test: Dict[str, float]
    improvement_pct: Dict[str, float]
    model_version: str
