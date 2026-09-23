"""
Inventory Recommendation & Safety Stock Optimization Module.
Translates ML point forecasts into actionable procurement recommendations
to prevent stockouts and minimize food waste.
"""

from typing import Dict, Any
import math

class InventoryOptimizer:
    """Calculates dynamic safety stocks and recommended replenishment quantities."""

    def __init__(self, service_level_z: float = 1.65, default_lead_time_days: int = 1):
        """
        Args:
            service_level_z: Z-score corresponding to desired service level (1.65 = 95% confidence).
            default_lead_time_days: Delivery lead time in days.
        """
        self.service_level_z = service_level_z
        self.default_lead_time_days = default_lead_time_days

    def calculate_safety_stock(self, demand_std: float, lead_time_days: int = 1) -> float:
        """
        Calculates dynamic safety stock using the industry standard inventory formula:
        Safety Stock = Z * sigma_demand * sqrt(Lead Time)
        
        Assumptions:
        - Lead time demand follows a normal distribution around the mean forecast.
        - Z = 1.65 ensures that in 95% of demand cycles, stockouts will not occur.
        """
        lead_time = max(1, lead_time_days)
        safety_stock = self.service_level_z * demand_std * math.sqrt(lead_time)
        return round(max(0.0, safety_stock), 1)

    def generate_recommendation(
        self,
        predicted_demand: float,
        current_stock: float,
        demand_std: float = 15.0,
        lead_time_days: int = 1,
        unit_price: float = 300.0,
        unit_cost: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Generates full inventory analysis and recommended procurement quantity.
        
        Basic Formula:
        Recommended Quantity = max(0, Predicted Demand + Safety Stock - Current Stock)
        """
        safety_stock = self.calculate_safety_stock(demand_std, lead_time_days)
        target_stock = predicted_demand + safety_stock
        raw_recommendation = target_stock - current_stock
        recommended_quantity = max(0.0, raw_recommendation)

        # Risk Analysis
        potential_shortage = max(0.0, predicted_demand - current_stock)
        potential_surplus = max(0.0, current_stock - predicted_demand)
        
        if current_stock < predicted_demand:
            stockout_risk = "HIGH" if current_stock < (predicted_demand * 0.5) else "MODERATE"
        else:
            stockout_risk = "LOW"

        waste_risk = "HIGH" if potential_surplus > (predicted_demand * 0.5) else "LOW"

        # Financial impact estimations
        estimated_lost_revenue = round(potential_shortage * unit_price, 2)
        estimated_waste_cost = round(potential_surplus * unit_cost, 2)

        return {
            "predicted_demand": round(predicted_demand, 1),
            "current_stock": round(current_stock, 1),
            "safety_stock": round(safety_stock, 1),
            "recommended_quantity": round(recommended_quantity, 1),
            "stockout_risk": stockout_risk,
            "waste_risk": waste_risk,
            "potential_shortage": round(potential_shortage, 1),
            "potential_surplus": round(potential_surplus, 1),
            "estimated_lost_revenue": estimated_lost_revenue,
            "estimated_waste_cost": estimated_waste_cost,
        }
