"""
Cost-Aware Business Decision Layer.
Evaluates asymmetric costs between under-prediction (stockout/lost sales)
and over-prediction (food spoilage and raw material loss).
"""

from typing import Dict, Any

def optimize_cost_policy(
    predicted_demand: float,
    unit_selling_price: float,
    unit_raw_cost: float,
    salvage_value: float = 0.0,
    goodwill_penalty: float = 20.0,
) -> Dict[str, Any]:
    """
    Computes optimal procurement buffer using Critical Fractile (Newsvendor Model):
    Critical Ratio = Cu / (Cu + Co)
    Where:
      Cu (Cost of Understocking) = Price - Cost + Goodwill Penalty
      Co (Cost of Overstocking) = Cost - Salvage Value
    """
    cu = max(1.0, (unit_selling_price - unit_raw_cost) + goodwill_penalty)
    co = max(1.0, unit_raw_cost - salvage_value)
    critical_ratio = cu / (cu + co)

    # If critical ratio > 0.5, understocking hurts more than overstocking -> add positive buffer
    buffer_multiplier = (critical_ratio - 0.5) * 2.0  # Normalized between -1.0 and +1.0
    recommended_buffer = round(predicted_demand * 0.10 * buffer_multiplier, 1)
    cost_optimal_demand = max(0.0, round(predicted_demand + recommended_buffer, 1))

    return {
        "cost_of_understocking_cu": round(cu, 2),
        "cost_of_overstocking_co": round(co, 2),
        "critical_ratio": round(critical_ratio, 3),
        "recommended_buffer": recommended_buffer,
        "cost_optimal_demand": cost_optimal_demand,
        "strategy": "Over-order buffer advised (stockouts are costly)" if critical_ratio > 0.5 else "Lean inventory advised (waste is costly)"
    }
