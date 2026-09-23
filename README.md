# 🍲 Food Demand Forecasting & Inventory Optimization System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-success.svg)]()

> A complete, production-grade Machine Learning and Software Engineering application designed for commercial restaurants and hotel kitchens to predict multi-horizon dish demand and optimize raw material replenishment.

---

## 📌 1. Project Overview & Business Problem

Commercial food businesses and kitchens face a daily dilemma:
- **Over-preparation / Over-procurement:** Unsold fresh food spoils, raw ingredients expire, cold-storage expenses increase, and net margins are lost.
- **Under-preparation / Stockout:** Customers cannot order popular menu items, direct revenue is lost, kitchen staff face operational bottlenecks, and customer brand loyalty drops.

### The Solution:
This system replaces guesswork with an **intelligent, data-driven forecasting and inventory replenishment engine**. It combines:
1. **Multi-Horizon Demand Forecasting**: Machine learning models predicting meal consumption 1 to 30 days ahead based on calendar cycles, promotional discounts, and lagged demand momentum.
2. **Dynamic Safety Stock Optimization**: Replenishment buffers mathematically calculated using demand volatility ($\sigma$) and service-level targets (95% non-stockout confidence).
3. **Decoupled Production Architecture**: A FastAPI REST API backend and an interactive Streamlit manager portal connected to a persistent relational database.

---

## 🏗️ 2. System Architecture

```
                  ┌──────────────────────────────────────────┐
                  │       Streamlit Interactive Portal       │
                  │ (Manager Dashboard, Charts, KPI Cards)   │
                  └────────────────────┬─────────────────────┘
                                       │ HTTP / REST
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │           FastAPI REST Backend           │
                  │  (/health, /predict, /forecast, /items)  │
                  └────────────────────┬─────────────────────┘
                                       │
                  ┌────────────────────┼─────────────────────┐
                  ▼                                          ▼
     ┌────────────────────────┐                 ┌────────────────────────┐
     │  Forecasting Service   │                 │   Relational Database  │
     │   (Inference Engine)   │                 │ (PostgreSQL / SQLite)  │
     └───────────┬────────────┘                 │   - food_items         │
                 ▼                              │   - forecasts          │
     ┌────────────────────────┐                 │   - sales & inventory  │
     │ Trained Model Artifact │                 └────────────────────────┘
     │ (xgboost_model.joblib) │
     └────────────────────────┘
```

---

## 🔬 3. Machine Learning Pipeline & Anti-Leakage Methodology

### A. Data Ingestion & Relational Merging
The system ingests the official benchmark **Food Demand Forecasting Challenge** dataset ($456,548$ order transactions across $145$ chronological weeks, $51$ distinct meals, and $77$ fulfillment centers) and performs relational Left-Joins to unify meal categories and center metadata without record loss.

### B. Strict Anti-Leakage Principles
In time-series machine learning, standard random `train_test_split` causes **temporal lookahead bias** (training on future information to predict the past). We enforce strict temporal integrity:
1. **Chronological Splitting**:
   - **Training Set:** Weeks 5 to 125 ($376,563$ records)
   - **Validation Set:** Weeks 126 to 135 ($32,799$ records — used for early stopping)
   - **Held-Out Test Set:** Weeks 136 to 145 ($32,812$ records — unseen real-world test)
2. **Lag Features ($t-1, t-2, t-3, t-4$)**: Target values shifted backward so only past data is available at prediction time.
3. **Rolling Window Statistics**: All moving averages (`rolling_mean_3`, `rolling_mean_7`, `rolling_std_3`) operate on shifted targets (`shift(1)`) with `closed='left'`.
4. **Log-Transform Target ($\log(1 + y)$)**: The raw demand exhibits positive skewness ($+6.93$). Applying `np.log1p` stabilizes variance and prevents heavy outliers from destabilizing gradient descent.

---

## 📊 4. Empirical Evaluation & Benchmark Results

As required, an advanced ML model must justify its deployment by beating a classical time-series baseline. We evaluated against a **7-period Moving Average Baseline**:

| Metric | Moving Average Baseline | XGBoost Regressor | Model Improvement | Business Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **MAE** | **84.79** | **63.39** | **+25.24% Reduction** | Predictions are on average 21.4 portions closer to true kitchen demand. |
| **RMSE** | **166.05** | **110.27** | **+33.59% Reduction** | Massive reduction in severe prediction misses that cause total stockouts. |
| **MAPE** | **67.66%** | **43.10%** | **+36.30% Reduction** | Significant relative percentage accuracy boost across all meal categories. |

### Top Predictive Features Identified by XGBoost:
1. `lag_1` & `rolling_mean_3`: Immediate short-term demand momentum.
2. `homepage_featured` & `emailer_for_promotion`: Marketing campaigns generate substantial demand surges.
3. `checkout_price` & `discount_percent`: High price elasticity of demand across fast-food categories.
4. `op_area` & `category_freq`: Center delivery catchment size and cuisine popularity.

---

## 📦 5. Inventory Recommendation & Safety Stock Formulas

### 1. Dynamic Safety Stock Formula
$$\text{Safety Stock} = Z \times \sigma_{\text{demand}} \times \sqrt{L}$$
Where:
- $Z = 1.65$ corresponds to a **95% service level confidence** (only 5% theoretical risk of stockout).
- $\sigma_{\text{demand}}$ is the historical standard deviation of demand for that meal item.
- $L$ is lead time in delivery periods (default = 1 day/week).

### 2. Recommended Procurement Formula
$$\text{Recommended Quantity} = \max\Big(0, \;\; \text{Predicted Demand} + \text{Safety Stock} - \text{Current Stock}\Big)$$
*Guarantee: Recommendations are bounded to non-negative quantities ($\ge 0$).*

---

## 🚀 6. Installation & Execution Guide

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### Step 1: Clone and Setup Environment
```bash
git clone https://github.com/your-username/food-demand-forecasting.git
cd "food demand ML"

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset & Run Pipeline
```bash
# 1. Download official benchmark dataset (auto-verifies files in data/raw/)
python src/data/download_data.py

# 2. Run Exploratory Data Analysis and generate charts
python src/data/eda.py

# 3. Run Feature Engineering (generates features_data.parquet)
python src/features/engineering.py

# 4. Train XGBoost Model, Benchmark Baseline, and Save Artifacts
python src/models/train.py
```

### Step 3: Launch FastAPI Backend Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: **http://127.0.0.1:8000/docs**
- Health Check: **http://127.0.0.1:8000/health**

### Step 4: Launch Interactive Streamlit Dashboard
In a new terminal window:
```bash
streamlit run dashboard/app.py
```
- Open browser at **http://localhost:8501**

### Step 5: Run Automated Test Suite
```bash
python -m pytest tests/
```
*Expected: 9 passed, 100% test coverage across features, inventory logic, and API endpoints.*

---

## 📡 7. API Reference

### 1. `POST /predict`
Predicts demand and recommends inventory replenishment for a specific item.
```json
// Request:
{
  "item": "Chicken Biryani",
  "meal_id": 1885,
  "center_id": 55,
  "date": "2026-10-01",
  "current_stock": 20.0,
  "checkout_price": 280.0,
  "base_price": 310.0,
  "emailer_for_promotion": 0,
  "homepage_featured": 0
}

// Response:
{
  "item": "Thai Beverages #1885",
  "meal_id": 1885,
  "date": "2026-10-01",
  "predicted_demand": 85.0,
  "safety_stock": 17.0,
  "current_stock": 20.0,
  "recommended_quantity": 82.0,
  "stockout_risk": "HIGH",
  "waste_risk": "LOW",
  "potential_shortage": 65.0,
  "potential_surplus": 0.0,
  "estimated_lost_revenue": 18200.0,
  "estimated_waste_cost": 0.0
}
```

### 2. `POST /forecast`
Generates a multi-day (7, 14, or 30 days) projected breakdown with daily distribution.

---

## 🎓 8. Academic Viva & Interview Guide (Q&A)

### Q1: Why did you use chronological splitting instead of random `train_test_split`?
> **Answer:** Random train-test splitting shuffles past and future records uniformly. If the model trains on Week 20 data while evaluating on Week 19, it experiences **temporal lookahead bias / data leakage**. Chronological splitting mimics the true production lifecycle where the model is trained strictly on historical weeks and evaluated only on future weeks.

### Q2: Why did you choose XGBoost over Deep Learning (LSTM / Transformers)?
> **Answer:** Tabular time-series datasets with structured metadata (promotions, price changes, meal category) consistently yield superior performance, lower latency, and higher explainability with gradient-boosted decision trees (XGBoost) than recurrent neural networks. Furthermore, tree-based models handle non-linear promotional spikes and missing values natively with minimal compute overhead.

### Q3: Why is Safety Stock derived using standard deviation rather than a fixed buffer?
> **Answer:** A fixed buffer (e.g. always keeping 20 units) either over-stocks stable items (causing spoilage) or under-stocks volatile items (causing stockouts). Deriving safety stock as $Z \times \sigma_{\text{demand}} \times \sqrt{L}$ dynamically allocates higher safety buffers to volatile dishes and leaner buffers to predictable dishes.

---

## 🔮 9. Limitations & Future Roadmap
- **Weather API Integration:** Adding rainfall and temperature signals to model demand surges for soups/beverages.
- **Deep Reinforcement Learning for Procurement:** Upgrading from static inventory formulas to dynamic continuous-action procurement agents.
- **Automated Supplier Purchase Order Generation:** Direct ERP webhooks for automated vendor ordering.
