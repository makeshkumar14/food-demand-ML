"""
Streamlit Web Dashboard for Food Demand Forecasting & Inventory Optimization.
Day-by-day forecast breakdown from weekly LightGBM predictions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR, MODEL_FILE, MODEL_METRICS_FILE, FIGURES_DIR
from src.forecasting.service import DemandForecaster
from src.inventory.recommendation import InventoryOptimizer

st.set_page_config(
    page_title="Food Demand Forecasting",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal black theme with subtle glass surfaces
st.markdown("""
<style>
    :root {
        --page: #000000;
        --surface: rgba(255, 255, 255, 0.035);
        --surface-hover: rgba(255, 255, 255, 0.075);
        --line: rgba(255, 255, 255, 0.11);
        --text: #ececec;
        --muted: #9b9b9b;
        --accent: #10a37f;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--page); color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(0, 0, 0, 0.82); }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.9);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] > div { background: transparent; }
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] { color: var(--text); }
    [data-testid="stCaptionContainer"], small { color: var(--muted) !important; }
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; }
    [data-testid="stRadio"] > div { gap: 0.35rem; }
    [data-testid="stRadio"] label {
        border: 1px solid var(--line); border-radius: 999px;
        background: var(--surface); padding: 0.45rem 0.85rem;
    }
    [data-testid="stRadio"] label:hover { background: var(--surface-hover); }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {
        background: rgba(255, 255, 255, 0.045); color: var(--text);
        border-color: var(--line); border-radius: 9px;
    }
    [data-testid="stExpander"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    hr { border-color: var(--line); }
    div.stButton > button:first-child {
        background: var(--accent); color: white; border: 1px solid var(--accent);
        border-radius: 9px; padding: 0.55rem 1.2rem;
        font-weight: 600; font-size: 0.95rem; transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: #0b8c6d; border-color: #0b8c6d;
    }
    .metric-box {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: 18px; margin-top: 10px;
    }
    .metric-value { font-size: 26px; font-weight: 700; color: #f5f5f5; }
    .metric-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px; }
    .day-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
        padding: 14px 10px; text-align: center; margin: 4px;
    }
    .day-name { font-size: 12px; color: #d4d4d4; font-weight: 600; letter-spacing: 0.3px; }
    .day-date { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
    .day-demand { font-size: 22px; font-weight: 700; color: #f5f5f5; }
    .day-unit { font-size: 11px; color: var(--muted); }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
    }
</style>
""", unsafe_allow_html=True)


# Intra-week day consumption weights (Mon–Sun)
DAY_WEIGHTS = [0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.16]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_EMOJIS = ["💼", "💼", "💼", "💼", "🎉", "🍽️", "🍽️"]


@st.cache_resource
def load_ml_services():
    try:
        return DemandForecaster(MODEL_FILE), InventoryOptimizer(service_level_z=1.65)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None, InventoryOptimizer()

forecaster, optimizer = load_ml_services()


@st.cache_data
def load_data():
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv") if (RAW_DATA_DIR / "train.csv").exists() else pd.DataFrame()
    meal_df = pd.read_csv(RAW_DATA_DIR / "meal_info.csv") if (RAW_DATA_DIR / "meal_info.csv").exists() else pd.DataFrame()
    center_df = pd.read_csv(RAW_DATA_DIR / "fulfilment_center_info.csv") if (RAW_DATA_DIR / "fulfilment_center_info.csv").exists() else pd.DataFrame()
    return train_df, meal_df, center_df

train_df, meal_df, center_df = load_data()

if "history" not in st.session_state:
    st.session_state.history = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "daily_forecast" not in st.session_state:
    st.session_state.daily_forecast = None


def generate_daily_forecast(weekly_pred: float, start_date: datetime, horizon_weeks: int = 2) -> pd.DataFrame:
    """
    Converts a weekly LightGBM forecast into a day-by-day demand breakdown
    across the given number of weeks using realistic intra-week day weights.
    """
    rows = []
    current_date = start_date

    # Decay factor to simulate natural weekly variation (±5-10% drift)
    rng = np.random.default_rng(seed=int(weekly_pred) % 999)
    for week_num in range(horizon_weeks):
        week_drift = 1.0 + rng.uniform(-0.08, 0.12)
        week_demand = weekly_pred * week_drift

        for day_idx in range(7):
            daily_demand = round(week_demand * DAY_WEIGHTS[day_idx], 1)
            rows.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_of_week": DAY_NAMES[day_idx],
                "day_short": DAY_SHORT[day_idx],
                "emoji": DAY_EMOJIS[day_idx],
                "week_num": week_num + 1,
                "predicted_demand": daily_demand,
            })
            current_date += timedelta(days=1)

    return pd.DataFrame(rows)


# ─── NAVIGATION ────────────────────────────────────────────────────────────────
nav_tab = st.radio(
    "",
    ["📈 Demand Forecasting", "📅 Day-by-Day Forecast", "📋 Inventory", "🎯 Viva Metrics"],
    horizontal=True,
    label_visibility="collapsed"
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: WEEKLY HISTORICAL + POINT FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
if nav_tab == "📈 Demand Forecasting":
    # ── SIDEBAR ──
    st.sidebar.markdown("### Forecast inputs")
    center_ids = sorted(train_df["center_id"].unique().tolist()) if not train_df.empty else [55]
    selected_center_id = st.sidebar.selectbox("Center ID", center_ids,
                                              index=center_ids.index(55) if 55 in center_ids else 0)
    available_meals = sorted(train_df[train_df["center_id"] == selected_center_id]["meal_id"].unique().tolist()) if not train_df.empty else [1062]
    selected_meal_id = st.sidebar.selectbox("Meal ID", available_meals,
                                            index=available_meals.index(1062) if 1062 in available_meals else 0)
    selected_week = st.sidebar.number_input("Week number", min_value=1, max_value=160, value=145)
    predict_btn = st.sidebar.button("Predict")

    # Meta
    center_meta = center_df[center_df["center_id"] == selected_center_id]
    meal_meta = meal_df[meal_df["meal_id"] == selected_meal_id]
    center_type = center_meta["center_type"].values[0] if not center_meta.empty else "TYPE_B"
    meal_category = meal_meta["category"].values[0] if not meal_meta.empty else "Beverages"
    meal_cuisine = meal_meta["cuisine"].values[0] if not meal_meta.empty else "Italian"

    # Historical series
    if not train_df.empty:
        series = train_df[(train_df["center_id"] == selected_center_id) &
                          (train_df["meal_id"] == selected_meal_id)]
        series = series[series["week"] <= selected_week].sort_values("week").reset_index(drop=True)
    else:
        series = pd.DataFrame()

    # ── MAIN ──
    st.markdown("## **Food Demand Forecasting**")
    st.caption("Predict weekly meal orders for a fulfilment center. Pick a center, a meal, and a week number, then click Predict.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_meta, col_chart = st.columns([1, 2.5])

    with col_meta:
        st.markdown("### **Selection**")
        st.markdown(f"**Center type:** {center_type}")
        st.markdown(f"**Meal category:** {meal_category}")
        st.markdown(f"**Cuisine:** {meal_cuisine}")
        st.markdown(f"**Weeks of history:** {len(series)}")

        if predict_btn and forecaster is not None:
            recent_orders = [float(v) for v in series["num_orders"].tail(4).tolist()] if len(series) >= 4 else [150.0] * 4
            pred_val = forecaster.predict_single(
                meal_id=int(selected_meal_id),
                center_id=int(selected_center_id),
                checkout_price=float(series["checkout_price"].iloc[-1]) if "checkout_price" in series.columns and len(series) > 0 else 300.0,
                base_price=float(series["base_price"].iloc[-1]) if "base_price" in series.columns and len(series) > 0 else 320.0,
                recent_lags=recent_orders,
                week=int(selected_week) + 1,
            )
            st.session_state.last_pred = {"meal_id": selected_meal_id, "center_id": selected_center_id, "pred": pred_val, "week": int(selected_week) + 1}

            # Generate daily forecast and store in session
            start = datetime.today() + timedelta(days=(7 - datetime.today().weekday()) % 7)
            st.session_state.daily_forecast = generate_daily_forecast(pred_val, start_date=start, horizon_weeks=2)

            st.session_state.history.append({
                "Center": selected_center_id, "Meal": selected_meal_id,
                "Category": meal_category, "Forecast Week": int(selected_week) + 1,
                "Predicted Orders": round(pred_val, 1)
            })

            opt_res = optimizer.generate_recommendation(
                predicted_demand=pred_val, current_stock=200.0,
                demand_std=float(np.std(recent_orders)) if len(recent_orders) > 1 else 30.0)

            st.markdown("---")
            st.markdown("### **Weekly Prediction**")
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Predicted Demand — Week {int(selected_week)+1}</div>
                <div class="metric-value">{round(pred_val):,} <span style="font-size:15px;color:#94a3b8;">orders / week</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Recommended Procurement</div>
                <div class="metric-value" style="color:#10b981;">{round(opt_res['recommended_quantity']):,} <span style="font-size:15px;color:#94a3b8;">portions</span></div>
                <div style="font-size:12px;color:#94a3b8;margin-top:4px;">Safety stock buffer: +{round(opt_res['safety_stock'])} portions</div>
            </div>""", unsafe_allow_html=True)
            st.info("👉 Switch to **Day-by-Day Forecast** tab to see the daily breakdown!")

    with col_chart:
        st.markdown("### **Historical vs Predicted Demand**")
        fig = go.Figure()
        if not series.empty:
            fig.add_trace(go.Scatter(x=series["week"], y=series["num_orders"],
                mode="lines", name="Historical orders",
                line=dict(color="#38bdf8", width=2)))
        if st.session_state.last_pred:
            p = st.session_state.last_pred
            if p["meal_id"] == selected_meal_id and p["center_id"] == selected_center_id:
                fig.add_trace(go.Scatter(x=[p["week"]], y=[p["pred"]],
                    mode="markers+text", name="Forecast",
                    marker=dict(color="#ef4444", size=12, symbol="circle"),
                    text=[f"  {int(p['pred'])} orders"], textposition="top right",
                    textfont=dict(color="#f87171", size=12)))
                if not series.empty:
                    fig.add_trace(go.Scatter(x=[series["week"].iloc[-1], p["week"]], y=[series["num_orders"].iloc[-1], p["pred"]],
                        mode="lines", line=dict(color="#ef4444", width=2, dash="dot"), showlegend=False))
        fig.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(title="week", tickfont=dict(color="#94a3b8"), showgrid=True, gridcolor="#1e293b", dtick=10),
            yaxis=dict(title="Historical orders", tickfont=dict(color="#94a3b8"), showgrid=True, gridcolor="#1e293b"),
            legend=dict(font=dict(color="#cbd5e1"), orientation="h", y=1.02, x=1, xanchor="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        if not predict_btn and not st.session_state.last_pred:
            st.caption("Click Predict to mark a forecast on this chart.")

    with st.expander("Prediction History", expanded=False):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.write("No predictions yet. Pick a meal and click **Predict**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DAY-BY-DAY FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_tab == "📅 Day-by-Day Forecast":
    st.markdown("## **Day-by-Day Demand Forecast**")
    st.caption("Daily breakdown of food order predictions for the coming days, derived from the LightGBM weekly model.")

    # ── SIDEBAR ──
    st.sidebar.markdown("### Forecast inputs")
    center_ids_d = sorted(train_df["center_id"].unique().tolist()) if not train_df.empty else [55]
    sel_center = st.sidebar.selectbox("Center ID", center_ids_d,
                                      index=center_ids_d.index(55) if 55 in center_ids_d else 0, key="d_center")
    meals_d = sorted(train_df[train_df["center_id"] == sel_center]["meal_id"].unique().tolist()) if not train_df.empty else [1062]
    sel_meal = st.sidebar.selectbox("Meal ID", meals_d,
                                    index=meals_d.index(1062) if 1062 in meals_d else 0, key="d_meal")
    horizon_weeks = st.sidebar.selectbox("Forecast Horizon", [1, 2, 4], index=1,
                                         format_func=lambda x: f"{x} week{'s' if x > 1 else ''} ({x*7} days)")
    start_date = st.sidebar.date_input("Starting From", value=datetime.today())
    forecast_btn = st.sidebar.button("Generate Day-by-Day Forecast", key="d_btn")

    if forecast_btn and forecaster is not None:
        # Get historical series for lags
        if not train_df.empty:
            s = train_df[(train_df["center_id"] == sel_center) & (train_df["meal_id"] == sel_meal)].sort_values("week")
            recent = [float(v) for v in s["num_orders"].tail(4).tolist()] if len(s) >= 4 else [150.0] * 4
            co_price = float(s["checkout_price"].iloc[-1]) if "checkout_price" in s.columns and len(s) > 0 else 300.0
            ba_price = float(s["base_price"].iloc[-1]) if "base_price" in s.columns and len(s) > 0 else 320.0
        else:
            recent = [150.0] * 4
            co_price, ba_price = 300.0, 320.0

        weekly_pred = forecaster.predict_single(
            meal_id=int(sel_meal), center_id=int(sel_center),
            checkout_price=co_price, base_price=ba_price,
            recent_lags=recent, week=146
        )

        daily_df = generate_daily_forecast(weekly_pred, start_date=datetime.combine(start_date, datetime.min.time()), horizon_weeks=horizon_weeks)
        st.session_state.daily_forecast = daily_df

    if st.session_state.daily_forecast is not None:
        daily_df = st.session_state.daily_forecast.head(horizon_weeks * 7 if forecast_btn else len(st.session_state.daily_forecast))

        # ── KPI Summary ──
        total_demand = daily_df["predicted_demand"].sum()
        peak_day = daily_df.loc[daily_df["predicted_demand"].idxmax()]
        low_day = daily_df.loc[daily_df["predicted_demand"].idxmin()]
        avg_daily = daily_df["predicted_demand"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📦 Total Projected Demand", f"{int(total_demand):,} portions", f"Over {len(daily_df)} days")
        k2.metric("📈 Peak Day", f"{peak_day['day_of_week'][:3]} {peak_day['date'][5:]}", f"{int(peak_day['predicted_demand'])} portions")
        k3.metric("📉 Lowest Day", f"{low_day['day_of_week'][:3]} {low_day['date'][5:]}", f"{int(low_day['predicted_demand'])} portions")
        k4.metric("📊 Daily Average", f"{avg_daily:.1f} portions", "per day")

        st.markdown("---")

        # ── Day Cards ──
        st.markdown("### Daily Demand Cards")
        days_to_show = min(14, len(daily_df))
        week1 = daily_df.head(7)
        week2 = daily_df.iloc[7:14] if len(daily_df) >= 14 else pd.DataFrame()

        st.markdown(f"**Week 1** — {week1['date'].iloc[0]} to {week1['date'].iloc[-1]}")
        cols = st.columns(7)
        for idx, row in week1.reset_index(drop=True).iterrows():
            with cols[idx]:
                weekend_color = "#818cf8" if row["day_of_week"] in ["Saturday", "Sunday"] else "#38bdf8"
                st.markdown(f"""<div class="day-card">
                    <div class="day-name">{row['emoji']} {row['day_short']}</div>
                    <div class="day-date">{row['date'][5:]}</div>
                    <div class="day-demand" style="color:{weekend_color};">{int(row['predicted_demand'])}</div>
                    <div class="day-unit">portions</div>
                </div>""", unsafe_allow_html=True)

        if not week2.empty:
            st.markdown(f"<br>**Week 2** — {week2['date'].iloc[0]} to {week2['date'].iloc[-1]}", unsafe_allow_html=True)
            cols2 = st.columns(7)
            for idx, row in week2.reset_index(drop=True).iterrows():
                with cols2[idx]:
                    weekend_color = "#818cf8" if row["day_of_week"] in ["Saturday", "Sunday"] else "#38bdf8"
                    st.markdown(f"""<div class="day-card">
                        <div class="day-name">{row['emoji']} {row['day_short']}</div>
                        <div class="day-date">{row['date'][5:]}</div>
                        <div class="day-demand" style="color:{weekend_color};">{int(row['predicted_demand'])}</div>
                        <div class="day-unit">portions</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar Chart ──
        st.markdown("### Day-by-Day Demand Trend")
        colors = ["#818cf8" if d in ["Saturday", "Sunday"] else "#38bdf8" for d in daily_df["day_of_week"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[f"{r['day_short']}<br>{r['date'][5:]}" for _, r in daily_df.iterrows()],
            y=daily_df["predicted_demand"],
            marker_color=colors,
            text=[f"{int(v)}" for v in daily_df["predicted_demand"]],
            textposition="outside",
            textfont=dict(color="#cbd5e1", size=11),
        ))
        # Rolling 3-day average overlay
        rolling_avg = daily_df["predicted_demand"].rolling(3, min_periods=1).mean()
        fig2.add_trace(go.Scatter(
            x=[f"{r['day_short']}<br>{r['date'][5:]}" for _, r in daily_df.iterrows()],
            y=rolling_avg,
            mode="lines+markers",
            name="3-Day Rolling Avg",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=5)
        ))
        fig2.update_layout(
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            margin=dict(l=30, r=20, t=20, b=60),
            xaxis=dict(tickfont=dict(color="#94a3b8", size=10), showgrid=False),
            yaxis=dict(title="Portions Required", tickfont=dict(color="#94a3b8"), showgrid=True, gridcolor="#1e293b"),
            bargap=0.25,
            legend=dict(font=dict(color="#cbd5e1")),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Data Table ──
        st.markdown("### Full Day-by-Day Forecast Table")
        display_df = daily_df[["date", "day_of_week", "week_num", "predicted_demand"]].copy()
        display_df.columns = ["Date", "Day of Week", "Week #", "Predicted Demand (Portions)"]
        display_df["Predicted Demand (Portions)"] = display_df["Predicted Demand (Portions)"].apply(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Inventory Recommendation per Week ──
        st.markdown("### Inventory Procurement Recommendation by Week")
        for wk in sorted(daily_df["week_num"].unique()):
            wk_data = daily_df[daily_df["week_num"] == wk]
            wk_total = wk_data["predicted_demand"].sum()
            wk_std = wk_data["predicted_demand"].std()
            inv = optimizer.generate_recommendation(predicted_demand=wk_total, current_stock=0.0, demand_std=float(wk_std))
            with st.expander(f"Week {wk} — Total {int(wk_total)} portions | {wk_data['date'].iloc[0]} to {wk_data['date'].iloc[-1]}"):
                w1, w2, w3 = st.columns(3)
                w1.metric("Weekly Demand", f"{int(wk_total)} portions")
                w2.metric("Safety Stock Buffer", f"{round(inv['safety_stock'])} portions")
                w3.metric("Recommended Order", f"{round(inv['recommended_quantity'])} portions")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 Select a Center ID, Meal ID, and horizon on the left sidebar, then click **Generate Day-by-Day Forecast**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_tab == "📋 Inventory":
    st.markdown("## **Smart Inventory & Procurement Engine**")
    c1, c2, c3 = st.columns(3)
    with c1:
        all_meals = sorted(train_df["meal_id"].unique().tolist()) if not train_df.empty else [1885]
        inp_meal = st.selectbox("Meal ID:", all_meals, key="inv_meal")
    with c2:
        inp_stock = st.number_input("Current Stock (Portions):", min_value=0.0, value=150.0, step=10.0)
    with c3:
        inp_z = st.selectbox("Service Level:", ["95% (Z=1.65)", "99% (Z=2.33)", "90% (Z=1.28)"])
        z = 1.65 if "95%" in inp_z else (2.33 if "99%" in inp_z else 1.28)

    if st.button("Calculate Procurement Order", type="primary"):
        pred = forecaster.predict_single(meal_id=int(inp_meal), center_id=55) if forecaster else 200.0
        opt = InventoryOptimizer(service_level_z=z)
        inv_res = opt.generate_recommendation(predicted_demand=pred, current_stock=inp_stock, demand_std=max(15.0, pred * 0.15))
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Predicted Weekly Demand", f"{round(pred)} portions")
        q2.metric("Safety Stock Buffer", f"{round(inv_res['safety_stock'])} portions")
        q3.metric("Current Stock", f"{round(inp_stock)} portions")
        q4.metric("Order This Week", f"{round(inv_res['recommended_quantity'])} portions",
                  delta=f"Stockout Risk: {inv_res['stockout_risk']}")
        if inv_res["stockout_risk"] in ["HIGH", "MODERATE"]:
            st.error(f"⚠️ Shortage of {round(inv_res['potential_shortage'])} portions if not replenished!")
        else:
            st.success("✅ Stock level is sufficient to cover expected demand.")
        with st.expander("Safety Stock Formula (Viva Reference)"):
            st.latex(r"\text{Safety Stock} = Z \times \sigma_{\text{demand}} \times \sqrt{L}")
            st.latex(r"\text{Recommended Qty} = \max(0, \text{ Predicted} + \text{Safety Stock} - \text{Current Stock})")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: VIVA METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav_tab == "🎯 Viva Metrics":
    st.markdown("## **Model Evaluation & Viva Benchmark Comparison**")
    if MODEL_METRICS_FILE.exists():
        with open(MODEL_METRICS_FILE, "r") as f:
            m = json.load(f)
        b, x, imp = m["baseline_test"], m["lightgbm_test"], m["improvement_pct"]
        k1, k2, k3 = st.columns(3)
        k1.metric("MAE", f"{x['MAE']}", f"-{imp['MAE_reduction']}% vs Baseline")
        k2.metric("RMSE", f"{x['RMSE']}", f"-{imp['RMSE_reduction']}% vs Baseline")
        k3.metric("MAPE", f"{x['MAPE']}%", f"-{imp['MAPE_reduction']}% vs Baseline")
        st.subheader("Model Benchmark Comparison")
        st.table(pd.DataFrame([
            {"Model": "7-Week Moving Average Baseline", "MAE": b["MAE"], "RMSE": b["RMSE"], "MAPE": f"{b['MAPE']}%"},
            {"Model": "LightGBM Regressor (Our Model)", "MAE": x["MAE"], "RMSE": x["RMSE"], "MAPE": f"{x['MAPE']}%"}
        ]))
        imp_fig = FIGURES_DIR / "7_feature_importance.png"
        pred_fig = FIGURES_DIR / "8_actual_vs_predicted.png"
        g1, g2 = st.columns(2)
        with g1:
            if imp_fig.exists(): st.image(str(imp_fig), caption="Top 15 Feature Importances")
        with g2:
            if pred_fig.exists(): st.image(str(pred_fig), caption="Actual vs. Predicted on Test Set")
