"""
Exploratory Data Analysis (EDA) module for Food Demand Forecasting.
Performs comprehensive data profiling, statistical analysis, and generates
publication-quality visualizations saved to reports/figures/.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data.loader import load_raw_datasets, merge_datasets
from src.utils.logger import get_logger

logger = get_logger("eda")


def run_statistical_profiling(df: pd.DataFrame) -> dict:
    """
    Computes rigorous statistical profiling metrics on the merged dataset.
    """
    logger.info("Computing statistical profiling...")
    
    # Target distribution statistics
    orders = df["num_orders"]
    q1 = float(orders.quantile(0.25))
    q3 = float(orders.quantile(0.75))
    iqr = q3 - q1
    upper_whisker = q3 + 1.5 * iqr
    outlier_count = int((orders > upper_whisker).sum())
    
    # Price and discount statistics
    discount_amount = df["base_price"] - df["checkout_price"]
    discount_pct = (discount_amount / df["base_price"]) * 100
    
    stats = {
        "record_count": len(df),
        "feature_count": len(df.columns),
        "features": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        "duplicates": int(df.duplicated().sum()),
        "time_span_weeks": {
            "min_week": int(df["week"].min()),
            "max_week": int(df["week"].max()),
            "total_weeks": int(df["week"].nunique()),
        },
        "entities": {
            "unique_meals": int(df["meal_id"].nunique()),
            "unique_categories": int(df["category"].nunique()),
            "unique_cuisines": int(df["cuisine"].nunique()),
            "unique_centers": int(df["center_id"].nunique()),
            "unique_cities": int(df["city_code"].nunique()),
            "unique_regions": int(df["region_code"].nunique()),
        },
        "target_distribution (num_orders)": {
            "mean": round(float(orders.mean()), 2),
            "std": round(float(orders.std()), 2),
            "median": round(float(orders.median()), 2),
            "min": int(orders.min()),
            "max": int(orders.max()),
            "skewness": round(float(orders.skew()), 2),
            "kurtosis": round(float(orders.kurtosis()), 2),
            "q25": round(q1, 2),
            "q75": round(q3, 2),
            "iqr_outliers_count": outlier_count,
            "iqr_outliers_pct": round((outlier_count / len(df)) * 100, 2),
        },
        "pricing_and_promotions": {
            "avg_base_price": round(float(df["base_price"].mean()), 2),
            "avg_checkout_price": round(float(df["checkout_price"].mean()), 2),
            "avg_discount_pct": round(float(discount_pct.mean()), 2),
            "emailer_promo_share_pct": round(float(df["emailer_for_promotion"].mean() * 100), 2),
            "homepage_featured_share_pct": round(float(df["homepage_featured"].mean() * 100), 2),
        },
        "top_categories_by_orders": df.groupby("category")["num_orders"].sum().sort_values(ascending=False).to_dict(),
        "top_cuisines_by_orders": df.groupby("cuisine")["num_orders"].sum().sort_values(ascending=False).to_dict(),
    }
    return stats


def generate_visualizations(df: pd.DataFrame):
    """
    Generates and exports high-resolution visual charts for reporting.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/script execution
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "figure.autolayout": True})
    
    # 1. Overall Demand Over Time (Weekly Aggregated Trend)
    logger.info("Generating figure: 1_overall_demand_over_time.png")
    weekly_trend = df.groupby("week")["num_orders"].sum().reset_index()
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(weekly_trend["week"], weekly_trend["num_orders"] / 1000, color="#1f77b4", lw=2)
    plt.title("Overall Food Demand Trend Over Time (145 Weeks)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Week Number", fontsize=11)
    plt.ylabel("Total Orders (in Thousands)", fontsize=11)
    # Add rolling moving average line
    weekly_trend["ma_4"] = (weekly_trend["num_orders"] / 1000).rolling(window=4).mean()
    plt.plot(weekly_trend["week"], weekly_trend["ma_4"], color="#ff7f0e", linestyle="--", lw=2, label="4-Week Moving Average")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "1_overall_demand_over_time.png")
    plt.close()

    # 2. Demand by Category
    logger.info("Generating figure: 2_demand_by_category.png")
    cat_demand = df.groupby("category")["num_orders"].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 6), dpi=150)
    bars = sns.barplot(data=cat_demand, x="num_orders", y="category", palette="Blues_r")
    plt.title("Total Demand by Food Category", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Total Orders Placed", fontsize=11)
    plt.ylabel("Category", fontsize=11)
    for bar in bars.patches:
        width = bar.get_width()
        plt.text(width + (0.01 * cat_demand["num_orders"].max()), bar.get_y() + bar.get_height() / 2,
                 f"{int(width):,}", va="center", fontsize=9, color="#333333")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_demand_by_category.png")
    plt.close()

    # 3. Top 10 Selling Food Items (Meals)
    logger.info("Generating figure: 3_top_selling_meals.png")
    top_meals = df.groupby(["meal_id", "category", "cuisine"])["num_orders"].sum().reset_index()
    top_meals["label"] = "Meal #" + top_meals["meal_id"].astype(str) + " (" + top_meals["category"] + ")"
    top_meals = top_meals.sort_values(by="num_orders", ascending=False).head(10)
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(data=top_meals, x="num_orders", y="label", palette="viridis")
    plt.title("Top 10 High-Demand Meal Items", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Total Orders", fontsize=11)
    plt.ylabel("Food Item / Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_top_selling_meals.png")
    plt.close()

    # 4. Distribution of Target Demand (Raw vs Log-Transformed)
    logger.info("Generating figure: 4_demand_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    sns.histplot(df["num_orders"], bins=50, kde=True, ax=axes[0], color="#2ca02c")
    axes[0].set_title("Raw Demand Distribution (Heavy Right Skew)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Orders")
    axes[0].set_ylabel("Frequency")
    
    # Log-transformed target
    log_orders = np.log1p(df["num_orders"])
    sns.histplot(log_orders, bins=50, kde=True, ax=axes[1], color="#9467bd")
    axes[1].set_title("Log1p Transformed Demand (Near Normal)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(num_orders + 1)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "4_demand_distribution.png")
    plt.close()

    # 5. Impact of Promotions on Demand
    logger.info("Generating figure: 5_promotion_impact.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    sns.boxplot(data=df, x="emailer_for_promotion", y="num_orders", ax=axes[0], showfliers=False, palette="Set2")
    axes[0].set_title("Demand vs. Email Promotion", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(["No Email Promo", "Email Promo Sent"])
    axes[0].set_ylabel("Orders per Week")
    
    sns.boxplot(data=df, x="homepage_featured", y="num_orders", ax=axes[1], showfliers=False, palette="Set2")
    axes[1].set_title("Demand vs. Homepage Feature", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(["Not Featured", "Featured on Homepage"])
    axes[1].set_ylabel("Orders per Week")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "5_promotion_impact.png")
    plt.close()

    # 6. Correlation Heatmap
    logger.info("Generating figure: 6_correlation_heatmap.png")
    df_corr = df[["checkout_price", "base_price", "emailer_for_promotion", "homepage_featured", "op_area", "num_orders"]].copy()
    df_corr["discount_amount"] = df_corr["base_price"] - df_corr["checkout_price"]
    corr_matrix = df_corr.corr()
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix: Pricing, Promotions & Demand", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "6_correlation_heatmap.png")
    plt.close()

    logger.info(f"All visualizations successfully saved to {FIGURES_DIR}")


def run_eda_pipeline() -> dict:
    """Runs complete end-to-end EDA pipeline and exports summary JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, meal_df, center_df = load_raw_datasets()
    merged_df = merge_datasets(train_df, meal_df, center_df)
    
    stats = run_statistical_profiling(merged_df)
    
    # Save statistics report JSON
    report_file = REPORTS_DIR / "eda_summary.json"
    with open(report_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved EDA statistical summary to {report_file}")
    
    # Generate charts
    generate_visualizations(merged_df)
    return stats


if __name__ == "__main__":
    run_eda_pipeline()
