"""
Helper script to generate the educational Jupyter Notebook: 01_data_exploration.ipynb.
"""

import json
from pathlib import Path

notebook_path = Path("notebooks/01_data_exploration.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 1: Exploratory Data Analysis (EDA)\n",
            "## Food Demand Forecasting & Inventory Optimization System\n",
            "\n",
            "**Academic Context:** B.Tech CSE Major / Portfolio Project  \n",
            "**Goal:** Inspect data shapes, missing values, demand distributions, promotion impacts, and time-series patterns across meal categories."
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
            "# Add parent directory to sys.path\n",
            "sys.path.append('..')\n",
            "from src.data.loader import load_raw_datasets, merge_datasets, get_dataset_summary\n",
            "\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "print('Libraries imported successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Ingestion & Merging of Metadata\n",
            "We load `train.csv`, `meal_info.csv`, and `fulfilment_center_info.csv` using our modular `loader`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df, meal_df, center_df = load_raw_datasets()\n",
            "df = merge_datasets(train_df, meal_df, center_df)\n",
            "print(f'Merged DataFrame dimensions: {df.shape}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Dataset Health Check & Data Types\n",
            "- Check missing values across all columns.\n",
            "- Check for duplicate records."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary = get_dataset_summary(df)\n",
            "print(f'Total Records: {summary[\"num_rows\"]:,}')\n",
            "print(f'Total Missing Values: {sum(summary[\"missing_values\"].values())}')\n",
            "print(f'Duplicate Rows: {summary[\"num_duplicates\"]}')\n",
            "print(f'Unique Meals: {summary[\"num_unique_meals\"]}, Unique Centers: {summary[\"num_unique_centers\"]}')\n",
            "print(f'Week Range: Week {summary[\"min_week\"]} to Week {summary[\"max_week\"]}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Demand Distribution & Skewness Analysis\n",
            "Food order quantities exhibit high variance and positive skewness. In retail ML, log transformation $\\log(1 + y)$ is commonly applied to stabilize variance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Summary Statistics of Order Demand (num_orders):')\n",
            "print(df['num_orders'].describe())\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "sns.histplot(df['num_orders'], bins=50, kde=True, ax=axes[0], color='forestgreen')\n",
            "axes[0].set_title('Raw Demand (Heavy Right Skew)')\n",
            "\n",
            "sns.histplot(np.log1p(df['num_orders']), bins=50, kde=True, ax=axes[1], color='purple')\n",
            "axes[1].set_title('Log1p Transformed Demand')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Demand Across Categories and Cuisines"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cat_order = df.groupby('category')['num_orders'].sum().sort_values(ascending=False).reset_index()\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(data=cat_order, x='num_orders', y='category', hue='category', palette='Blues_r', legend=False)\n",
            "plt.title('Total Historical Orders by Meal Category')\n",
            "plt.xlabel('Total Orders Placed')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Promotional Impact (Emailers & Homepage Features)\n",
            "Examining whether marketing triggers statistically significant surges in food demand."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "sns.boxplot(data=df, x='emailer_for_promotion', y='num_orders', ax=axes[0], showfliers=False, hue='emailer_for_promotion', palette='Set2', legend=False)\n",
            "axes[0].set_title('Demand Uplift: Email Promotions')\n",
            "axes[0].set_xticks([0, 1], ['No Email', 'Email Sent'])\n",
            "\n",
            "sns.boxplot(data=df, x='homepage_featured', y='num_orders', ax=axes[1], showfliers=False, hue='homepage_featured', palette='Set2', legend=False)\n",
            "axes[1].set_title('Demand Uplift: Homepage Featured')\n",
            "axes[1].set_xticks([0, 1], ['Not Featured', 'Featured'])\n",
            "plt.tight_layout()\n",
            "plt.show()"
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
