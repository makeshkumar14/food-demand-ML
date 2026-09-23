"""
SQLAlchemy ORM models for normalized database tables:
- FoodItem (food_items)
- SaleRecord (sales)
- InventoryItem (inventory)
- ForecastRecord (forecasts)
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.database import Base, engine, SessionLocal
from src.config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("db_models")


class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, index=True)
    meal_id = Column(Integer, unique=True, index=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    cuisine = Column(String(50), nullable=False)
    base_price = Column(Float, default=300.0)

    # Relationships
    sales = relationship("SaleRecord", back_populates="food_item")
    inventory = relationship("InventoryItem", back_populates="food_item")
    forecasts = relationship("ForecastRecord", back_populates="food_item")


class SaleRecord(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    food_item_id = Column(Integer, ForeignKey("food_items.id"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    week = Column(Integer, nullable=False)
    quantity_sold = Column(Float, nullable=False)
    checkout_price = Column(Float, nullable=False)

    food_item = relationship("FoodItem", back_populates="sales")


class InventoryItem(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    food_item_id = Column(Integer, ForeignKey("food_items.id"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    current_stock = Column(Float, default=0.0)
    safety_stock = Column(Float, default=15.0)

    food_item = relationship("FoodItem", back_populates="inventory")


class ForecastRecord(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    food_item_id = Column(Integer, ForeignKey("food_items.id"), nullable=False)
    forecast_date = Column(String(20), nullable=False)
    predicted_demand = Column(Float, nullable=False)
    model_version = Column(String(50), default="XGBoost_v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    food_item = relationship("FoodItem", back_populates="forecasts")


def init_db():
    """Initializes tables and seeds initial items if database is empty."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        count = db.query(FoodItem).count()
        if count == 0:
            logger.info("Seeding food_items table from meal_info.csv...")
            meal_file = RAW_DATA_DIR / "meal_info.csv"
            if meal_file.exists():
                df = pd.read_csv(meal_file)
                for _, row in df.iterrows():
                    item = FoodItem(
                        meal_id=int(row["meal_id"]),
                        name=f"{row['cuisine']} {row['category']} #{row['meal_id']}",
                        category=str(row["category"]),
                        cuisine=str(row["cuisine"]),
                        base_price=320.0,
                    )
                    db.add(item)
                db.commit()
                logger.info(f"Seeded {len(df)} food items into database.")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Database tables initialized and seeded successfully.")
