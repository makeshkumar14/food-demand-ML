"""
Database connection and session management.
Supports PostgreSQL for production with seamless SQLite local dev fallback.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "food_demand.db"

# Default SQLite URL handles Windows backslashes cleanly
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"

# Load from environment or fallback
import os
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
if DATABASE_URL.startswith("sqlite:///./"):
    DATABASE_URL = DEFAULT_DB_URL

# For SQLite, enable check_same_thread=False for multithreaded FastAPI requests
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
