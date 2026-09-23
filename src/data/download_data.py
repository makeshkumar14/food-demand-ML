"""
Automated dataset acquisition and validation script.
Fetches the benchmark Food Demand Forecasting dataset into data/raw/
and verifies data integrity.
"""

import sys
import urllib.request
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR, TRAIN_CSV, MEAL_INFO_CSV, FULFILMENT_CENTER_CSV
from src.utils.logger import get_logger

logger = get_logger("download_data")

DATASET_URLS = {
    "train.csv": "https://raw.githubusercontent.com/CallMeAmartya/Food-Demand-Forecasting/master/train.csv",
    "meal_info.csv": "https://raw.githubusercontent.com/CallMeAmartya/Food-Demand-Forecasting/master/meal_info.csv",
    "fulfilment_center_info.csv": "https://raw.githubusercontent.com/CallMeAmartya/Food-Demand-Forecasting/master/fulfilment_center_info.csv",
}

EXPECTED_COLUMNS = {
    "train.csv": ["id", "week", "center_id", "meal_id", "checkout_price", "base_price", "emailer_for_promotion", "homepage_featured", "num_orders"],
    "meal_info.csv": ["meal_id", "category", "cuisine"],
    "fulfilment_center_info.csv": ["center_id", "city_code", "region_code", "center_type", "op_area"],
}


def download_file(filename: str, url: str, target_path: Path) -> bool:
    """Downloads a single file from URL with progress reporting."""
    logger.info(f"Downloading {filename} from {url}...")
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                if block_num % 500 == 0 or percent == 100:
                    sys.stdout.write(f"\rDownloading {filename}: {percent}%")
                    sys.stdout.flush()

        urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
        print()  # newline after progress
        logger.info(f"Successfully downloaded {filename} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False


def verify_file(filename: str, file_path: Path) -> bool:
    """Verifies that the CSV file exists, is non-empty, and contains expected columns."""
    if not file_path.exists():
        logger.error(f"Verification failed: {file_path} does not exist.")
        return False

    try:
        # Read header only to check columns
        df_head = pd.read_csv(file_path, nrows=5)
        expected = EXPECTED_COLUMNS.get(filename, [])
        missing = [col for col in expected if col not in df_head.columns]
        if missing:
            logger.error(f"Verification failed for {filename}: Missing columns {missing}")
            return False

        logger.info(f"Verification passed for {filename} ({len(df_head.columns)} columns verified).")
        return True
    except Exception as e:
        logger.error(f"Error inspecting {filename}: {e}")
        return False


def acquire_dataset(force_download: bool = False) -> bool:
    """
    Ensures all dataset files are present and verified in data/raw/.
    Downloads any missing files automatically.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for filename, url in DATASET_URLS.items():
        target_path = RAW_DATA_DIR / filename
        if not target_path.exists() or force_download:
            success = download_file(filename, url, target_path)
            if not success:
                all_ok = False
                continue

        if not verify_file(filename, target_path):
            all_ok = False

    if all_ok:
        logger.info("All dataset files are verified and ready in data/raw/!")
    else:
        logger.warning("Some dataset files could not be acquired or verified.")
    return all_ok


if __name__ == "__main__":
    acquire_dataset()
