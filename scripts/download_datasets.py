from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_activity_dss.data import download_datasets


if __name__ == "__main__":
    download_datasets(ROOT / "data" / "raw")
    print("Dataset download and extraction completed.")
