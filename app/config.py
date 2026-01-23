"""Application configuration constants and helpers."""
from __future__ import annotations
import pandas as pd

import os
from pathlib import Path
import pytz

# Base directories
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"
DB_PATH = PROJECT_ROOT / "database.db"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Timezones (uso lógico apenas, nunca em índice de gráfico)
BR_TZ = pytz.timezone("America/Sao_Paulo")
UTC_TZ = pytz.UTC

def normalize_datetime_index(df):
    """
    FORÇA DatetimeIndex para datetime64[ns] (naive).
    Elimina qualquer tz, independente da origem.
    """
    if not hasattr(df, "index"):
        return df

    # força conversão total
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = df.index.tz_convert(None)
    df.index = df.index.astype("datetime64[ns]")

    return df
# Price configuration
TICK_SIZE_MAP = {
    "WIN": 5.0,
    "IND": 5.0,
    "WDO": 0.5,
    "DOL": 0.5,
}
DEFAULT_TICK_SIZE = 0.01

# Static assets
PLACEHOLDER_IMAGE_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/lbexOwAAAABJRU5ErkJggg=="
)

# Caches
PRICE_CACHE_EXPIRY = 30
CHART_CACHE_EXPIRY = 15

# Flask settings
DEFAULT_SECRET_KEY = "sua-chave-secreta-aqui"

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "REPORTS_DIR",
    "CACHE_DIR",
    "DB_PATH",
    "BR_TZ",
    "UTC_TZ",
    "normalize_datetime_index",
    "TICK_SIZE_MAP",
    "DEFAULT_TICK_SIZE",
    "PLACEHOLDER_IMAGE_DATA_URL",
    "PRICE_CACHE_EXPIRY",
    "CHART_CACHE_EXPIRY",
    "DEFAULT_SECRET_KEY",
]
