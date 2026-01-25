"""Application configuration constants and helpers."""
from __future__ import annotations
import pandas as pd

import importlib.util
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

# Institution / branding (can be overridden by project-root config.py)
DEFAULT_INSTITUTION_NAME = "Relatórios Trade"
DEFAULT_LOGO_PATH = "static/logo.png"
DEFAULT_INSTITUTION_TEXT = ""


def _load_external_institution_config(project_root: Path) -> dict:
    external_path = project_root / "config.py"
    if not external_path.exists() or not external_path.is_file():
        return {}

    try:
        spec = importlib.util.spec_from_file_location("relatoriostrade_external_config", external_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            "INSTITUTION_NAME": getattr(module, "INSTITUTION_NAME", None),
            "LOGO_PATH": getattr(module, "LOGO_PATH", None),
            "INSTITUTION_TEXT": getattr(module, "INSTITUTION_TEXT", None),
        }
    except Exception:
        return {}


_external_institution = _load_external_institution_config(PROJECT_ROOT)

INSTITUTION_NAME = str(_external_institution.get("INSTITUTION_NAME") or DEFAULT_INSTITUTION_NAME)
LOGO_PATH = str(_external_institution.get("LOGO_PATH") or DEFAULT_LOGO_PATH)
INSTITUTION_TEXT = str(_external_institution.get("INSTITUTION_TEXT") or DEFAULT_INSTITUTION_TEXT)

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
    "INSTITUTION_NAME",
    "LOGO_PATH",
    "INSTITUTION_TEXT",
]
