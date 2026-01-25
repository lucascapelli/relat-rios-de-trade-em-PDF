"""Central registry for core application services."""
from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheManager
from .charts import ChartGenerator
from .database import Database
from .market_data import FinanceData
from .realtime import RealTimeManager
from .reports import ReportGenerator


@dataclass
class Services:

    finance_data: FinanceData
    chart_generator: ChartGenerator
    database: Database
    report_generator: ReportGenerator
    realtime_manager: RealTimeManager
    price_cache: CacheManager
    chart_cache: CacheManager


__all__ = ["Services"]
