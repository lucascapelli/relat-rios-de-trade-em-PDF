"""Portfolio domain models and calculations (new Carteiras session).

This module is intentionally isolated from any legacy portfolio logic. It provides
pure data structures and calculation helpers so future features (e.g. PDF
generation) can consume the same models without refactor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Optional


MAX_ASSETS = 5
DEFAULT_WINDOW_DAYS = 7


@dataclass
class PortfolioAsset:
    symbol: str
    entry: float
    objective: float
    stop_loss: float
    entry_maxima: Optional[float]
    entry_minima: Optional[float]
    ultimo_preco: Optional[float]
    percentual: float
    retorno_pct: float
    risco_pct: float
    risco_zero_preco: float
    risco_zero_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WeeklyPortfolio:
    start_date: date
    end_date: date
    assets: List[PortfolioAsset]

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "assets": [a.to_dict() for a in self.assets],
        }


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_asset(symbol: str, entry, objective, stop_loss, entry_maxima=None, entry_minima=None, ultimo_preco=None, total_assets: int = 1) -> PortfolioAsset:
    entry_f = _safe_float(entry) or 0.0
    objective_f = _safe_float(objective) or 0.0
    stop_f = _safe_float(stop_loss) or 0.0
    entry_max_f = _safe_float(entry_maxima)
    entry_min_f = _safe_float(entry_minima)
    last_price_f = _safe_float(ultimo_preco)

    if entry_max_f is None:
        entry_max_f = entry_f
    if entry_min_f is None:
        entry_min_f = entry_f

    # % = variação do último preço vs. entrada (como na planilha de referência).
    percentual = 0.0
    if entry_f > 0 and last_price_f is not None:
        percentual = round(((last_price_f - entry_f) / entry_f) * 100, 2)

    retorno_pct = 0.0
    risco_pct = 0.0
    if entry_f > 0:
        retorno_pct = round(((objective_f - entry_f) / entry_f) * 100, 2)
        risco_pct = round(((entry_f - stop_f) / entry_f) * 100, 2)

    # Risco zero: entrada + (entrada - stop)
    risco_zero_preco = entry_f + (entry_f - stop_f)
    risco_zero_pct = 0.0
    if entry_f > 0:
        risco_zero_pct = round(((risco_zero_preco - entry_f) / entry_f) * 100, 2)

    return PortfolioAsset(
        symbol=symbol.upper().strip(),
        entry=entry_f,
        objective=objective_f,
        stop_loss=stop_f,
        entry_maxima=entry_max_f,
        entry_minima=entry_min_f,
        ultimo_preco=last_price_f,
        percentual=percentual,
        retorno_pct=retorno_pct,
        risco_pct=risco_pct,
        risco_zero_preco=risco_zero_preco,
        risco_zero_pct=risco_zero_pct,
    )


def build_weekly_portfolio(raw_assets: List[dict], start_date: Optional[date] = None, end_date: Optional[date] = None, last_prices: Optional[dict] = None) -> WeeklyPortfolio:
    if raw_assets is None:
        raw_assets = []
    if len(raw_assets) == 0 or len(raw_assets) > MAX_ASSETS:
        raise ValueError("A carteira deve conter de 1 a 5 ativos")

    last_prices = last_prices or {}
    assets: List[PortfolioAsset] = []
    total_assets = len(raw_assets)

    for item in raw_assets:
        symbol = str(item.get("symbol") or "").upper().strip()
        if not symbol:
            raise ValueError("Campo 'symbol' é obrigatório para cada ativo")
        asset = compute_asset(
            symbol=symbol,
            entry=item.get("entrada"),
            objective=item.get("objetivo"),
            stop_loss=item.get("stop_loss"),
            entry_maxima=item.get("entrada_maxima"),
            entry_minima=item.get("entrada_minima"),
            ultimo_preco=last_prices.get(symbol),
            total_assets=total_assets,
        )
        assets.append(asset)

    start = start_date or date.today()
    end = end_date or (start + timedelta(days=DEFAULT_WINDOW_DAYS))

    return WeeklyPortfolio(start_date=start, end_date=end, assets=assets)
