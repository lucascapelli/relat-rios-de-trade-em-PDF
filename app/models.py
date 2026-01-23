"""Data models used across the application."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .config import BR_TZ, DEFAULT_TICK_SIZE, TICK_SIZE_MAP


__all__ = ["TickerInfo", "Operation", "Candle", "asdict"]


@dataclass
class TickerInfo:
    symbol: str
    price: float
    change: float
    change_percent: float
    open: float
    high: float
    low: float
    volume: int
    previous_close: float
    name: str
    currency: str = "BRL"
    market_cap: float = 0
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(BR_TZ).strftime("%H:%M:%S")


@dataclass
class Operation:
    symbol: str
    tipo: str
    entrada: float
    stop: float
    alvo: float
    quantidade: int
    timeframe: str = "15m"
    observacoes: str = ""
    entrada_min: Optional[float] = None
    entrada_max: Optional[float] = None
    parcial_preco: Optional[float] = None
    parcial_pontos: Optional[float] = None
    tick_size: Optional[float] = None
    preco_atual: float = 0
    pontos_alvo: float = 0
    pontos_stop: float = 0
    risco_retorno: Optional[float] = None
    status: str = "ABERTA"
    created_at: Optional[str] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    context_event: Optional[str] = None
    context_level: Optional[float] = None
    narrative_text: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(BR_TZ).isoformat()

        self.tipo = (self.tipo or "COMPRA").upper()

        if self.tick_size is None or self.tick_size <= 0:
            self.tick_size = self._infer_tick_size(self.symbol)

        faixa = [value for value in [self.entrada_min, self.entrada_max] if value is not None]
        if not faixa:
            self.entrada_min = self.entrada
            self.entrada_max = self.entrada
        else:
            self.entrada_min = min(faixa)
            self.entrada_max = max(faixa)

        referencia = self.entrada
        if referencia is None and self.entrada_min is not None and self.entrada_max is not None:
            referencia = (self.entrada_min + self.entrada_max) / 2

        if referencia is None:
            referencia = 0

        alvo_diff, stop_diff = self._directional_diffs(referencia)
        self.pontos_alvo = self._price_diff_to_ticks(alvo_diff)
        self.pontos_stop = self._price_diff_to_ticks(stop_diff)

        if self.parcial_preco is not None:
            parcial_diff = self._directional_diff_single(referencia, self.parcial_preco)
            self.parcial_pontos = self._price_diff_to_ticks(parcial_diff)

        if self.pontos_stop > 0:
            self.risco_retorno = round(self.pontos_alvo / self.pontos_stop, 2) if self.pontos_alvo else None

    @staticmethod
    def _infer_tick_size(symbol: str) -> float:
        clean = "".join(ch for ch in symbol.upper() if ch.isalpha())
        for prefix, size in TICK_SIZE_MAP.items():
            if clean.startswith(prefix):
                return size
        return DEFAULT_TICK_SIZE

    def _directional_diffs(self, referencia: float) -> Tuple[float, float]:
        if self.tipo == "COMPRA":
            alvo_diff = max((self.alvo or referencia) - referencia, 0)
            stop_diff = max(referencia - (self.stop or referencia), 0)
        else:
            alvo_diff = max(referencia - (self.alvo or referencia), 0)
            stop_diff = max((self.stop or referencia) - referencia, 0)
        return alvo_diff, stop_diff

    def _directional_diff_single(self, referencia: float, target_price: float) -> float:
        if self.tipo == "COMPRA":
            return max(target_price - referencia, 0)
        return max(referencia - target_price, 0)

    def _price_diff_to_ticks(self, diff: float) -> float:
        if self.tick_size is None or self.tick_size <= 0:
            return 0
        return round(diff / self.tick_size, 2)


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "time_str": self.time.strftime("%Y-%m-%d %H:%M:%S"),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
