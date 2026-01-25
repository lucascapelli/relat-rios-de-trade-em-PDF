"""Real-time updates manager built on top of Flask-SocketIO."""
from __future__ import annotations

import time
import re
from datetime import datetime
from typing import Dict, Optional, Set

from flask_socketio import SocketIO

from .config import BR_TZ
from .market_data import FinanceData
from .models import TickerInfo, asdict
from .utils import logger


class RealTimeManager:
    def __init__(self, socketio: SocketIO, finance_data: FinanceData, update_interval: int = 5) -> None:
        self.socketio = socketio
        self.finance_data = finance_data
        self.update_interval = update_interval
        self.active_symbols: Set[str] = set()
        self.symbol_data: Dict[str, Dict[str, object]] = {}
        self.running = False

    def _normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return symbol

        normalized = symbol.strip().upper()
        if normalized.endswith(".SA") or "." in normalized:
            return normalized

        # Only suffix .SA for common B3 equity/ETF tickers like PETR4 / VALE3 / BOVA11.
        # Avoid breaking futures like WING26 / WDOG26.
        if re.match(r"^[A-Z]{4}\d{1,2}$", normalized):
            return f"{normalized}.SA"

        return normalized

    def update_symbol(self, symbol: str) -> Optional[TickerInfo]:
        try:
            info = self.finance_data.get_ticker_info(symbol)

            if info:
                room_symbol = symbol
                display_symbol = room_symbol.replace(".SA", "")
                self.symbol_data[symbol] = {
                    **asdict(info),
                    "timestamp": datetime.now(BR_TZ).isoformat(),
                    "update_count": self.symbol_data.get(symbol, {}).get("update_count", 0) + 1,
                }

                self.socketio.emit(
                    "price_update",
                    {
                        "symbol": display_symbol,
                        "data": asdict(info),
                    },
                    room=f"symbol_{room_symbol}",
                )

                return info

        except Exception as exc:
            logger.error(f"Erro ao atualizar {symbol}: {exc}")

        return None

    def start_updates(self) -> None:
        self.running = True
        while self.running:
            try:
                symbols = list(self.active_symbols)

                if symbols:
                    for symbol in symbols:
                        self.update_symbol(symbol)

                time.sleep(self.update_interval)

            except Exception as exc:
                logger.error(f"Erro no loop de atualizações: {exc}")
                time.sleep(10)

    def stop_updates(self) -> None:
        self.running = False

    def subscribe(self, symbol: str) -> str:
        normalized = self._normalize_symbol(symbol)
        self.active_symbols.add(normalized)
        return normalized

    def unsubscribe(self, symbol: str) -> str:
        normalized = self._normalize_symbol(symbol)
        if normalized in self.active_symbols:
            self.active_symbols.remove(normalized)
        return normalized
