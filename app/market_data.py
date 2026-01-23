"""Market data access layer built on top of yfinance with fallbacks."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import BR_TZ
from .indicators import TechnicalIndicators
from .models import TickerInfo
from .utils import logger


class FinanceData:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        )
        self.tickers = self._load_tickers()
        self.indicators = TechnicalIndicators()

    def _load_tickers(self) -> Dict[str, str]:
        try:
            url = "https://raw.githubusercontent.com/guilhermecappi/b3-tickers/master/data/tickers.csv"
            df = pd.read_csv(url)
            return dict(zip(df["ticker"], df["company"]))
        except Exception as exc:
            logger.warning(f"Erro ao carregar tickers: {exc}")
            return self._get_default_tickers()

    def _get_default_tickers(self) -> Dict[str, str]:
        return {
            "PETR4.SA": "Petrobras PN",
            "VALE3.SA": "Vale ON",
            "ITUB4.SA": "Itaú Unibanco PN",
            "BBDC4.SA": "Bradesco PN",
            "BBAS3.SA": "Banco do Brasil ON",
            "ABEV3.SA": "Ambev ON",
            "WEGE3.SA": "Weg ON",
            "MGLU3.SA": "Magazine Luiza ON",
            "VIIA3.SA": "Via ON",
            "BOVA11.SA": "ETF Ibovespa",
        }

    def get_ticker_info(self, symbol: str) -> TickerInfo:
        try:
            if not symbol.endswith(".SA") and symbol[-1].isdigit():
                symbol = f"{symbol}.SA"

            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d", interval="5m")

            if hist.empty:
                return self._create_fallback_info(symbol)

            return self._parse_ticker_info(symbol, info, hist)
        except Exception as exc:
            logger.error(f"Erro ao buscar info para {symbol}: {exc}")
            return self._create_fallback_info(symbol)

    def _parse_ticker_info(self, symbol: str, info: Dict, hist: pd.DataFrame) -> TickerInfo:
        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close

        return TickerInfo(
            symbol=symbol.replace(".SA", ""),
            price=last_close,
            change=last_close - prev_close,
            change_percent=((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0,
            open=float(hist["Open"].iloc[-1]) if "Open" in hist.columns else last_close,
            high=float(hist["High"].iloc[-1]) if "High" in hist.columns else last_close,
            low=float(hist["Low"].iloc[-1]) if "Low" in hist.columns else last_close,
            volume=int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
            previous_close=prev_close,
            name=info.get("longName", info.get("shortName", symbol)),
            currency=info.get("currency", "BRL"),
            market_cap=info.get("marketCap", 0),
        )

    def _create_fallback_info(self, symbol: str) -> TickerInfo:
        return TickerInfo(
            symbol=symbol.replace(".SA", ""),
            price=100.0,
            change=0,
            change_percent=0,
            open=100.0,
            high=101.0,
            low=99.0,
            volume=1000000,
            previous_close=100.0,
            name=symbol,
            currency="BRL",
        )

    def get_candles(self, symbol: str, interval: str = "15m", periods: int = 100) -> pd.DataFrame:
        try:
            interval_norm = str(interval).strip().lower()
            symbol_yf = f"{symbol}.SA" if not symbol.endswith(".SA") and symbol[-1].isdigit() else symbol
            df = self._fetch_yfinance_data(symbol_yf, interval_norm, periods)

            if df is None or df.empty:
                logger.warning(f"Sem dados reais para {symbol} ({interval}), usando fallback")
                return self._generate_fallback_data(symbol, interval, periods, reason="fallback_no_data")

            df = self._process_candle_data(df, symbol_yf, periods)

            if interval_norm in {"6m", "6min"}:
                df = self._resample_ohlc(df, "6min")
                if len(df) > periods:
                    df = df.iloc[-periods:]
                df["time"] = df.index
                df["time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df = self.indicators.calculate_all(df)

            valid = df[["open", "high", "low", "close"]].dropna()
            if len(valid) < 2:
                logger.warning(
                    f"Candles insuficientes para {symbol} ({interval}): {len(valid)} disponíveis; usando fallback"
                )
                return self._generate_fallback_data(symbol, interval, periods, reason="fallback_insufficient")

            df.attrs["source"] = "yfinance"
            self._log_dataset_snapshot(df, symbol, interval, source="yfinance")

            return df
        except Exception as exc:
            logger.error(f"Erro ao buscar candles para {symbol}: {exc}")
            return self._generate_fallback_data(symbol, interval, periods, reason="fallback_error")

    def _fetch_yfinance_data(self, symbol: str, interval: str, periods: int) -> Optional[pd.DataFrame]:
        interval_map = {
            "1m": ("1m", "1d"),
            "6m": ("1m", "7d"),
            "6min": ("1m", "7d"),
            "5m": ("5m", "5d"),
            "15m": ("15m", "5d"),
            "30m": ("30m", "10d"),
            "1h": ("60m", "30d"),
            "60m": ("60m", "30d"),
            "1d": ("1d", "3mo"),
            "1w": ("1wk", "2y"),
            "15min": ("15m", "5d"),
            "60min": ("60m", "30d"),
            "daily": ("1d", "3mo"),
        }

        yf_interval, yf_period = interval_map.get(interval, ("15m", "5d"))

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=yf_period, interval=yf_interval, timeout=10)
            return df if df is not None and not df.empty else None
        except Exception as exc:
            logger.error(f"Erro yfinance para {symbol}: {exc}")
            return None

    @staticmethod
    def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        if df is None or df.empty or not required.issubset(df.columns):
            return df

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            agg["volume"] = "sum"

        return df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])

    def _process_candle_data(self, df: pd.DataFrame, symbol: str, periods: int) -> pd.DataFrame:
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        if df.index.tz is not None:
            df.index = df.index.tz_convert(BR_TZ)
        else:
            df.index = df.index.tz_localize("UTC").tz_convert(BR_TZ)

        if len(df) > periods:
            df = df.iloc[-periods:]

        df["time"] = df.index
        df["time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        df = self._validate_candles(df)

        logger.info(f"Dados carregados para {symbol}: R$ {df['close'].iloc[-1]:.2f} ({len(df)} candles)")

        return df

    def _validate_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        mask = (
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        )

        return df[mask].copy()

    def _log_dataset_snapshot(self, df: pd.DataFrame, symbol: str, interval: str, source: str) -> None:
        if df is None or df.empty:
            logger.warning(f"[MarketData] {source} {symbol} ({interval}): dataset vazio")
            return

        first = df.iloc[0]
        last = df.iloc[-1]

        def safe_time(row: pd.Series) -> str:
            if "time_str" in row.index:
                return str(row["time_str"])
            if "time" in row.index:
                return str(row["time"])
            return str(row.name)

        def safe_value(row: pd.Series, key: str) -> Optional[float]:
            if key in row.index and not pd.isna(row[key]):
                value = row[key]
                if isinstance(value, (int, float, np.floating)):
                    return float(value)
            return None

        def fmt(value: Optional[float]) -> str:
            return f"{value:.2f}" if value is not None else "-"

        msg = (
            f"[MarketData] {source} {symbol} ({interval}) -> registros={len(df)} "
            f"primeiro={safe_time(first)} O={fmt(safe_value(first, 'open'))} C={fmt(safe_value(first, 'close'))} "
            f"último={safe_time(last)} O={fmt(safe_value(last, 'open'))} C={fmt(safe_value(last, 'close'))}"
        )
        logger.info(msg)

    def _generate_fallback_data(self, symbol: str, interval: str, periods: int, reason: str = "fallback") -> pd.DataFrame:
        config = {
            "1m": {"freq": "1min", "vol": 0.002},
            "5m": {"freq": "5min", "vol": 0.005},
            "15m": {"freq": "15min", "vol": 0.008},
            "30m": {"freq": "30min", "vol": 0.012},
            "1h": {"freq": "1h", "vol": 0.015},
            "1d": {"freq": "1D", "vol": 0.02},
            "1w": {"freq": "1W", "vol": 0.03},
        }

        cfg = config.get(interval, config["15m"])
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=cfg["freq"])

        base_price = 100 + (hash(symbol) % 100)
        trend_direction = 1 if (hash(symbol) % 2) == 0 else -1
        trend = np.linspace(0, trend_direction * cfg["vol"] * 20, periods)
        noise = np.random.normal(0, cfg["vol"], periods)
        close_prices = base_price * (1 + trend + noise.cumsum())

        data = []
        for i in range(periods):
            open_price = close_prices[i - 1] if i > 0 else close_prices[i] * (1 + np.random.uniform(-0.005, 0.005))
            close_price_val = close_prices[i]

            is_bullish = close_price_val >= open_price
            body_range = abs(close_price_val - open_price)
            wick_range = body_range + base_price * cfg["vol"] * 2

            if is_bullish:
                low = open_price - np.random.uniform(0, wick_range * 0.3)
                high = close_price_val + np.random.uniform(0, wick_range * 0.7)
            else:
                low = close_price_val - np.random.uniform(0, wick_range * 0.7)
                high = open_price + np.random.uniform(0, wick_range * 0.3)

            high = max(high, low + 0.01)
            open_price = max(low, min(high, open_price))
            close_price_val = max(low, min(high, close_price_val))

            volume = np.random.randint(10000, 1000000) * (1 + body_range / base_price)

            data.append(
                {
                    "time": dates[i],
                    "time_str": dates[i].strftime("%Y-%m-%d %H:%M:%S"),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_price_val, 2),
                    "volume": int(volume),
                }
            )

        df = pd.DataFrame(data)
        df = self.indicators.calculate_all(df)
        df.attrs["source"] = reason
        self._log_dataset_snapshot(df, symbol, interval, source=reason)
        logger.info(f"Fallback gerado para {symbol} ({interval}) com razão '{reason}': {len(df)} candles")

        return df
