"""Technical indicator calculations and contextual narrative helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta as ta

from .config import DEFAULT_TICK_SIZE
from .models import Operation
from .utils import logger


class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["SMA_9"] = TechnicalIndicators._safe_calc(ta.sma, df["close"], 9)
        df["SMA_21"] = TechnicalIndicators._safe_calc(ta.sma, df["close"], 21)
        df["SMA_50"] = TechnicalIndicators._safe_calc(ta.sma, df["close"], 50)

        df["EMA_12"] = TechnicalIndicators._safe_calc(ta.ema, df["close"], 12)
        df["EMA_26"] = TechnicalIndicators._safe_calc(ta.ema, df["close"], 26)
        df["EMA_21"] = TechnicalIndicators._safe_calc(ta.ema, df["close"], 21)
        df["EMA_200"] = TechnicalIndicators._safe_calc(ta.ema, df["close"], 200)

        df["RSI"] = TechnicalIndicators._safe_calc(ta.rsi, df["close"], 14)

        macd = TechnicalIndicators._safe_calc(ta.macd, df["close"])
        if macd is not None and not macd.empty:
            TechnicalIndicators._add_macd_columns(df, macd)

        bb = TechnicalIndicators._safe_calc(ta.bbands, df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            TechnicalIndicators._add_bb_columns(df, bb)

        df["ATR"] = TechnicalIndicators._safe_calc(ta.atr, df["high"], df["low"], df["close"], 14)
        df["Volume_SMA"] = TechnicalIndicators._safe_calc(ta.sma, df["volume"], 20)

        if len(df) > 20:
            df["Resistance"] = df["high"].rolling(window=20).max()
            df["Support"] = df["low"].rolling(window=20).min()

        return df

    @staticmethod
    def _safe_calc(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.debug(f"Erro ao calcular indicador {func.__name__}: {exc}")
            return None

    @staticmethod
    def _add_macd_columns(df: pd.DataFrame, macd: pd.DataFrame) -> None:
        for col in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]:
            if col in macd.columns:
                df[col.replace("_12_26_9", "")] = macd[col]

    @staticmethod
    def _add_bb_columns(df: pd.DataFrame, bb: pd.DataFrame) -> None:
        column_map = {
            "BBU_20_2.0": "BB_upper",
            "BBM_20_2.0": "BB_middle",
            "BBL_20_2.0": "BB_lower",
        }
        for old_col, new_col in column_map.items():
            if old_col in bb.columns:
                df[new_col] = bb[old_col]


class TechnicalNarrative:
    @staticmethod
    def _normalize_timeframe(tf: Optional[str]) -> str:
        if not tf:
            return ""
        mapping = {
            "6min": "6m",
            "6m": "6m",
            "15min": "15m",
            "15": "15m",
            "60min": "1h",
            "60": "1h",
            "60m": "1h",
            "diario": "1d",
            "daily": "1d",
            "dia": "1d",
            "1dia": "1d",
            "weekly": "1w",
            "semanal": "1w",
        }
        tf_norm = str(tf).strip().lower()
        return mapping.get(tf_norm, tf_norm)

    @staticmethod
    def _label_for(tf: str) -> str:
        labels = {
            "6m": "6M",
            "15m": "15M",
            "1h": "60M",
            "1d": "DIÁRIO",
            "1w": "SEMANAL",
        }
        return labels.get(tf, tf.upper())

    @staticmethod
    def _fmt_price(value: Optional[float]) -> str:
        if value is None:
            return "R$ -"
        return f"R$ {value:.2f}"

    @staticmethod
    def analyze(df: Optional[pd.DataFrame], operation: Operation) -> Dict[str, Any]:
        if df is None or df.empty or "close" not in df.columns:
            return {
                "support": None,
                "resistance": None,
                "event": None,
                "level": None,
                "text": None,
            }

        recent = df.tail(180).copy()
        if recent.empty:
            return {
                "support": None,
                "resistance": None,
                "event": None,
                "level": None,
                "text": None,
            }

        for col in ["Support", "Resistance"]:
            if col not in recent.columns:
                if col == "Support":
                    recent[col] = recent["low"].rolling(window=20, min_periods=10).min()
                else:
                    recent[col] = recent["high"].rolling(window=20, min_periods=10).max()

        support_series = recent["Support"].dropna()
        resistance_series = recent["Resistance"].dropna()
        support_level = float(support_series.iloc[-1]) if not support_series.empty else None
        resistance_level = float(resistance_series.iloc[-1]) if not resistance_series.empty else None

        last_close = float(recent["close"].iloc[-1])
        tick_size = operation.tick_size or DEFAULT_TICK_SIZE
        tolerance = tick_size * 8 if tick_size else max(last_close * 0.002, 0.5)

        def _last_touch(series: pd.Series, level: Optional[float], price_field: str) -> Optional[int]:
            if level is None:
                return None
            prices = recent[price_field]
            proximity = (prices - level).abs() <= tolerance
            indices = proximity[proximity].index
            if len(indices) == 0:
                return None
            return recent.index.get_loc(indices[-1])

        last_support_touch_idx = _last_touch(recent["low"], support_level, "low")
        last_resistance_touch_idx = _last_touch(recent["high"], resistance_level, "high")

        bars_since_support = (
            len(recent) - 1 - last_support_touch_idx
        ) if last_support_touch_idx is not None else None
        bars_since_resistance = (
            len(recent) - 1 - last_resistance_touch_idx
        ) if last_resistance_touch_idx is not None else None

        normalized_tf = TechnicalNarrative._normalize_timeframe(operation.timeframe)
        tf_label = TechnicalNarrative._label_for(normalized_tf)
        direction = operation.tipo.upper() if operation.tipo else "COMPRA"

        event = None
        event_level = None
        text = None

        if direction == "COMPRA":
            if support_level is not None and bars_since_support is not None and bars_since_support <= 12:
                event = "support_retest"
                event_level = support_level
                text = (
                    f"O ativo {operation.symbol} demonstra viés técnico positivo após teste e reação "
                    f"no suporte na região dos {TechnicalNarrative._fmt_price(support_level)}, "
                    f"favorecendo {direction} no timeframe de {tf_label}."
                )
            elif resistance_level is not None and last_close > resistance_level + tolerance:
                event = "resistance_breakout"
                event_level = resistance_level
                text = (
                    f"O ativo {operation.symbol} confirma rompimento da resistência em "
                    f"{TechnicalNarrative._fmt_price(resistance_level)}, favorecendo {direction} "
                    f"no timeframe de {tf_label}."
                )
        else:
            if (
                resistance_level is not None
                and bars_since_resistance is not None
                and bars_since_resistance <= 12
            ):
                event = "resistance_rejection"
                event_level = resistance_level
                text = (
                    f"O ativo {operation.symbol} demonstra viés técnico negativo após rejeição "
                    f"na resistência na região dos {TechnicalNarrative._fmt_price(resistance_level)}, "
                    f"favorecendo {direction} no timeframe de {tf_label}."
                )
            elif support_level is not None and last_close < support_level - tolerance:
                event = "support_breakdown"
                event_level = support_level
                text = (
                    f"O ativo {operation.symbol} rompeu o suporte em {TechnicalNarrative._fmt_price(support_level)}, "
                    f"favorecendo {direction} no timeframe de {tf_label}."
                )

        if text is None:
            if direction == "COMPRA" and support_level is not None:
                text = (
                    f"O ativo {operation.symbol} mantém estrutura de alta acima do suporte em "
                    f"{TechnicalNarrative._fmt_price(support_level)}, favorecendo {direction} no timeframe de {tf_label}."
                )
                event = "support_reference"
                event_level = support_level
            elif direction == "VENDA" and resistance_level is not None:
                text = (
                    f"O ativo {operation.symbol} mantém pressão vendedora após reação na resistência em "
                    f"{TechnicalNarrative._fmt_price(resistance_level)}, favorecendo {direction} no timeframe de {tf_label}."
                )
                event = "resistance_reference"
                event_level = resistance_level

        return {
            "support": support_level,
            "resistance": resistance_level,
            "event": event,
            "level": event_level,
            "text": text,
        }
