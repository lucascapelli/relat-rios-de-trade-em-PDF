"""HTTP routes for the reporting system."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly
from flask import Blueprint, jsonify, render_template, request, send_from_directory

from ..config import PLACEHOLDER_IMAGE_DATA_URL, REPORTS_DIR
from ..indicators import TechnicalNarrative
from ..models import Operation, asdict
from ..services import Services
from ..utils import logger


def _build_chart_components(df: pd.DataFrame, limit: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    df_tail = df.iloc[-limit:].copy()
    candles: List[Dict[str, Any]] = []
    series: Dict[str, List[Any]] = {
        "time": [],
        "time_str": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }

    indicator_columns = {
        "sma_9": "SMA_9",
        "sma_21": "SMA_21",
        "ema_12": "EMA_12",
        "ema_26": "EMA_26",
        "rsi_series": "RSI",
        "bb_upper": "BB_upper",
        "bb_middle": "BB_middle",
        "bb_lower": "BB_lower",
    }

    for key, col in indicator_columns.items():
        if col in df_tail.columns:
            series[key] = []

    for _, row in df_tail.iterrows():
        volume_value = row.get("volume", 0)
        if pd.isna(volume_value):
            volume_value = 0

        iso_time = row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"])
        time_str = row["time_str"] if "time_str" in row.index else iso_time

        candles.append(
            {
                "time": iso_time,
                "time_str": time_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(volume_value),
            }
        )

        series["time"].append(iso_time)
        series["time_str"].append(time_str)
        series["open"].append(float(row["open"]))
        series["high"].append(float(row["high"]))
        series["low"].append(float(row["low"]))
        series["close"].append(float(row["close"]))
        series["volume"].append(int(volume_value))

        for key, col in indicator_columns.items():
            if key in series:
                value = row.get(col)
                series[key].append(float(value) if value is not None and not pd.isna(value) else None)

    return candles, series


def register_api_routes(app, services: Services) -> None:
    bp = Blueprint("core", __name__)

    finance_data = services.finance_data
    chart_generator = services.chart_generator
    database = services.database
    report_generator = services.report_generator

    @bp.route("/")
    def index():
        symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4"]
        stocks: List[Dict[str, Any]] = []

        for sym in symbols:
            try:
                info = finance_data.get_ticker_info(sym)
                if info:
                    stocks.append(asdict(info))
            except Exception as exc:
                logger.error(f"Erro ao buscar {sym}: {exc}")

        try:
            df = finance_data.get_candles("PETR4", "15m", 50)
            fig = chart_generator.create_plotly_chart(df, "PETR4 - 15 Minutos")
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as exc:
            logger.error(f"Erro no gráfico inicial: {exc}")
            graph_json = "{}"

        return render_template(
            "index.html",
            stocks=stocks[:4],
            graphJSON=graph_json,
            timeframe="15m",
        )

    @bp.route("/api/quote/<symbol>")
    def api_quote(symbol: str):
        try:
            info = finance_data.get_ticker_info(symbol)
            return jsonify(asdict(info))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/chart/<symbol>/<interval>")
    def api_chart(symbol: str, interval: str):
        try:
            df = finance_data.get_candles(symbol, interval, 100)
            fig = chart_generator.create_plotly_chart(df, f"{symbol} - {interval}")

            candles_payload, series = _build_chart_components(df, limit=100)

            source = df.attrs.get("source", "unknown")
            if candles_payload:
                last_candle = candles_payload[-1]
                last_open = float(last_candle.get("open", 0) or 0)
                last_close = float(last_candle.get("close", 0) or 0)
                logger.info(
                    f"/api/chart {symbol}/{interval} -> source={source} candles={len(candles_payload)} "
                    f"último={last_candle.get('time_str')} O={last_open:.2f} C={last_close:.2f}"
                )
            else:
                logger.warning(f"/api/chart {symbol}/{interval} -> source={source} sem candles retornados")

            indicators_snapshot = {
                "sma_9": float(df["SMA_9"].iloc[-1]) if "SMA_9" in df.columns else None,
                "sma_21": float(df["SMA_21"].iloc[-1]) if "SMA_21" in df.columns else None,
                "rsi": float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None,
            }

            return jsonify(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "source": source,
                    "candles": candles_payload,
                    "series": series,
                    "indicators": indicators_snapshot,
                    "chart_data": json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                }
            )

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/operacao", methods=["POST"])
    def api_operation():
        try:
            data = request.get_json() or {}

            def _get_first(keys: List[str]):
                for key in keys:
                    if key in data and data[key] is not None:
                        return data[key]
                return None

            def _parse_float(value: Any) -> Optional[float]:
                try:
                    if value is None or value == "":
                        return None
                    return float(value)
                except (TypeError, ValueError):
                    return None

            raw_symbol = _get_first(["ativo", "symbol", "ticker"]) or ""
            symbol = (raw_symbol or "").upper()
            info = finance_data.get_ticker_info(symbol)

            if not info:
                return jsonify({"error": "Símbolo não encontrado"}), 400

            entrada = _parse_float(_get_first(["entrada", "entrada_base"]))
            stop = _parse_float(_get_first(["stop", "stop_loss"]))
            alvo = _parse_float(_get_first(["alvo", "target"]))

            if entrada is None or stop is None or alvo is None:
                return jsonify({"error": "Campos de preço obrigatórios ausentes"}), 400

            entrada_min = _parse_float(_get_first(["entrada_min", "entradaMin", "faixa_min"]))
            entrada_max = _parse_float(_get_first(["entrada_max", "entradaMax", "faixa_max"]))

            if entrada_min is None and entrada_max is not None:
                entrada_min = entrada
            if entrada_max is None and entrada_min is not None:
                entrada_max = entrada
            if entrada_min is None and entrada_max is None:
                entrada_min = entrada
                entrada_max = entrada

            parcial_preco = _parse_float(_get_first(["saida_parcial", "parcial", "parcial_preco"]))
            tick_size = _parse_float(_get_first(["tick_size", "tickSize"]))

            timeframe = (_get_first(["timeframe", "timeframe_base"]) or "15m").lower()

            operation = Operation(
                symbol=symbol,
                tipo=data.get("tipo", "COMPRA"),
                entrada=entrada,
                stop=stop,
                alvo=alvo,
                quantidade=int(data.get("quantidade", 100)),
                observacoes=data.get("observacoes", ""),
                preco_atual=info.price,
                timeframe=timeframe,
                entrada_min=entrada_min,
                entrada_max=entrada_max,
                parcial_preco=parcial_preco,
                tick_size=tick_size,
            )

            if operation.tipo == "COMPRA":
                if operation.preco_atual >= operation.alvo:
                    operation.status = "ALVO ATINGIDO"
                elif operation.preco_atual <= operation.stop:
                    operation.status = "STOP ATINGIDO"
            else:
                if operation.preco_atual <= operation.alvo:
                    operation.status = "ALVO ATINGIDO"
                elif operation.preco_atual >= operation.stop:
                    operation.status = "STOP ATINGIDO"

            preferred_timeframes: List[str] = []
            base_timeframes = ["15m", "1h", "1d", "1w"]
            if operation.timeframe and operation.timeframe not in base_timeframes:
                preferred_timeframes.append(operation.timeframe)
            for candidate in base_timeframes:
                if candidate not in preferred_timeframes:
                    preferred_timeframes.append(candidate)

            chart_images: List[Dict[str, Any]] = []
            indicator_df: Optional[pd.DataFrame] = None

            for tf in preferred_timeframes:
                df_real = finance_data.get_candles(operation.symbol, tf, 150)
                df = df_real

                def _has_candles(frame: Optional[pd.DataFrame]) -> bool:
                    if frame is None or frame.empty:
                        return False
                    valid = frame[["open", "high", "low", "close"]].dropna()
                    return len(valid.index) >= 2

                if not _has_candles(df):
                    logger.warning(
                        "Dataset insuficiente para %s no timeframe %s; usando fallback",
                        operation.symbol,
                        tf,
                    )
                    df = finance_data._generate_fallback_data(operation.symbol, tf, 150, reason=f"pdf_fallback_{tf}")

                if indicator_df is None and _has_candles(df_real):
                    indicator_df = df_real
                if indicator_df is None and _has_candles(df):
                    indicator_df = df

                img = chart_generator.generate_chart_image(
                    df,
                    f"{operation.symbol} - {tf}",
                    operation=operation,
                )

                last_row = df.iloc[-1] if len(df.index) > 0 else None
                chart_images.append(
                    {
                        "timeframe": tf,
                        "image": img if img else PLACEHOLDER_IMAGE_DATA_URL,
                        "source": df.attrs.get("source", "desconhecido"),
                        "last_close": float(last_row["close"]) if last_row is not None and "close" in df.columns else None,
                        "last_time": str(last_row["time_str"]) if last_row is not None and "time_str" in df.columns else None,
                    }
                )

                if not img:
                    logger.warning(
                        "Não foi possível renderizar imagem para %s em %s mesmo após fallback",
                        operation.symbol,
                        tf,
                    )

            if not chart_images:
                logger.warning(
                    "Nenhum gráfico gerado para %s no timeframe %s; utilizando fallback sintético",
                    operation.symbol,
                    operation.timeframe,
                )
                fallback_df = finance_data._generate_fallback_data(
                    operation.symbol,
                    operation.timeframe,
                    120,
                    reason="pdf_fallback",
                )
                fallback_img = chart_generator.generate_chart_image(
                    fallback_df,
                    f"{operation.symbol} - {operation.timeframe}",
                    operation=operation,
                )
                fallback_row = fallback_df.iloc[-1] if len(fallback_df.index) > 0 else None
                chart_images.append(
                    {
                        "timeframe": operation.timeframe,
                        "image": fallback_img if fallback_img else PLACEHOLDER_IMAGE_DATA_URL,
                        "source": fallback_df.attrs.get("source", "fallback"),
                        "last_close": float(fallback_row["close"]) if fallback_row is not None and "close" in fallback_df.columns else None,
                        "last_time": str(fallback_row["time_str"]) if fallback_row is not None and "time_str" in fallback_df.columns else None,
                    }
                )
                if indicator_df is None:
                    indicator_df = fallback_df

            if indicator_df is not None and not indicator_df.empty:
                context_info = TechnicalNarrative.analyze(indicator_df, operation)
                operation.support_level = context_info.get("support")
                operation.resistance_level = context_info.get("resistance")
                operation.context_event = context_info.get("event")
                operation.context_level = context_info.get("level")
                operation.narrative_text = context_info.get("text")

            pdf_path = report_generator.generate_pdf_report(operation, chart_images)

            indicators = {
                "sma_9": float(indicator_df["SMA_9"].iloc[-1]) if indicator_df is not None and "SMA_9" in indicator_df.columns else None,
                "sma_21": float(indicator_df["SMA_21"].iloc[-1]) if indicator_df is not None and "SMA_21" in indicator_df.columns else None,
                "rsi": float(indicator_df["RSI"].iloc[-1]) if indicator_df is not None and "RSI" in indicator_df.columns else None,
            }

            op_id = database.insert_operation(operation, pdf_path, indicators)

            return jsonify(
                {
                    "id": op_id,
                    "status": "success",
                    "pdf_url": f"/reports/{os.path.basename(pdf_path)}",
                    "operation": asdict(operation),
                }
            )

        except Exception as exc:
            logger.error(f"Erro ao registrar operação: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/search/<query>")
    def api_search(query: str):
        results: List[Dict[str, str]] = []

        for symbol, name in finance_data.tickers.items():
            if query.upper() in symbol or query.upper() in name.upper():
                results.append(
                    {
                        "symbol": symbol.replace(".SA", ""),
                        "name": name,
                        "type": "Ação" if ".SA" in symbol else "ETF",
                    }
                )

        return jsonify(results[:10])

    @bp.route("/api/history")
    def api_history():
        try:
            operations = database.get_operations(50)
            return jsonify(operations)
        except Exception as exc:
            logger.error(f"Erro ao buscar histórico: {exc}")
            return jsonify([])

    @bp.route("/reports/<path:filename>")
    def serve_report(filename: str):
        return send_from_directory(str(REPORTS_DIR), filename)

    app.register_blueprint(bp)
