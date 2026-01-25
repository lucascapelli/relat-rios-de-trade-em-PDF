"""Socket.IO event handlers."""
from __future__ import annotations

import json

import plotly
from flask_socketio import SocketIO, emit, join_room, leave_room

from ..charts import ChartGenerator, format_timeframe_label
from ..models import asdict
from ..services import Services
from ..utils import logger
from .api import _build_chart_components


def register_socket_handlers(socketio: SocketIO, services: Services) -> None:

    finance_data = services.finance_data
    realtime_manager = services.realtime_manager
    chart_generator = services.chart_generator

    @socketio.on("subscribe")
    def handle_subscribe(data):
        symbol = data.get("symbol") if isinstance(data, dict) else None

        if symbol:
            room_symbol = realtime_manager.subscribe(symbol)
            join_room(f"symbol_{room_symbol}")

            info = finance_data.get_ticker_info(symbol)
            if info:
                emit(
                    "price_update",
                    {
                        "symbol": info.symbol,
                        "data": asdict(info),
                    },
                )

            emit(
                "subscription_confirmed",
                {
                    "symbol": symbol,
                    "message": f"Inscrito em {symbol}",
                },
            )

    @socketio.on("unsubscribe")
    def handle_unsubscribe(data):
        symbol = data.get("symbol") if isinstance(data, dict) else None

        if symbol:
            room_symbol = realtime_manager.unsubscribe(symbol)
            leave_room(f"symbol_{room_symbol}")

            emit(
                "unsubscription_confirmed",
                {
                    "symbol": symbol,
                    "message": f"Inscrição cancelada para {symbol}",
                },
            )

    @socketio.on("request_chart")
    def handle_chart_request(data):
        try:
            symbol = data.get("symbol") if isinstance(data, dict) else None
            interval = data.get("interval", "15m") if isinstance(data, dict) else "15m"
            limit_raw = data.get("limit") if isinstance(data, dict) else None
            try:
                limit_value = int(limit_raw) if limit_raw is not None else 100
            except (TypeError, ValueError):
                limit_value = 100
            limit = max(50, min(limit_value, 1500))

            if not symbol:
                emit("chart_error", {"error": "Símbolo não informado"})
                return

            df_raw = finance_data.get_candles(symbol, interval, limit)
            df_norm = ChartGenerator.prepare_ohlc_dataframe(df_raw, interval)
            df_norm = ChartGenerator._ensure_overlay_columns(df_norm)
            payload_df = df_norm.reset_index().rename(columns={df_norm.index.name or "index": "time"})
            source_raw = df_norm.attrs.get("source", df_raw.attrs.get("source", "unknown"))
            source = source_raw
            if isinstance(source_raw, str) and source_raw.lower().startswith("fallback"):
                source = "fallback"
            last_update_str = ""
            if not df_norm.empty:
                try:
                    last_update_str = df_norm.index[-1].strftime("%d/%m/%Y %H:%M")
                except Exception:
                    last_update_str = str(df_norm.index[-1])
            tf_label = format_timeframe_label(interval)
            title_parts = [symbol, tf_label]
            if source and source != "unknown":
                title_parts.append(f"Fonte: {source}")
            if last_update_str:
                title_parts.append(f"Atualizado: {last_update_str}")
            fig_title = " • ".join([p for p in title_parts if p])
            fig = chart_generator.create_plotly_chart(df_norm, fig_title, timeframe=interval)
            candles_payload, series = _build_chart_components(payload_df, limit=limit)

            emit(
                "chart_data",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "source": source,
                    "candles": candles_payload,
                    "series": series,
                    "indicators": {
                        "sma_9": float(df_norm["SMA_9"].iloc[-1]) if "SMA_9" in df_norm.columns else None,
                        "sma_21": float(df_norm["SMA_21"].iloc[-1]) if "SMA_21" in df_norm.columns else None,
                        "ema_21": float(df_norm["EMA_21"].iloc[-1]) if "EMA_21" in df_norm.columns else None,
                        "ema_200": float(df_norm["EMA_200"].iloc[-1]) if "EMA_200" in df_norm.columns else None,
                        "rsi": float(df_norm["RSI"].iloc[-1]) if "RSI" in df_norm.columns else None,
                    },
                    "chart": json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                },
            )

        except Exception as exc:
            logger.error(f"Erro ao processar request_chart: {exc}")
            emit("chart_error", {"error": str(exc)})
