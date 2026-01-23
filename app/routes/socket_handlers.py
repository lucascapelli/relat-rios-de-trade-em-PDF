"""Socket.IO event handlers."""
from __future__ import annotations

import json

import plotly
from flask_socketio import SocketIO, emit, join_room, leave_room

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

			if not symbol:
				emit("chart_error", {"error": "Símbolo não informado"})
				return

			df = finance_data.get_candles(symbol, interval, 100)
			fig = chart_generator.create_plotly_chart(df, f"{symbol} - {interval}")
			candles_payload, series = _build_chart_components(df, limit=100)
			source = df.attrs.get("source", "unknown")

			emit(
				"chart_data",
				{
					"symbol": symbol,
					"interval": interval,
					"source": source,
					"candles": candles_payload,
					"series": series,
					"indicators": {
						"sma_9": float(df["SMA_9"].iloc[-1]) if "SMA_9" in df.columns else None,
						"sma_21": float(df["SMA_21"].iloc[-1]) if "SMA_21" in df.columns else None,
						"rsi": float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None,
					},
					"chart": json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
				},
			)

		except Exception as exc:
			logger.error(f"Erro ao processar request_chart: {exc}")
			emit("chart_error", {"error": str(exc)})
