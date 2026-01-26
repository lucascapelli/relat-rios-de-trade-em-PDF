"""HTTP routes for the reporting system."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly
from flask import Blueprint, jsonify, render_template, request, send_from_directory

from ..charts import ChartGenerator, format_timeframe_label
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
        "ema_21": "EMA_21",
        "ema_200": "EMA_200",
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

        # Emit time in a JS-friendly ISO format (seconds precision).
        # Some browsers/Plotly paths can behave badly with 6-digit microseconds.
        try:
            time_ts = pd.Timestamp(row["time"]).to_pydatetime().replace(microsecond=0)
            iso_time = time_ts.isoformat()
        except Exception:
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

    def _parse_iso_date(value: Optional[str]):
        from datetime import datetime

        if not value:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except Exception:
            return None

    def _get_period_range(period_raw: str, reference_date_raw: Optional[str]):
        from datetime import date, timedelta
        import calendar

        period_norm = (period_raw or "").strip().lower()
        if period_norm in {"semanal", "semana", "weekly", "week", "w"}:
            period_norm = "weekly"
        elif period_norm in {"mensal", "mes", "mês", "monthly", "month", "m"}:
            period_norm = "monthly"
        elif period_norm in {"geral", "all", "range", "custom"}:
            period_norm = "range"

        ref = _parse_iso_date(reference_date_raw) or date.today()

        if period_norm == "weekly":
            start = ref - timedelta(days=ref.weekday())
            end = start + timedelta(days=6)
            return period_norm, start, end

        if period_norm == "monthly":
            start = ref.replace(day=1)
            last_day = calendar.monthrange(ref.year, ref.month)[1]
            end = ref.replace(day=last_day)
            return period_norm, start, end

        return "range", ref, ref

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
            fig = chart_generator.create_plotly_chart(df, f"PETR4 • {format_timeframe_label('15m')}", timeframe="15m")
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
            limit_raw = request.args.get("limit", type=int, default=100)
            limit = max(50, min(limit_raw or 100, 1500))

            df_raw = finance_data.get_candles(symbol, interval, limit)
            df_norm = ChartGenerator.prepare_ohlc_dataframe(df_raw, interval)
            df_norm = ChartGenerator._ensure_overlay_columns(df_norm)
            payload_df = df_norm.reset_index().rename(columns={df_norm.index.name or "index": "time"})
            date_format = ChartGenerator._get_date_format(interval)
            if "time" in payload_df.columns:
                try:
                    payload_df["time_str"] = payload_df["time"].dt.strftime(date_format)
                except Exception:
                    payload_df["time_str"] = payload_df["time"].astype(str)

            source_raw = df_norm.attrs.get("source", df_raw.attrs.get("source", "unknown"))
            source = source_raw
            if isinstance(source_raw, str) and source_raw.lower().startswith("fallback"):
                source = "fallback"
            last_update_str = ""
            if not df_norm.empty:
                try:
                    last_update_str = pd.Timestamp(df_norm.index[-1]).strftime("%d/%m/%Y %H:%M")
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
                "sma_9": float(df_norm["SMA_9"].iloc[-1]) if "SMA_9" in df_norm.columns else None,
                "sma_21": float(df_norm["SMA_21"].iloc[-1]) if "SMA_21" in df_norm.columns else None,
                "ema_21": float(df_norm["EMA_21"].iloc[-1]) if "EMA_21" in df_norm.columns else None,
                "ema_200": float(df_norm["EMA_200"].iloc[-1]) if "EMA_200" in df_norm.columns else None,
                "rsi": float(df_norm["RSI"].iloc[-1]) if "RSI" in df_norm.columns else None,
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

            def _normalize_tf(value: Any) -> str:
                raw = (str(value or "").strip().lower())
                mapping = {
                    "60m": "1h",
                    "60min": "1h",
                    "1hour": "1h",
                    "1d": "1d",
                    "daily": "1d",
                    "diario": "1d",
                    "1w": "1w",
                    "weekly": "1w",
                    "semanal": "1w",
                    "15": "15m",
                    "15min": "15m",
                }
                return mapping.get(raw, raw)

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

            timeframe = _normalize_tf(_get_first(["timeframe", "timeframe_base"]) or "15m")

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
            # PDFs: manter apenas visão tática (15m), confirmação (1h) e contexto (1d)
            base_timeframes = ["15m", "1h", "1d"]
            if operation.timeframe and operation.timeframe not in base_timeframes:
                preferred_timeframes.append(operation.timeframe)
            for candidate in base_timeframes:
                if candidate not in preferred_timeframes:
                    preferred_timeframes.append(candidate)

            chart_images: List[Dict[str, Any]] = []
            indicator_df: Optional[pd.DataFrame] = None

            def _fmt_source(value: str) -> str:
                raw = (value or "").strip().lower()
                if not raw:
                    return "desconhecido"
                if raw.startswith("fallback"):
                    return "fallback"
                return raw

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
                    f"{operation.symbol} • {format_timeframe_label(tf)}",
                    timeframe=tf,
                    operation=operation,
                )

                last_row = df.iloc[-1] if len(df.index) > 0 else None
                chart_images.append(
                    {
                        "timeframe": tf,
                        "image": img if img else PLACEHOLDER_IMAGE_DATA_URL,
                        "source": _fmt_source(df.attrs.get("source", "desconhecido")),
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
                    timeframe=operation.timeframe,
                    operation=operation,
                )
                fallback_row = fallback_df.iloc[-1] if len(fallback_df.index) > 0 else None
                chart_images.append(
                    {
                        "timeframe": operation.timeframe,
                        "image": fallback_img if fallback_img else PLACEHOLDER_IMAGE_DATA_URL,
                        "source": _fmt_source(fallback_df.attrs.get("source", "fallback")),
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

            # Inclui swing trades no histórico (novo fluxo usa swing/day trade e não preenche mais `operations`).
            swing_trades = database.get_swing_trades(limit=100)
            swing_ops: List[Dict[str, Any]] = []

            for swing in swing_trades:
                try:
                    direction = str(swing.get("direction") or "LONG").upper()
                    tipo = "COMPRA" if direction == "LONG" else "VENDA"

                    op_model = Operation(
                        symbol=str(swing.get("symbol") or "").upper(),
                        tipo=tipo,
                        entrada=float(swing.get("entry") or 0),
                        stop=float(swing.get("stop") or 0),
                        alvo=float(swing.get("target") or 0),
                        quantidade=int(swing.get("quantity") or 0),
                        timeframe=str(swing.get("timeframe_major") or "1d"),
                        entrada_min=swing.get("entry_min"),
                        entrada_max=swing.get("entry_max"),
                        observacoes=str(swing.get("analytical_text") or ""),
                    )

                    op_dict = asdict(op_model)
                    op_dict.update(
                        {
                            "id": f"swing-{swing.get('id')}",
                            "pdf_path": swing.get("pdf_path"),
                            "created_at": swing.get("created_at") or swing.get("trade_date"),
                            "status": swing.get("status") or "ABERTA",
                            "source": "swing_trade",
                        }
                    )
                    swing_ops.append(op_dict)
                except Exception as exc:
                    logger.warning("Falha ao mapear swing trade para histórico: %s", exc, exc_info=True)

            # Marca origem para diferenciar e faz merge.
            for op in operations:
                op.setdefault("source", "operation")

            merged = operations + swing_ops

            def _sort_key(item: Dict[str, Any]):
                ts = pd.to_datetime(item.get("created_at"), errors="coerce")
                if pd.isna(ts):
                    return pd.Timestamp.min
                return ts

            merged.sort(key=_sort_key, reverse=True)

            return jsonify(merged)
        except Exception as exc:
            logger.error(f"Erro ao buscar histórico: {exc}")
            return jsonify([])

    @bp.route("/reports/<path:filename>")
    def serve_report(filename: str):
        return send_from_directory(str(REPORTS_DIR), filename)

    # ========== SWING TRADE ROUTES ==========
    @bp.route("/api/swing-trade", methods=["POST"])
    def api_swing_trade():
        try:
            data = request.get_json() or {}
            from datetime import datetime
            from ..config import BR_TZ

            swing_data = {
                "symbol": data.get("symbol", "").upper(),
                "direction": data.get("direction", "LONG").upper(),
                "entry": float(data.get("entry", 0)),
                "entry_min": float(data.get("entry_min", data.get("entry", 0))),
                "entry_max": float(data.get("entry_max", data.get("entry", 0))),
                "target": float(data.get("target", 0)),
                "stop": float(data.get("stop", 0)),
                "quantity": int(data.get("quantity", 1)),
                "trade_date": data.get("trade_date", datetime.now(BR_TZ).strftime("%Y-%m-%d")),
                "timeframe_major": data.get("timeframe_major", "1d"),
                "timeframe_minor": data.get("timeframe_minor", "1h"),
                "risk_amount": data.get("risk_amount"),
                "risk_percent": data.get("risk_percent"),
                "target_percent": data.get("target_percent"),
                "stop_percent": data.get("stop_percent"),
                "analytical_text": data.get("analytical_text", ""),
                "client_name": data.get("client_name"),
                "status": "ABERTA",
                "created_at": datetime.now(BR_TZ).isoformat(),
            }

            trade_id = database.insert_swing_trade(swing_data)

            return jsonify({"id": trade_id, "status": "success", "message": "Swing trade registrado com sucesso"})
        except Exception as exc:
            logger.error(f"Erro ao registrar swing trade: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/swing-trade/list", methods=["GET"])
    def api_swing_trade_list():
        try:
            trades = database.get_swing_trades(limit=100)
            return jsonify(trades)
        except Exception as exc:
            logger.error(f"Erro ao listar swing trades: {exc}")
            return jsonify([])

    # ========== DAY TRADE ROUTES ==========
    @bp.route("/api/day-trade", methods=["POST"])
    def api_day_trade():
        try:
            data = request.get_json() or {}
            from datetime import datetime
            from ..config import BR_TZ

            session_data = {
                "trade_date": data.get("trade_date", datetime.now(BR_TZ).strftime("%Y-%m-%d")),
                "timeframe_major": data.get("timeframe_major", "1h"),
                "timeframe_minor": data.get("timeframe_minor", "15m"),
                "risk_amount": data.get("risk_amount"),
                "risk_percent": data.get("risk_percent"),
                "created_at": datetime.now(BR_TZ).isoformat(),
            }

            entries = data.get("entries", [])
            session_id = database.insert_day_trade_session(session_data, entries)
            return jsonify({"id": session_id, "status": "success", "message": "Day trade registrado com sucesso"})
        except Exception as exc:
            logger.error(f"Erro ao registrar day trade: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/day-trade/list", methods=["GET"])
    def api_day_trade_list():
        try:
            sessions = database.get_day_trade_sessions(limit=100)
            return jsonify(sessions)
        except Exception as exc:
            logger.error(f"Erro ao listar day trades: {exc}")
            return jsonify([])

    # ========== PORTFOLIO ROUTES ==========
    @bp.route("/api/portfolio", methods=["POST"])
    def api_portfolio():
        try:
            data = request.get_json() or {}
            from datetime import datetime
            from ..config import BR_TZ
            
            portfolio_data = {
                "portfolio_type": data.get("portfolio_type", "GERAL"),
                "state": data.get("state", "CONSTRUIR"),
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
                "analytical_text": data.get("analytical_text", ""),
                "version": data.get("version", 1),
                "created_at": datetime.now(BR_TZ).isoformat()
            }
            
            assets = data.get("assets", [])
            
            portfolio_id = database.insert_portfolio(portfolio_data, assets)
            
            return jsonify({
                "id": portfolio_id,
                "status": "success",
                "message": "Carteira registrada com sucesso"
            })
        except Exception as exc:
            logger.error(f"Erro ao registrar carteira: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/portfolio/list", methods=["GET"])
    def api_portfolio_list():
        try:
            portfolios = database.get_portfolios(limit=100)
            return jsonify(portfolios)
        except Exception as exc:
            logger.error(f"Erro ao listar carteiras: {exc}")
            return jsonify([])

    @bp.route("/api/portfolio/manipulated", methods=["GET"])
    def api_portfolio_manipulated_assets():
        """List assets manipulated in Day Trade and Swing Trade for a period."""
        try:
            period = request.args.get("period", default="weekly", type=str)
            reference_date = request.args.get("reference_date", default=None, type=str)
            include_swing = request.args.get("include_swing", default="1", type=str) != "0"
            include_daytrade = request.args.get("include_daytrade", default="1", type=str) != "0"

            period_norm, start_d, end_d = _get_period_range(period, reference_date)
            start_date = start_d.strftime("%Y-%m-%d")
            end_date = end_d.strftime("%Y-%m-%d")

            assets = database.get_manipulated_assets(
                start_date=start_date,
                end_date=end_date,
                include_swing=include_swing,
                include_daytrade=include_daytrade,
            )

            return jsonify(
                {
                    "period": period_norm,
                    "reference_date": reference_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "assets": assets,
                }
            )
        except Exception as exc:
            logger.error(f"Erro ao listar ativos manipulados: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/portfolio/manipulated/pdf", methods=["POST"])
    def api_portfolio_manipulated_pdf():
        """Generate a portfolio PDF based on manipulated assets (no DB persistence)."""
        try:
            data = request.get_json() or {}
            period = str(data.get("period") or "weekly")
            reference_date = data.get("reference_date")
            include_swing = bool(data.get("include_swing", True))
            include_daytrade = bool(data.get("include_daytrade", True))
            analytical_text = str(data.get("analytical_text") or "")

            period_norm, start_d, end_d = _get_period_range(period, reference_date)
            start_date = start_d.strftime("%Y-%m-%d")
            end_date = end_d.strftime("%Y-%m-%d")

            assets = database.get_manipulated_assets(
                start_date=start_date,
                end_date=end_date,
                include_swing=include_swing,
                include_daytrade=include_daytrade,
            )

            portfolio_payload = {
                "portfolio_type": "SEMANAL" if period_norm == "weekly" else "MENSAL" if period_norm == "monthly" else "GERAL",
                "state": "DERIVADA",
                "start_date": start_date,
                "end_date": end_date,
                "version": 1,
                "analytical_text": analytical_text,
                "assets": [
                    {
                        "symbol": a.get("symbol"),
                        "entry": a.get("entry"),
                        "entry_max": a.get("entry_max"),
                        "risk_zero": a.get("risk_zero"),
                        "target": a.get("target"),
                        "stop": a.get("stop"),
                    }
                    for a in assets
                ],
            }

            pdf_path = report_generator.generate_portfolio_pdf(portfolio_payload)

            return jsonify(
                {
                    "status": "success",
                    "filename": os.path.basename(pdf_path),
                    "url": f"/reports/{os.path.basename(pdf_path)}",
                    "start_date": start_date,
                    "end_date": end_date,
                    "count": len(assets),
                }
            )
        except Exception as exc:
            logger.error(f"Erro ao gerar PDF de carteira derivada: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/swing-trade/<int:trade_id>/pdf", methods=["GET"])
    def api_swing_trade_pdf(trade_id):
        """Generate PDF for a specific swing trade"""
        try:
            trade = database.get_swing_trade(int(trade_id))
            if not trade:
                return jsonify({"error": "Trade não encontrado"}), 404

            from ..models import Operation

            direction = str(trade.get("direction") or "LONG").upper()
            tipo = "COMPRA" if direction == "LONG" else "VENDA"
            op = Operation(
                symbol=str(trade.get("symbol") or "").upper(),
                tipo=tipo,
                entrada=float(trade.get("entry") or 0),
                stop=float(trade.get("stop") or 0),
                alvo=float(trade.get("target") or 0),
                quantidade=int(trade.get("quantity") or 1),
                timeframe=str(trade.get("timeframe_major") or "1d"),
                entrada_min=float(trade.get("entry_min") or trade.get("entry") or 0),
                entrada_max=float(trade.get("entry_max") or trade.get("entry") or 0),
                observacoes=str(trade.get("analytical_text") or ""),
            )

            # Try to enrich with current price (non-fatal)
            try:
                info = finance_data.get_ticker_info(op.symbol)
                if info and getattr(info, "price", None):
                    op.preco_atual = float(info.price)
            except Exception:
                pass

            def _normalize_tf(value: Any) -> str:
                raw = (str(value or "").strip().lower())
                mapping = {
                    "60m": "1h",
                    "60min": "1h",
                    "1hour": "1h",
                    "daily": "1d",
                    "diario": "1d",
                    "weekly": "1w",
                    "semanal": "1w",
                }
                return mapping.get(raw, raw)

            # Para swing trade, os timeframes principais são major (principal) e minor (entrada)
            # A ordem correta é: major -> minor -> outros complementares
            base_tfs = ["1d", "1h", "15m"]
            major = _normalize_tf(trade.get("timeframe_major"))
            minor = _normalize_tf(trade.get("timeframe_minor"))
            
            tfs: List[str] = []
            # Primeiro adiciona o major (timeframe principal)
            if major and major not in tfs:
                tfs.append(major)
            # Depois o minor (timeframe de entrada)
            if minor and minor not in tfs:
                tfs.append(minor)
            # Por fim, os complementares
            for tf in base_tfs:
                if tf and tf not in tfs:
                    tfs.append(tf)

            chart_images: List[Dict[str, Any]] = []
            for tf in tfs:
                try:
                    df_raw = finance_data.get_candles(op.symbol, tf, 150)
                    df_norm = ChartGenerator.prepare_ohlc_dataframe(df_raw, tf)
                    df_norm = ChartGenerator._ensure_overlay_columns(df_norm)
                    title = f"{op.symbol} • {format_timeframe_label(tf)}"
                    image_b64 = chart_generator.generate_chart_image(df_norm, title, timeframe=tf, operation=op)

                    source_raw = df_norm.attrs.get("source", df_raw.attrs.get("source", "unknown"))
                    source = source_raw
                    if isinstance(source_raw, str) and source_raw.lower().startswith("fallback"):
                        source = "fallback"

                    last_time = None
                    last_close = None
                    if not df_norm.empty:
                        try:
                            last_time = pd.Timestamp(df_norm.index[-1]).strftime("%d/%m/%Y %H:%M")
                        except Exception:
                            last_time = str(df_norm.index[-1])
                        try:
                            last_close = float(df_norm["close"].iloc[-1]) if "close" in df_norm.columns else None
                        except Exception:
                            last_close = None

                    chart_images.append(
                        {
                            "timeframe": tf,
                            "image": image_b64,
                            "source": source,
                            "last_time": last_time,
                            "last_close": last_close,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Falha ao gerar chart para swing trade {op.symbol} {tf}: {exc}")

            pdf_path = report_generator.generate_pdf_report(op, chart_images)
            database.update_swing_trade_pdf(int(trade_id), pdf_path)

            return jsonify(
                {
                    "status": "success",
                    "filename": os.path.basename(pdf_path),
                    "url": f"/reports/{os.path.basename(pdf_path)}",
                }
            )
            
        except Exception as exc:
            logger.error(f"Erro ao gerar PDF de swing trade: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/day-trade/<int:session_id>/pdf", methods=["GET"])
    def api_day_trade_pdf(session_id):
        """Generate PDF for a specific day trade session with charts per símbolo/timeframe."""
        try:
            session = database.get_day_trade_session(int(session_id))

            if not session:
                return jsonify({"error": "Sessão não encontrada"}), 404

            entries = session.get("entries") or []
            symbols = sorted({str(e.get("symbol") or "").upper() for e in entries if e.get("symbol")})

            charts_payload: List[Dict[str, Any]] = []
            preferred_timeframes = ["15m", "1h", "1d"]

            for symbol in symbols:
                for tf in preferred_timeframes:
                    try:
                        df_raw = finance_data.get_candles(symbol, tf, 150)
                        df_norm = ChartGenerator.prepare_ohlc_dataframe(df_raw, tf)
                        df_norm = ChartGenerator._ensure_overlay_columns(df_norm)
                        title = f"{symbol} • {format_timeframe_label(tf)}"
                        image_b64 = chart_generator.generate_chart_image(df_norm, title, timeframe=tf)

                        source_raw = df_norm.attrs.get("source", df_raw.attrs.get("source", "unknown"))
                        source = source_raw
                        if isinstance(source_raw, str) and source_raw.lower().startswith("fallback"):
                            source = "fallback"

                        last_time = None
                        last_close = None
                        if not df_norm.empty:
                            try:
                                last_time = pd.Timestamp(df_norm.index[-1]).strftime("%d/%m/%Y %H:%M")
                            except Exception:
                                last_time = str(df_norm.index[-1])
                            try:
                                last_close = float(df_norm["close"].iloc[-1]) if "close" in df_norm.columns else None
                            except Exception:
                                last_close = None

                        charts_payload.append(
                            {
                                "symbol": symbol,
                                "timeframe": tf,
                                "image": image_b64,
                                "source": source,
                                "last_time": last_time,
                                "last_close": last_close,
                            }
                        )
                    except Exception as exc_inner:
                        logger.warning(
                            "Falha ao gerar chart para day trade %s %s: %s",
                            symbol,
                            tf,
                            exc_inner,
                        )

            pdf_path = report_generator.generate_day_trade_pdf(session, charts_payload)
            database.update_day_trade_session_pdf(int(session_id), pdf_path)

            return jsonify(
                {
                    "status": "success",
                    "filename": os.path.basename(pdf_path),
                    "url": f"/reports/{os.path.basename(pdf_path)}",
                }
            )

        except Exception as exc:
            logger.error(f"Erro ao gerar PDF de day trade: {exc}")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/portfolio/<int:portfolio_id>/pdf", methods=["GET"])
    def api_portfolio_pdf(portfolio_id):
        """Generate PDF for a specific portfolio"""
        try:
            portfolio = database.get_portfolio(int(portfolio_id))
            
            if not portfolio:
                return jsonify({"error": "Carteira não encontrada"}), 404

            pdf_path = report_generator.generate_portfolio_pdf(portfolio)
            database.update_portfolio_pdf(int(portfolio_id), pdf_path)

            return jsonify(
                {
                    "status": "success",
                    "filename": os.path.basename(pdf_path),
                    "url": f"/reports/{os.path.basename(pdf_path)}",
                }
            )
            
        except Exception as exc:
            logger.error(f"Erro ao gerar PDF de carteira: {exc}")
            return jsonify({"error": str(exc)}), 500

    app.register_blueprint(bp)

