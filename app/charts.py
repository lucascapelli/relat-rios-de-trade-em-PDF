from __future__ import annotations

import base64
import io
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import dates as mdates
from matplotlib.lines import Line2D

from .models import Operation
from .utils import logger


HOUR_START = "10:00"
HOUR_END = "17:00"
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 420

DATE_FORMAT_BY_TF: Dict[str, str] = {
    "1m": "%d/%m %H:%M",
    "5m": "%d/%m %H:%M",
    "6m": "%d/%m %H:%M",
    "15m": "%d/%m %H:%M",
    "30m": "%d/%m %H:%M",
    "45m": "%d/%m %H:%M",
    "1h": "%d/%m %H:%M",
    "4h": "%d/%m %H:%M",
    "1d": "%d/%m/%Y",
    "1w": "%d/%m/%Y",
}

HEIGHT_BY_TF: Dict[str, int] = {
    "1m": 440,
    "5m": 430,
    "6m": 430,
    "15m": 450,
    "30m": 420,
    "45m": 400,
    "1h": 400,
    "4h": 380,
    "60m": 400,
    "1d": 400,
    "1w": 360,
}

HOLIDAYS: Sequence[str] = (
    "2024-01-01",
    "2024-02-12",
    "2024-02-13",
    "2024-03-29",
    "2024-04-21",
    "2024-05-01",
    "2024-09-07",
    "2024-10-12",
    "2024-11-02",
    "2024-11-15",
    "2024-12-25",
    "2025-01-01",
    "2025-02-03",
    "2025-02-04",
    "2025-04-18",
    "2025-05-01",
    "2025-09-07",
    "2025-10-12",
    "2025-11-02",
    "2025-11-15",
    "2025-12-25",
    "2026-01-01",
)

HOLIDAY_INDEX = pd.to_datetime(HOLIDAYS, utc=False).normalize()
INTRADAY_SET = {"1m", "5m", "6m", "15m", "30m", "45m", "1h", "4h"}

TIMEFRAME_LABELS: Dict[str, str] = {
    "1m": "1 min",
    "5m": "5 min",
    "6m": "6 min",
    "15m": "15 min",
    "30m": "30 min",
    "45m": "45 min",
    "1h": "60 min",
    "4h": "4h",
    "1d": "Diário",
    "1w": "Semanal",
}

MAX_BARS_BY_TF: Dict[str, int] = {
    "1m": 280,
    "5m": 260,
    "6m": 240,
    "15m": 220,
    "30m": 200,
    "45m": 180,
    "1h": 180,
    "4h": 180,
    "1d": 90,
    "1w": 120,
}


def format_timeframe_label(timeframe: Optional[str]) -> str:
    tf = (timeframe or "").strip().lower()
    return TIMEFRAME_LABELS.get(tf, tf.upper() if tf else "")


def _tail_for_timeframe(df: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tf = (timeframe or "").strip().lower()
    limit = MAX_BARS_BY_TF.get(tf)
    if not limit:
        return df
    return df.tail(limit)

OVERLAY_STYLE: Sequence[Tuple[str, str, str, str, float]] = (
    ("SMA_9", "SMA 9", "#8E24AA", "solid", 1.2),
    ("SMA_21", "SMA 21", "#FB8C00", "solid", 1.2),
    ("EMA_21", "MME 21", "#1E88E5", "dash", 1.4),
    ("EMA_200", "MME 200", "#D32F2F", "dash", 1.6),
)


def _debug_time_data(df: Optional[pd.DataFrame], name: str = "Dataframe") -> None:
    """Emit logs sobre intervalo temporal para facilitar diagnósticos."""

    if df is None or df.empty:
        logger.warning("%s: Vazio ou None", name)
        return

    try:
        first = df.index[0]
        last = df.index[-1]
        span = last - first
        logger.info("%s: %s registros", name, len(df))
        logger.info("%s: Primeiro índice=%s", name, first)
        logger.info("%s: Último índice=%s", name, last)
        logger.info("%s: Intervalo total=%s", name, span)

        if hasattr(first, "year"):
            years = getattr(df.index, "year", None)
            if years is not None:
                distinct_years = sorted(set(int(y) for y in years))
                if len(distinct_years) > 3:
                    logger.warning("%s: Muitos anos distintos detectados: %s", name, distinct_years)
    except Exception:
        logger.debug("Falha ao inspecionar dados temporais para %s", name, exc_info=True)


def _mpl_locator_formatter(tf_key: str) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    """Return locator/formatter tuned for PDF exports without losing readability."""

    tf = (tf_key or "").lower()

    if tf in {"1m", "5m", "6m"}:
        locator = mdates.HourLocator(interval=2)
        formatter = mdates.DateFormatter("%d/%m %Hh")
        return locator, formatter
    
    if tf in {"15m"}:
        locator = mdates.HourLocator(interval=1)
        formatter = mdates.DateFormatter("%d/%m %Hh")
        return locator, formatter
    
    if tf in {"30m", "45m"}:
        locator = mdates.HourLocator(interval=6)
        formatter = mdates.DateFormatter("%d/%m %Hh")
        return locator, formatter
    
    if tf in {"1h", "60m"}:
        locator = mdates.HourLocator(interval=6)
        formatter = mdates.DateFormatter("%d/%m %Hh")
        return locator, formatter
    
    if tf in {"4h"}:
        locator = mdates.DayLocator(interval=2)
        formatter = mdates.DateFormatter("%d/%m")
        return locator, formatter
    
    if tf in {"1d"}:
        locator = mdates.DayLocator(interval=5)
        formatter = mdates.DateFormatter("%d/%m")
        return locator, formatter
    
    if tf in {"1w"}:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=4)
        formatter = mdates.DateFormatter("%d/%m")
        return locator, formatter

    auto_locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    return auto_locator, mdates.ConciseDateFormatter(auto_locator)


_TIME_WINDOW_BY_TF: Dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(hours=4),
    "5m": pd.Timedelta(hours=8),
    "6m": pd.Timedelta(hours=12),
    "15m": pd.Timedelta(hours=4),
    "30m": pd.Timedelta(days=2),
    "45m": pd.Timedelta(days=3),
    "1h": pd.Timedelta(days=1),
    "60m": pd.Timedelta(days=1),
    "4h": pd.Timedelta(days=7),
    "1d": pd.Timedelta(days=30),
}

_MIN_CANDLES_FOR_WINDOW: Dict[str, int] = {
    "15m": 16,
    "30m": 20,
    "45m": 15,
    "1h": 12,
    "60m": 12,
    "4h": 15,
    "1d": 20,
}


def prepare_ohlc_dataframe(
    df: pd.DataFrame,
    timeframe: Optional[str] = None,
    remove_nontrading: bool = True,
    market_hours: bool = True,
    reindex: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty OHLC dataframe")

    frame = df.copy()

    index_names: List[Optional[str]]
    if isinstance(frame.index, pd.MultiIndex):
        index_names = list(frame.index.names)
    else:
        index_names = [frame.index.name]

    has_time_index = any(name == "time" for name in index_names if name is not None)
    if has_time_index:
        if isinstance(frame.index, pd.MultiIndex):
            renamed_levels = [f"{name}_index" if name == "time" else name for name in index_names]
            frame.index = frame.index.rename(renamed_levels)
        else:
            frame.index = frame.index.rename("time_index")

    if "time" in frame.columns:
        time_values = frame["time"]
    else:
        # Se o index já for datetime, use-o diretamente; se for RangeIndex, não invente epoch (1970).
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            frame["time"] = frame.index
            time_values = frame["time"]
        else:
            index_name = frame.index.name or "index"
            frame = frame.reset_index().rename(columns={index_name: "time"})
            time_values = frame["time"]

    if pd.api.types.is_integer_dtype(time_values):
        numeric = pd.to_numeric(time_values, errors="coerce")
        parsed = numeric.dropna()

        def _looks_like_row_index(values: pd.Series) -> bool:
            try:
                if values.empty:
                    return False
                # Heurística: sequência curta/pequena (0..n) com passo 1 -> não é timestamp.
                v = values.astype("int64")
                if v.min() not in (0, 1):
                    return False
                if v.max() - v.min() != len(v.unique()) - 1:
                    return False
                if len(v) < 3:
                    return True
                diffs = np.diff(np.sort(v.unique()))
                return bool(np.all(diffs == 1))
            except Exception:
                return False

        if parsed.empty:
            frame["time"] = pd.to_datetime(time_values, errors="coerce")
        elif _looks_like_row_index(parsed):
            # Evita converter 0..N em epoch (1970). Tenta outras fontes de tempo antes de desistir.
            if "time_str" in frame.columns:
                frame["time"] = pd.to_datetime(frame["time_str"], errors="coerce")
            elif isinstance(df.index, pd.DatetimeIndex):
                frame["time"] = df.index
            else:
                raise ValueError("Missing datetime information for chart (time column looks like row index)")
        else:
            median_value = float(parsed.median())
            abs_value = abs(median_value)
            # Detecta unidade por ordem de grandeza.
            if abs_value >= 1e17:
                unit = "ns"
            elif abs_value >= 1e14:
                unit = "us"
            elif abs_value >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            frame["time"] = pd.to_datetime(numeric, unit=unit, origin="unix", errors="coerce")
    else:
        frame["time"] = pd.to_datetime(time_values, errors="coerce")

    frame = frame.dropna(subset=["time"]).copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    if getattr(frame.index, "tz", None) is not None:
        try:
            frame.index = frame.index.tz_convert("America/Sao_Paulo").tz_localize(None)
        except Exception:
            frame.index = frame.index.tz_localize(None)

    tf_key = (timeframe or "").lower()

    if market_hours and tf_key in INTRADAY_SET:
        try:
            frame = frame.between_time(HOUR_START, HOUR_END)
        except Exception:
            logger.warning("Falha ao filtrar horário de pregão", exc_info=True)

    if remove_nontrading:
        frame = frame[frame.index.dayofweek < 5]
        frame = frame[~frame.index.normalize().isin(HOLIDAY_INDEX)]

    if reindex and not frame.empty:
        freq_map = {
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "45m": "45T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W-MON",
        }
        freq = freq_map.get(tf_key)
        if freq:
            full_index = pd.date_range(frame.index[0], frame.index[-1], freq=freq)
            frame = frame.reindex(full_index)
            frame = frame.dropna(subset=["open", "close"], how="all")

    return frame


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    close = enriched["close"].astype(float)

    enriched["SMA_9"] = close.rolling(window=9, min_periods=1).mean()
    enriched["SMA_21"] = close.rolling(window=21, min_periods=1).mean()
    enriched["EMA_21"] = close.ewm(span=21, adjust=False, min_periods=1).mean()
    enriched["EMA_200"] = close.ewm(span=200, adjust=False, min_periods=1).mean()

    return enriched


def render_price_chart_base64(
    df: pd.DataFrame,
    title: str = "",
    timeframe: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    operation: Optional[Operation] = None,
) -> str:
    axis_limits: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    try:
        tf_key = (timeframe or "").lower()
        
        normalized = prepare_ohlc_dataframe(df, timeframe)
        logger.info(f"Dados normalizados: {len(normalized)} candles para {tf_key}")
        _debug_time_data(normalized, "Dados Normalizados")
        
        normalized = _tail_for_timeframe(normalized, timeframe)
        logger.info(f"Após tail: {len(normalized)} candles")
        
        enriched = _add_indicators(normalized)
        logger.info(f"Após indicadores: {len(enriched)} candles")
        _debug_time_data(enriched, "Dados após tail e indicadores")
        if not enriched.empty:
            logger.info("Intervalo temporal: %s até %s", enriched.index[0], enriched.index[-1])
            time_span = enriched.index[-1] - enriched.index[0]
            logger.info("Span temporal total: %s", time_span)
            if getattr(time_span, "days", 0) > 365:
                logger.error("Span temporal muito grande: %s dias", time_span.days)
                max_candles = MAX_BARS_BY_TF.get(tf_key, 200)
                enriched = enriched.tail(max_candles)
                logger.info("Forçado tail para %s candles", len(enriched))
                _debug_time_data(enriched, "Dados após correção de span")

        window = _TIME_WINDOW_BY_TF.get(tf_key)
        
        if window is not None and not enriched.empty and len(enriched) > 5:
            try:
                end_ts = pd.Timestamp(enriched.index[-1])
                start_ts = end_ts - window
                if start_ts < enriched.index[0]:
                    start_ts = enriched.index[0]
                window_df = enriched.loc[enriched.index >= start_ts]
                min_required = _MIN_CANDLES_FOR_WINDOW.get(tf_key, 15)

                logger.info("Janela calculada: %s a %s", start_ts, end_ts)
                logger.info("Candles na janela: %s", len(window_df))

                if len(window_df) >= min_required:
                    enriched = window_df
                    axis_limits = (window_df.index[0], window_df.index[-1])
                    logger.info("Usando janela temporal de %s candles", len(window_df))
                else:
                    if tf_key in INTRADAY_SET:
                        fallback_count = max(min_required * 2, 30)
                    else:
                        fallback_count = max(min_required, 20)

                    if len(enriched) >= fallback_count:
                        enriched = enriched.tail(fallback_count)
                    axis_limits = None
                    logger.info("Usando fallback: %s candles", len(enriched))
            except Exception as e:
                logger.warning(f"Erro ao aplicar janela temporal para {tf_key}: {e}")
                # Em caso de erro, use todo o dataset disponível
                max_candles = MAX_BARS_BY_TF.get(tf_key, 200)
                enriched = enriched.tail(max_candles)
                axis_limits = None

        if not enriched.empty:
            final_span = enriched.index[-1] - enriched.index[0]
            total_hours = final_span.total_seconds() / 3600 if hasattr(final_span, "total_seconds") else 0.0
            logger.info(
                "Span temporal final: %s dias e %.2f horas",
                final_span.days,
                total_hours,
            )

        # Cálculo dinâmico de max_ticks baseado no número de candles
        num_candles = len(enriched)
        if num_candles > 100:
            dynamic_max_ticks = max(5, int(num_candles / 20))
        elif num_candles > 50:
            dynamic_max_ticks = 8
        else:
            dynamic_max_ticks = 12

        plot_df = enriched.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )

        addplots: List = []
        legend_entries: List[Tuple[str, str, str, float]] = []
        for column, label, color, linestyle, width_line in OVERLAY_STYLE:
            if column in plot_df and plot_df[column].notna().any():
                series = plot_df[column].dropna()
                if len(series.index) >= 2:
                    addplots.append(
                        mpf.make_addplot(
                            series,
                            color=color,
                            linestyle=linestyle.replace("dash", "--"),
                            width=width_line,
                        )
                    )
                    legend_entries.append((label, color, linestyle.replace("dash", "--"), width_line))

        market_colors = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="yahoo",
            marketcolors=market_colors,
            facecolor="white",
            figcolor="white",
        )

        show_nontrading = tf_key in {"1d", "1w"}
        show_volume = bool("Volume" in plot_df.columns and plot_df["Volume"].notna().any())

        last_update = enriched.index[-1] if not enriched.empty else None
        footer = ""
        if last_update is not None:
            try:
                footer = f"Atualizado: {pd.Timestamp(last_update).strftime('%d/%m/%Y %H:%M')} (BRT)"
            except Exception:
                footer = f"Atualizado: {last_update}"

        fig, axes = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            addplot=addplots or None,
            volume=show_volume,
            title=title,
            ylabel="Preço (R$)",
            ylabel_lower="Volume" if show_volume else "",
            figsize=(width / 100, (height or DEFAULT_HEIGHT) / 100),
            returnfig=True,
            show_nontrading=show_nontrading,
            datetime_format=DATE_FORMAT_BY_TF.get(tf_key, "%d/%m %H:%M"),
        )

        ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes

        locator, formatter = _mpl_locator_formatter(tf_key)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        tick_labels = ax.get_xticklabels()
        logger.info("Total de ticks gerados: %s para timeframe %s", len(tick_labels), tf_key)
        
        for label in tick_labels:
            label.set_rotation(90)
            label.set_horizontalalignment("center")
            label.set_fontsize(7)

        # Controle adaptativo de visibilidade baseado no timeframe e número de candles
        if tf_key in {"1d"}:
            max_ticks = 10
        elif tf_key in {"1h", "60m"}:
            max_ticks = 8
        elif tf_key in {"1w"}:
            max_ticks = 8
        else:
            max_ticks = min(dynamic_max_ticks, 8)

        if len(tick_labels) > max_ticks:
            step = int(np.ceil(len(tick_labels) / max_ticks))
            visible_count = 0
            for idx, label in enumerate(tick_labels):
                is_visible = (idx % step == 0)
                label.set_visible(is_visible)
                if is_visible:
                    visible_count += 1
            logger.info("Ticks visíveis após filtro: %s de %s (step=%s)", visible_count, len(tick_labels), step)

        plt.setp(ax.xaxis.get_minorticklabels(), visible=False)

        if legend_entries:
            handles = [
                Line2D([], [], color=color, linestyle=linestyle, linewidth=width_line, label=label)
                for label, color, linestyle, width_line in legend_entries
            ]
            ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

        if axis_limits is not None:
            start_ts, end_ts = axis_limits
            if end_ts > start_ts:
                ax.set_xlim(start_ts, end_ts)

        if operation is not None:
            _draw_operation_levels(ax, plot_df, operation)

        if footer:
            try:
                fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=7, color="#4b5563")
            except Exception:
                logger.debug("Falha ao adicionar rodapé de atualização", exc_info=True)

        # mplfinance cria eixos extras (ex: volume) que nem sempre são compatíveis com tight_layout.
        try:
            if not show_volume:
                fig.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.95)
            else:
                fig.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.95)
        except Exception:
            logger.debug("Falha ao ajustar layout do gráfico", exc_info=True)

        buffer = io.BytesIO()
        # Evita bbox_inches='tight' aqui para não gerar imagens gigantes em casos extremos.
        fig.savefig(buffer, format="png", dpi=160)
        plt.close(fig)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + encoded

    except Exception as exc:
        logger.exception("Erro ao renderizar gráfico de preço", exc_info=True)
        raise


def _draw_operation_levels(ax, df: pd.DataFrame, operation: Operation) -> None:
    try:
        levels: List[Tuple[str, float, str]] = []
        for attr, label, color in (
            ("entrada", "Entrada", "#1E88E5"),
            ("alvo", "Alvo", "#0B8043"),
            ("stop", "Stop", "#D32F2F"),
        ):
            value = getattr(operation, attr, None)
            if value is not None:
                levels.append((label, float(value), color))

        if getattr(operation, "entrada_min", None) is not None and getattr(operation, "entrada_max", None) is not None:
            try:
                faixa_min = float(operation.entrada_min)
                faixa_max = float(operation.entrada_max)
                if faixa_min != faixa_max:
                    ax.axhspan(min(faixa_min, faixa_max), max(faixa_min, faixa_max), color="#1E88E5", alpha=0.08)
            except Exception:
                logger.debug("Não foi possível desenhar faixa de entrada", exc_info=True)

        for label, price, color in levels:
            ax.axhline(y=price, linestyle="--", color=color, linewidth=1.4, alpha=0.85)
            ax.text(
                df.index[-1],
                price,
                f" {label}: R$ {price:.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.8, "pad": 2},
            )

        for attr, label, color in (
            ("support_level", "Suporte", "#6b7280"),
            ("resistance_level", "Resistência", "#6b7280"),
        ):
            value = getattr(operation, attr, None)
            if value is None:
                continue
            price = float(value)
            ax.axhline(y=price, linestyle=":", color=color, linewidth=1.2, alpha=0.9)
            ax.text(
                df.index[-1],
                price,
                f" {label}: R$ {price:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                color=color,
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.75, "pad": 2},
            )
    except Exception as exc:
        logger.warning("Falha ao desenhar níveis da operação: %s", exc, exc_info=True)


class ChartGenerator:
    """Renderiza gráficos para a interface web e relatórios PDF."""

    def create_plotly_chart(
        self,
        df: pd.DataFrame,
        title: str = "",
        timeframe: Optional[str] = None,
    ) -> go.Figure:
        if df is None or df.empty:
            raise ValueError("Empty dataframe passed to create_plotly_chart")

        from plotly.subplots import make_subplots

        normalized = prepare_ohlc_dataframe(df, timeframe, reindex=False)
        normalized = _tail_for_timeframe(normalized, timeframe)
        enriched = _add_indicators(normalized)

        has_volume = "volume" in enriched.columns and pd.to_numeric(enriched["volume"], errors="coerce").fillna(0).sum() > 0
        rows = 2 if has_volume else 1
        row_heights = [0.78, 0.22] if has_volume else [1.0]
        figure = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=row_heights,
        )

        figure.add_trace(
            go.Candlestick(
                x=enriched.index,
                open=enriched["open"],
                high=enriched["high"],
                low=enriched["low"],
                close=enriched["close"],
                name="Preço",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a",
                decreasing_fillcolor="#ef5350",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        for column, label, color, dash, width in OVERLAY_STYLE:
            if column in enriched.columns:
                series = pd.to_numeric(enriched[column], errors="coerce").dropna()
                if len(series.index) >= 2:
                    figure.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode="lines",
                            name=label,
                            line={"color": color, "dash": dash, "width": width},
                        ),
                        row=1,
                        col=1,
                    )

        if has_volume:
            volume_series = pd.to_numeric(enriched["volume"], errors="coerce").fillna(0)
            inc = (enriched["close"] >= enriched["open"]).astype(bool)
            colors_bar = np.where(inc, "#26a69a", "#ef5350")
            figure.add_trace(
                go.Bar(
                    x=enriched.index,
                    y=volume_series.values,
                    name="Volume",
                    marker_color=colors_bar,
                    opacity=0.55,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        tf_key = (timeframe or "").lower()
        tick_format = DATE_FORMAT_BY_TF.get(tf_key, "%d/%m %H:%M")
        rangebreaks = []
        if tf_key in INTRADAY_SET:
            rangebreaks = [
                {"bounds": ["sat", "mon"]},
                {"bounds": [17, 10], "pattern": "hour"},
            ]
        figure.update_layout(
            title=title or "",
            template="plotly_white",
            margin={"l": 50, "r": 20, "t": 58, "b": 40},
            legend={"orientation": "h", "y": -0.15},
        )

        figure.update_xaxes(
            rangeslider_visible=False,
            type="date",
            tickformat=tick_format,
            rangebreaks=rangebreaks,
            showgrid=True,
            gridcolor="#eef2f7",
        )
        figure.update_yaxes(title_text="Preço (R$)", row=1, col=1)
        if has_volume:
            figure.update_yaxes(title_text="Volume", row=2, col=1)
        return figure

    def generate_chart_image(
        self,
        df: pd.DataFrame,
        title: str = "",
        timeframe: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: Optional[int] = None,
        operation: Optional[Operation] = None,
    ) -> str:
        return render_price_chart_base64(df, title, timeframe, width, height, operation)

    @staticmethod
    def export_png_base64(
        df: pd.DataFrame,
        title: str = "",
        timeframe: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: Optional[int] = None,
        operation: Optional[Operation] = None,
    ) -> str:
        return render_price_chart_base64(df, title, timeframe, width, height, operation)

    @staticmethod
    def prepare_ohlc_dataframe(
        df: pd.DataFrame,
        timeframe: Optional[str] = None,
        remove_nontrading: bool = True,
        market_hours: bool = True,
        reindex: bool = False,
    ) -> pd.DataFrame:
        return prepare_ohlc_dataframe(
            df,
            timeframe=timeframe,
            remove_nontrading=remove_nontrading,
            market_hours=market_hours,
            reindex=reindex,
        )

    @staticmethod
    def _ensure_overlay_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return _add_indicators(df)

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        return _add_indicators(df)

    @staticmethod
    def _get_date_format(timeframe: Optional[str]) -> str:
        if timeframe and timeframe.lower() in DATE_FORMAT_BY_TF:
            return DATE_FORMAT_BY_TF[timeframe.lower()]
        return "%d/%m %H:%M"

    @staticmethod
    def _get_height_for_timeframe(timeframe: Optional[str], fallback: int = DEFAULT_HEIGHT) -> int:
        if timeframe and timeframe.lower() in HEIGHT_BY_TF:
            return HEIGHT_BY_TF[timeframe.lower()]
        return fallback


__all__ = [
    "ChartGenerator",
    "prepare_ohlc_dataframe",
    "render_price_chart_base64",
    "format_timeframe_label",
]