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
    "1m": "%H:%M",
    "5m": "%H:%M",
    "6m": "%H:%M",
    "15m": "%H:%M",
    "30m": "%H:%M",
    "45m": "%H:%M",
    "1h": "%d/%m %H:%M",
    "4h": "%d/%m %H:%M",
    "1d": "%d/%m",
    "1w": "%d/%m/%Y",
}

HEIGHT_BY_TF: Dict[str, int] = {
    "1m": 440,
    "5m": 430,
    "6m": 430,
    "15m": 420,
    "30m": 400,
    "45m": 390,
    "1h": 380,
    "4h": 360,
    "1d": 360,
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

OVERLAY_STYLE: Sequence[Tuple[str, str, str, str, float]] = (
    ("SMA_9", "SMA 9", "#8E24AA", "solid", 1.2),
    ("SMA_21", "SMA 21", "#FB8C00", "solid", 1.2),
    ("EMA_21", "MME 21", "#1E88E5", "dash", 1.4),
    ("EMA_200", "MME 200", "#D32F2F", "dash", 1.6),
)


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

    if "time" in frame.columns:
        time_values = frame["time"]
    else:
        index_name = frame.index.name or "index"
        frame = frame.reset_index().rename(columns={index_name: "time"})
        time_values = frame["time"]

    if pd.api.types.is_integer_dtype(time_values):
        parsed = pd.to_numeric(time_values, errors="coerce").dropna()
        if parsed.empty:
            frame["time"] = pd.to_datetime(time_values, errors="coerce")
        else:
            median_value = float(parsed.median())
            unit = "ms" if median_value > 1_000_000_000_000 else "s"
            frame["time"] = pd.to_datetime(time_values, unit=unit, origin="unix", errors="coerce")
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
    try:
        normalized = prepare_ohlc_dataframe(df, timeframe)
        enriched = _add_indicators(normalized)
    except Exception as exc:
        logger.error("Falha ao preparar dados do gráfico: %s", exc, exc_info=True)
        return ""

    if len(enriched.index) < 2:
        return ""

    plot_df = enriched.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )

    addplots: List = []
    legend_entries: List[Tuple[str, str, str, float]] = []
    for column, label, color, linestyle, width_line in OVERLAY_STYLE:
        if column in plot_df and plot_df[column].notna().any():
            series = plot_df[column].dropna()
            if len(series.index) >= 2:
                addplots.append(
                    mpf.make_addplot(series, color=color, linestyle=linestyle.replace("dash", "--"), width=width_line)
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

    tf_key = (timeframe or "").lower()
    show_nontrading = tf_key in {"1d", "1w"}

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        addplot=addplots or None,
        volume=False,
        title=title,
        ylabel="Preço (R$)",
        figsize=(width / 100, (height or DEFAULT_HEIGHT) / 100),
        returnfig=True,
        show_nontrading=show_nontrading,
        datetime_format=DATE_FORMAT_BY_TF.get(tf_key, "%d/%m %H:%M"),
    )

    ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes

    if tf_key in {"1m", "5m", "6m", "15m", "30m", "45m"}:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif tf_key in {"1h", "4h"}:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    elif tf_key == "1d":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    elif tf_key == "1w":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if legend_entries:
        handles = [
            Line2D([], [], color=color, linestyle=linestyle, linewidth=width_line, label=label)
            for label, color, linestyle, width_line in legend_entries
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    if operation is not None:
        _draw_operation_levels(ax, plot_df, operation)

    fig.tight_layout()

    buffer = io.BytesIO()
    # Evita bbox_inches='tight' aqui para não gerar imagens gigantes em casos extremos.
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded


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

        normalized = prepare_ohlc_dataframe(df, timeframe, reindex=False)
        enriched = _add_indicators(normalized)

        figure = go.Figure()
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
            )
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
                        )
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
            xaxis={
                "rangeslider": {"visible": False},
                "tickformat": tick_format,
                "rangebreaks": rangebreaks,
            },
            yaxis={"title": "Preço (R$)"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            legend={"orientation": "h", "y": -0.15},
        )
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
]
