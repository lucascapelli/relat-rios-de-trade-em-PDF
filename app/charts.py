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
from matplotlib import ticker as mticker
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
    "15m": "%d/%m %H:%M",
    "30m": "%d/%m %H:%M",
    "45m": "%d/%m %H:%M",
    "1h": "%d/%m %H:%M",
    "60m": "%d/%m %H:%M",
    "4h": "%d/%m %H:%M",
    "1d": "%d/%m/%Y",
    "1w": "%d/%m/%Y",
}

HEIGHT_BY_TF: Dict[str, int] = {
    "1m": 440,
    "5m": 430,
    "15m": 450,
    "30m": 420,
    "45m": 400,
    "1h": 400,
    "60m": 400,
    "4h": 380,
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

# incluiu 60m como intraday — necessário para consistência
INTRADAY_SET = {"1m", "5m", "15m", "30m", "45m", "1h", "60m", "4h"}

TIMEFRAME_LABELS: Dict[str, str] = {
    "1m": "1 min",
    "5m": "5 min",
    "15m": "15 min",
    "30m": "30 min",
    "45m": "45 min",
    "1h": "60 min",
    "60m": "60 min",
    "4h": "4h",
    "1d": "Diário",
    "1w": "Semanal",
}

MAX_BARS_BY_TF: Dict[str, int] = {
    "1m": 280,
    "5m": 260,
    "15m": 220,
    "30m": 200,
    "45m": 180,
    "1h": 180,
    "60m": 180,
    "4h": 180,
    "1d": 90,
    "1w": 120,
}

OVERLAY_STYLE: Sequence[Tuple[str, str, str, str, float]] = (
    ("SMA_9", "SMA 9", "#8E24AA", "solid", 1.2),
    ("SMA_21", "SMA 21", "#FB8C00", "solid", 1.2),
    ("EMA_21", "MME 21", "#1E88E5", "dash", 1.4),
    ("EMA_200", "MME 200", "#D32F2F", "dash", 1.6),
)

_TIME_WINDOW_BY_TF: Dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(days=1),
    "5m": pd.Timedelta(days=2),
    "15m": pd.Timedelta(days=4),   # 4 dias conforme solicitado
    "30m": pd.Timedelta(days=7),
    "45m": pd.Timedelta(days=10),
    "1h": pd.Timedelta(days=7),    # 7 dias para 1h
    "60m": pd.Timedelta(days=7),   # 60m também
    "4h": pd.Timedelta(days=60),
    "1d": pd.Timedelta(days=5),    # 5 dias para diário (visual)
    "1w": pd.Timedelta(days=365),
}

_MIN_CANDLES_FOR_WINDOW: Dict[str, int] = {
    "15m": 50,
    "30m": 40,
    "45m": 30,
    "1h": 20,
    "60m": 20,
    "4h": 20,
    "1d": 5,
    "1w": 20,
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


def _apply_time_window(df: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
    """Garante recorte por janela temporal fixa (em dias) para cada timeframe.

    - 15m → últimos 4 dias
    - 60m/1h → últimos 7 dias
    - 1d → últimos 5 dias
    - demais seguem _TIME_WINDOW_BY_TF
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    tf_key = (timeframe or "").strip().lower()
    window = _TIME_WINDOW_BY_TF.get(tf_key)
    if window is None:
        return df

    end_ts = df.index[-1]
    start_ts = end_ts - window
    window_df = df.loc[df.index >= start_ts]
    return window_df if not window_df.empty else df


def _debug_time_data(df: Optional[pd.DataFrame], name: str = "Dataframe") -> None:
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


def _log_stage(df: Optional[pd.DataFrame], stage: str) -> None:
    """Loga info resumida sobre datas para depuração."""
    if df is None or df.empty:
        logger.info("[%s] vazio", stage)
        return
    try:
        start = df.index[0]
        end = df.index[-1]
        span_days = (end - start).total_seconds() / 86400 if isinstance(end, pd.Timestamp) else None
        logger.info(
            "[%s] rows=%s start=%s end=%s span_dias=%.2f",
            stage,
            len(df),
            start,
            end,
            span_days if span_days is not None else -1,
        )
    except Exception:
        logger.debug("Falha ao logar estágio %s", stage, exc_info=True)


# -------------------------
# Helpers de tempo
# -------------------------
def _clamp_future_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove candles com timestamp no futuro (causa comum de janelas ancoradas em datas erradas)."""
    if frame is None or frame.empty:
        return frame
    # Usa fuso America/Sao_Paulo e remove tz para comparar com índice já normalizado.
    now = pd.Timestamp.now(tz="America/Sao_Paulo").tz_localize(None)
    clamped = frame.loc[frame.index <= now]
    if clamped.empty:
        # se tudo estiver no futuro, mantemos o original para diagnóstico (logger) e retornamos empty
        logger.warning("Todos os timestamps estavam no futuro; resultado vazio após clamp")
    return clamped


def _enforce_final_window(df: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
    """Aplica janela final obrigatória baseada no timeframe.

    Garante que o eixo X não extrapole além do desejado, mesmo se etapas anteriores falharem.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    tf_key = (timeframe or "").strip().lower()
    window = _TIME_WINDOW_BY_TF.get(tf_key)
    if window is None:
        return df

    end_ts = df.index.max()
    start_ts = end_ts - window
    clipped = df.loc[df.index >= start_ts]
    return clipped if not clipped.empty else df


# ============================================================================
# FUNÇÕES REFATORADAS - prepare_ohlc_dataframe
# ============================================================================

def prepare_ohlc_dataframe(
    df: pd.DataFrame,
    timeframe: Optional[str] = None,
    remove_nontrading: bool = True,
    market_hours: bool = True,
    reindex: bool = False,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame OHLC vazio ou nulo")

    frame = df.copy()

    frame = _normalize_index(frame)
    frame = _process_time_column(frame, df)
    frame = _convert_numeric_columns(frame)
    frame = _clean_and_sort(frame)
    frame = _normalize_timezone(frame, timezone)
    frame = _filter_outlier_dates(frame)

    tf_key = (timeframe or "").lower()
    frame = _apply_market_filters(frame, tf_key, market_hours, remove_nontrading)

    if reindex and not frame.empty:
        frame = _reindex_timeseries(frame, tf_key)

    return frame


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex):
        index_names = list(frame.index.names)
        if "time" in index_names:
            renamed = [f"{name}_index" if name == "time" else name for name in index_names]
            frame.index = frame.index.rename(renamed)
    elif frame.index.name == "time":
        frame.index = frame.index.rename("time_index")
    return frame


def _process_time_column(frame: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    if "time" in frame.columns:
        time_values = frame["time"]
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame["time"] = frame.index
        time_values = frame["time"]
    else:
        index_name = frame.index.name or "index"
        frame = frame.reset_index().rename(columns={index_name: "time"})
        time_values = frame["time"]

    frame["time"] = _parse_time_values(time_values, frame, original_df)
    return frame


def _parse_time_values(
    time_values: pd.Series,
    frame: pd.DataFrame,
    original_df: pd.DataFrame
) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(time_values):
        return pd.to_datetime(time_values, errors="coerce")

    if pd.api.types.is_integer_dtype(time_values):
        numeric = pd.to_numeric(time_values, errors="coerce")
        parsed = numeric.dropna()

        if parsed.empty:
            return pd.to_datetime(time_values, errors="coerce", dayfirst=True)

        if _is_sequential_index(parsed):
            return _resolve_sequential_index(frame, original_df)

        return _parse_unix_timestamp(numeric)

    # Datas no padrão brasileiro podem ser invertidas pelo parser default.
    # dayfirst=True corrige datas dd/mm/aaaa vindas do front/API.
    return pd.to_datetime(time_values, errors="coerce", dayfirst=True)


def _is_sequential_index(values: pd.Series, tolerance: float = 0.1) -> bool:
    try:
        if values.empty or len(values) < 3:
            return False

        v = values.astype("int64")
        min_val = v.min()
        if min_val not in (0, 1):
            return False

        max_val = v.max()
        expected_range = max_val - min_val + 1
        actual_count = len(v.unique())
        return actual_count >= (1 - tolerance) * expected_range

    except Exception:
        return False


def _resolve_sequential_index(frame: pd.DataFrame, original_df: pd.DataFrame) -> pd.Series:
    if "time_str" in frame.columns:
        return pd.to_datetime(frame["time_str"], errors="coerce")

    if isinstance(original_df.index, pd.DatetimeIndex):
        return original_df.index

    raise ValueError(
        "Coluna 'time' parece ser índice sequencial, mas não foi encontrada "
        "informação de datetime válida (procurou: time_str, DatetimeIndex)"
    )


def _parse_unix_timestamp(numeric: pd.Series) -> pd.Series:
    median_value = float(numeric.median())
    abs_value = abs(median_value)

    if abs_value >= 1e17:
        unit = "ns"
    elif abs_value >= 1e14:
        unit = "us"
    elif abs_value >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    timestamps = pd.to_datetime(numeric, unit=unit, origin="unix", errors="coerce")

    if not timestamps.empty and pd.notnull(timestamps.median()) and timestamps.median().year < 2000:
        logger.warning(
            f"Timestamps parseados estão antes de 2000 (mediana: {timestamps.median()}). "
            f"Verifique se a unidade '{unit}' está correta."
        )

    return timestamps


def _convert_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _clean_and_sort(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["time"]).copy()
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("time")
    frame = frame.drop_duplicates(subset=["time"], keep="last")
    frame = frame.set_index("time")
    return frame


def _filter_outlier_dates(
    frame: pd.DataFrame,
    min_year: int = 1990,
    max_future_days: int = 3,
) -> pd.DataFrame:
    """Remove candles com datas inválidas (muito antigas ou muito no futuro).

    Uma única candle fora do eixo comprime os rótulos e torna as datas ilegíveis
    nos PNG/PDF exportados.
    """
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame

    now = pd.Timestamp.now()
    max_allowed = now + pd.Timedelta(days=max_future_days)
    mask = (frame.index.year >= min_year) & (frame.index <= max_allowed)

    if not mask.all():
        removed = len(frame) - mask.sum()
        logger.warning(
            "Removendo %s candles fora do intervalo temporal (%s a %s)",
            removed,
            min_year,
            max_allowed.date(),
        )

    return frame.loc[mask]


def _normalize_timezone(frame: pd.DataFrame, target_tz: str) -> pd.DataFrame:
    try:
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_convert(target_tz).tz_localize(None)
    except Exception as e:
        logger.warning(f"Falha ao converter timezone para {target_tz}: {e}")
        try:
            frame.index = frame.index.tz_localize(None)
        except Exception:
            pass
    return frame


def _apply_market_filters(
    frame: pd.DataFrame,
    tf_key: str,
    market_hours: bool,
    remove_nontrading: bool
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame

    # Filtro de horário (apenas para intraday) — só aplica se index for DatetimeIndex
    if market_hours and tf_key in INTRADAY_SET and isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.between_time(HOUR_START, HOUR_END)
        except Exception as e:
            logger.warning(f"Falha ao filtrar horário de pregão: {e}")

    if remove_nontrading:
        frame = frame[frame.index.dayofweek < 5]
        if len(HOLIDAY_INDEX) > 0:
            frame = frame[~frame.index.normalize().isin(HOLIDAY_INDEX)]

    return frame


def _reindex_timeseries(frame: pd.DataFrame, tf_key: str) -> pd.DataFrame:
    freq_map = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "45m": "45T",
        "1h": "1H",
        "60m": "1H",
        "4h": "4H",
        "1d": "1D",
        "1w": "1W-MON",
    }

    freq = freq_map.get(tf_key)
    if not freq:
        logger.warning(f"Timeframe '{tf_key}' não suportado para reindex")
        return frame

    try:
        full_index = pd.date_range(start=frame.index[0], end=frame.index[-1], freq=freq)
        frame = frame.reindex(full_index)
        frame = frame.dropna(subset=["open", "close"], how="all")
    except Exception as e:
        logger.warning(f"Falha ao reindexar com freq '{freq}': {e}")

    return frame


# ============================================================================
# Locator/Formatter
# ============================================================================

def get_mpl_locator_formatter(
    timeframe: str,
    span_days: float
) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    tf_key = (timeframe or "").lower()
    if tf_key in INTRADAY_SET:
        if span_days <= 1:
            return _get_intraday_single_day_format(tf_key)
        else:
            return _get_intraday_multi_day_format(span_days)
    if tf_key == "1d":
        return _get_daily_format(span_days)
    if tf_key == "1w":
        return _get_weekly_format(span_days)
    return _get_auto_format()


def _get_intraday_single_day_format(tf_key: str) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    config = {
        "1m": (3, "%d/%m %Hh"),
        "5m": (3, "%d/%m %Hh"),
        "15m": (2, "%d/%m %Hh"),
        "30m": (8, "%d/%m %Hh"),
        "45m": (8, "%d/%m %Hh"),
        "1h": (8, "%d/%m %Hh"),
        "60m": (8, "%d/%m %Hh"),
        "4h": (4, "%d/%m %Hh"),
    }
    interval, fmt = config.get(tf_key, (4, "%d/%m %Hh"))
    locator = mdates.HourLocator(interval=interval)
    formatter = mdates.DateFormatter(fmt)
    return locator, formatter


def _get_intraday_multi_day_format(span_days: float) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    """
    Para intraday multi-day, usar AutoDateLocator (evita 'embaralhamento' causado por DayLocator
    quando há gaps intraday/feriados) e manter formato com hora.
    """
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%d/%m %Hh")
    return locator, formatter


def _get_daily_format(span_days: float) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    if span_days <= 30:
        interval = max(1, int(span_days // 6))
    elif span_days <= 90:
        interval = max(1, int(span_days // 10))
    else:
        interval = max(1, int(span_days // 15))
    locator = mdates.DayLocator(interval=interval)
    formatter = mdates.DateFormatter("%d/%m")
    return locator, formatter


def _get_weekly_format(span_days: float) -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    approx_months = max(1, int(span_days // 30))
    interval = max(1, approx_months // 5)
    locator = mdates.MonthLocator(interval=interval)
    formatter = mdates.DateFormatter("%m/%Y")
    return locator, formatter


def _get_auto_format() -> Tuple[mdates.DateLocator, mdates.DateFormatter]:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    return locator, formatter


def _force_xlim(ax, index: pd.DatetimeIndex) -> None:
    """Força limites do eixo X para o range real dos dados, evitando ticks fora da janela."""
    if index is None or len(index) == 0:
        return
    try:
        ax.set_xlim(index.min(), index.max())
    except Exception:
        logger.debug("Não foi possível forçar xlim", exc_info=True)


def _log_ohlc_health(df: pd.DataFrame, label: str) -> None:
    """Loga estatísticas básicas para entender por que velas podem não aparecer."""
    if df is None or df.empty:
        logger.warning("[%s] dataframe vazio", label)
        return
    try:
        cols = [c for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume"}]
        subset = df[cols]
        notna_counts = subset.notna().sum().to_dict()
        min_price = subset[[c for c in subset.columns if c.lower() in {"open", "high", "low", "close"}]].min().min()
        max_price = subset[[c for c in subset.columns if c.lower() in {"open", "high", "low", "close"}]].max().max()
        logger.info(
            "[%s] rows=%s notna=%s min_price=%.4f max_price=%.4f head=%s tail=%s",
            label,
            len(df),
            notna_counts,
            float(min_price) if pd.notna(min_price) else -1.0,
            float(max_price) if pd.notna(max_price) else -1.0,
            subset.head(1).to_dict(orient="records"),
            subset.tail(1).to_dict(orient="records"),
        )
    except Exception:
        logger.debug("Falha ao logar saúde OHLC para %s", label, exc_info=True)


# ============================================================================
# Indicadores e plotagem
# ============================================================================

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
        tf_key = (timeframe or "").lower()
        normalized = prepare_ohlc_dataframe(df, timeframe)
        _log_stage(normalized, "prep")

        normalized = _clamp_future_index(normalized)
        _log_stage(normalized, "clamp_future")

        normalized = _tail_for_timeframe(normalized, timeframe)
        _log_stage(normalized, "tail")

        normalized = _apply_time_window(normalized, timeframe)
        _log_stage(normalized, "apply_window")

        normalized = _enforce_final_window(normalized, timeframe)
        _log_stage(normalized, "enforce_final_window")

        enriched = _add_indicators(normalized)
        if enriched is None or enriched.empty:
            raise ValueError("Dados insuficientes para plotagem (após remoção de timestamps futuros)")

        # Renomeia colunas para mpf/plotly
        plot_df = enriched.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )

        try:
            logger.info(
                "[mpf] candles=%s start=%s end=%s cols=%s",
                len(plot_df),
                plot_df.index.min() if not plot_df.empty else None,
                plot_df.index.max() if not plot_df.empty else None,
                list(plot_df.columns),
            )
        except Exception:
            logger.debug("Falha ao logar mpf plot_df", exc_info=True)

        _log_ohlc_health(plot_df, "mpf_plot_df")

        addplots: List = []
        legend_entries: List[Tuple[str, str, str, float]] = []
        for column, label, color, linestyle, width_line in OVERLAY_STYLE:
            if column in plot_df and plot_df[column].notna().any():
                series = plot_df[column].dropna()
                if len(series) >= 2:
                    addplots.append(
                        mpf.make_addplot(
                            series,
                            color=color,
                            linestyle="--" if "dash" in linestyle else "-",
                            width=width_line,
                        )
                    )
                    legend_entries.append((label, color, "--" if "dash" in linestyle else "-", width_line))

        market_colors = mpf.make_marketcolors(
            up="#2c8f74",
            down="#d9534f",
            edge="#444444",
            wick="#444444",
            volume="inherit",
            ohlc="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="yahoo",
            marketcolors=market_colors,
            facecolor="white",
            figcolor="white",
            gridstyle=":",
            rc={"font.size": 8, "axes.labelsize": 8},
        )

        show_volume = bool("Volume" in plot_df.columns and plot_df["Volume"].notna().any())
        figsize = (width / 100.0, (height or DEFAULT_HEIGHT) / 100.0)
        compress_intraday = tf_key in INTRADAY_SET

        fig, axes = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            addplot=addplots or None,
            volume=show_volume,
            title=title,
            ylabel="Preço (R$)",
            ylabel_lower="Volume" if show_volume else "",
            figsize=figsize,
            returnfig=True,
            # Só comprime fora do pregão para intraday; diário/weekly mantêm eixo de data real.
            show_nontrading=not compress_intraday,
            datetime_format=DATE_FORMAT_BY_TF.get(tf_key, "%d/%m %H:%M"),
            xrotation=45,
        )

        ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes

        # Calcula span em dias de forma robusta
        span = plot_df.index[-1] - plot_df.index[0]
        span_days = span.total_seconds() / 86400.0 if span.total_seconds() >= 0 else 0.0

        if tf_key in INTRADAY_SET:
            # Eixo posicional: fixa range e rótulos com datas reais alinhadas às posições.
            idx_list = list(plot_df.index)
            last_pos = max(0, len(idx_list) - 1)
            ax.set_xlim(-0.5, last_pos + 0.5)

            max_ticks = 8
            raw_positions = np.linspace(0, last_pos, num=min(last_pos + 1, max_ticks))
            positions = sorted(set(int(round(p)) for p in raw_positions))

            def _fmt_positional(x: float, _pos: int) -> str:
                i = int(round(x))
                if i < 0 or i >= len(idx_list):
                    return ""
                ts = pd.Timestamp(idx_list[i])
                fmt = DATE_FORMAT_BY_TF.get(tf_key, "%d/%m %H:%M")
                return ts.strftime(fmt)

            ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_positional))
        else:
            locator, formatter = get_mpl_locator_formatter(tf_key, span_days)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            # Garante que o eixo não extrapole para o dia seguinte em diário/weekly.
            try:
                ax.set_xlim(plot_df.index.min(), plot_df.index.max())
            except Exception:
                logger.debug("Falha ao ajustar xlim para eixo de data real", exc_info=True)

        try:
            logger.info(
                "[mpf_axes] xlim=%s ylim=%s", ax.get_xlim(), ax.get_ylim()
            )
        except Exception:
            logger.debug("Falha ao logar limites do eixo", exc_info=True)

        # Força atualização dos ticks e ajusta visibilidade evitando 'embaralhamento'
        fig.canvas.draw()
        tick_labels = ax.get_xticklabels()
        num_ticks = len(tick_labels)
        max_ticks = 8

        if num_ticks > max_ticks:
            step = max(1, num_ticks // max_ticks)
            for idx, label in enumerate(tick_labels):
                if idx % step != 0:
                    label.set_visible(False)
                else:
                    label.set_rotation(45)
                    label.set_horizontalalignment("right")
                    label.set_fontsize(7)
        else:
            for label in tick_labels:
                label.set_rotation(45)
                label.set_horizontalalignment("right")
                label.set_fontsize(7)

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="major", labelsize=8)
        ax.tick_params(axis="x", which="major", labelsize=7, pad=2)

        if legend_entries:
            handles = [
                Line2D([], [], color=color, linestyle=linestyle, linewidth=width_line, label=label)
                for label, color, linestyle, width_line in legend_entries
            ]
            ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)

        if operation is not None:
            _draw_operation_levels(ax, plot_df, operation)

        try:
            ymin, ymax = ax.get_ylim()
            pad = (ymax - ymin) * 0.06
            ax.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            pass

        try:
            if show_volume:
                fig.subplots_adjust(top=0.92, bottom=0.25, left=0.10, right=0.96, hspace=0.08)
            else:
                fig.subplots_adjust(top=0.94, bottom=0.25, left=0.10, right=0.96)
        except Exception:
            pass

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=200)
        try:
            logger.info("[mpf_png] bytes=%s", len(buffer.getvalue()))
        except Exception:
            logger.debug("Falha ao logar tamanho do PNG", exc_info=True)
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
        _log_stage(normalized, "prep_plotly")

        normalized = _clamp_future_index(normalized)
        _log_stage(normalized, "clamp_future_plotly")

        normalized = _tail_for_timeframe(normalized, timeframe)
        _log_stage(normalized, "tail_plotly")

        normalized = _apply_time_window(normalized, timeframe)
        _log_stage(normalized, "apply_window_plotly")

        normalized = _enforce_final_window(normalized, timeframe)
        _log_stage(normalized, "enforce_final_plotly")

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

        try:
            logger.info(
                "[plotly] candles=%s start=%s end=%s cols=%s",
                len(enriched),
                enriched.index.min() if not enriched.empty else None,
                enriched.index.max() if not enriched.empty else None,
                list(enriched.columns),
            )
        except Exception:
            logger.debug("Falha ao logar plotly enriched", exc_info=True)

        _log_ohlc_health(enriched, "plotly_enriched")

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

        try:
            logger.info(
                "[plotly_traces] candles=%s points=%s", len(enriched), len(enriched.index)
            )
        except Exception:
            logger.debug("Falha ao logar plotly traces", exc_info=True)

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
            # Remove fins de semana e o período em que o mercado está fechado (17h→10h).
            rangebreaks = [
                {"bounds": ["sat", "mon"]},
                {"pattern": "hour", "bounds": [17, 10]},
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