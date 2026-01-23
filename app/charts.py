"""Chart generation utilities for interactive and static outputs."""
from __future__ import annotations

import base64
import io
import math
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import dates as mdates

from .models import Operation
from .utils import logger


class ChartGenerator:
    @staticmethod
    def create_plotly_chart(
        df: pd.DataFrame,
        title: str = "",
        show_volume: bool = True,
        show_indicators: bool = True,
    ) -> go.Figure:
        if df is None or df.empty:
            logger.warning("DataFrame vazio recebido para gráfico")
            return go.Figure()

        df = ChartGenerator._prepare_dataframe(df)

        if len(df) < 2:
            logger.warning(f"Poucos dados válidos: {len(df)} candles")
            return go.Figure()

        candlestick = go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Preço",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            line=dict(width=1),
            whiskerwidth=0.8,
            hoverinfo="text",
            text=ChartGenerator._generate_hover_text(df),
        )

        data = [candlestick]

        if show_indicators:
            data.extend(ChartGenerator._add_indicators(df))

        layout = ChartGenerator._create_layout(title, show_volume)

        if show_volume and "volume" in df.columns:
            data.append(ChartGenerator._create_volume_trace(df))
            layout = ChartGenerator._add_volume_axis(layout)

        fig = go.Figure(data=data, layout=layout)
        fig = ChartGenerator._add_range_selector(fig)

        return fig

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "time_str" in df.columns and "time" not in df.columns:
            df["time"] = pd.to_datetime(df["time_str"], errors="coerce")
        elif "time" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], errors="coerce")

        if hasattr(df["time"].dt, "tz") and df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[
            (df["high"] >= df["low"]) &
            (df["close"] <= df["high"]) & (df["close"] >= df["low"]) &
            (df["open"] <= df["high"]) & (df["open"] >= df["low"])
        ]

        return df

    @staticmethod
    def _generate_hover_text(df: pd.DataFrame) -> List[str]:
        return [
            f"Abertura: R$ {o:.2f}<br>Máxima: R$ {h:.2f}<br>Mínima: R$ {l:.2f}<br>Fechamento: R$ {c:.2f}<br>Volume: {v:,.0f}"
            for o, h, l, c, v in zip(
                df["open"],
                df["high"],
                df["low"],
                df["close"],
                df.get("volume", [0] * len(df)),
            )
        ]

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> List[go.Scatter]:
        traces: List[go.Scatter] = []

        for period, color in [(9, "#2196F3"), (21, "#FF9800")]:
            col = f"SMA_{period}"
            if col in df.columns and df[col].notna().sum() > 0:
                traces.append(
                    go.Scatter(
                        x=df["time"],
                        y=df[col],
                        mode="lines",
                        name=f"SMA {period}",
                        line=dict(color=color, width=1.5),
                        opacity=0.7,
                    )
                )

        if all(col in df.columns for col in ["BB_upper", "BB_lower", "BB_middle"]):
            if df["BB_upper"].notna().sum() > 0:
                traces.extend(
                    [
                        go.Scatter(
                            x=df["time"],
                            y=df["BB_upper"],
                            mode="lines",
                            name="BB Superior",
                            line=dict(color="rgba(158, 158, 158, 0.5)", width=1, dash="dash"),
                            showlegend=False,
                        ),
                        go.Scatter(
                            x=df["time"],
                            y=df["BB_lower"],
                            mode="lines",
                            name="BB Inferior",
                            fill="tonexty",
                            fillcolor="rgba(158, 158, 158, 0.1)",
                            line=dict(color="rgba(158, 158, 158, 0.5)", width=1, dash="dash"),
                            showlegend=False,
                        ),
                    ]
                )

        return traces

    @staticmethod
    def _create_layout(title: str, show_volume: bool) -> go.Layout:
        return go.Layout(
            title=dict(
                text=title,
                font=dict(size=16, color="#2c3e50", family="Arial Black"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                type="date",
                gridcolor="#ecf0f1",
                showgrid=True,
                rangeslider=dict(visible=False),
                nticks=15,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Preço (R$)",
                gridcolor="#ecf0f1",
                showgrid=True,
                side="right",
                tickformat=".2f",
                tickprefix="R$ ",
                tickfont=dict(size=10),
            ),
            height=550,
            margin=dict(l=50, r=60, t=70, b=50),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="#f8f9fa",
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#bdc3c7",
                borderwidth=1,
                font=dict(size=9),
            ),
            font=dict(family="Arial, sans-serif"),
        )

    @staticmethod
    def _create_volume_trace(df: pd.DataFrame) -> go.Bar:
        colors = [
            "rgba(38, 166, 154, 0.6)" if c >= o else "rgba(239, 83, 80, 0.6)"
            for c, o in zip(df["close"], df["open"])
        ]

        return go.Bar(
            x=df["time"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
            yaxis="y2",
            hoverinfo="y",
        )

    @staticmethod
    def _add_volume_axis(layout: go.Layout) -> go.Layout:
        layout.update(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="left",
                showgrid=False,
                tickformat=",.0f",
                tickfont=dict(size=9),
            )
        )
        return layout

    @staticmethod
    def _add_range_selector(fig: go.Figure) -> go.Figure:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=30, label="30min", step="minute", stepmode="backward"),
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=4, label="4h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all", label="Tudo"),
                    ]
                ),
                bgcolor="rgba(255, 255, 255, 0.9)",
                activecolor="#2196F3",
                x=0.01,
                y=1.05,
                font=dict(size=9),
            )
        )
        return fig

    @staticmethod
    def _configure_date_axis(ax, times: Any) -> None:
        if ax is None or times is None:
            return

        try:
            if isinstance(times, pd.Index):
                time_index = pd.to_datetime(times)
            else:
                time_index = pd.to_datetime(pd.Series(times))

            time_index = time_index.dropna().sort_values()
            if time_index.empty:
                return

            if hasattr(time_index, "tz") and time_index.tz is not None:
                time_index = time_index.tz_localize(None)

            min_step = None
            if len(time_index) > 1:
                diffs = time_index.to_series().diff().dropna()
                if not diffs.empty:
                    min_step = diffs.min()

            span = time_index.iloc[-1] - time_index.iloc[0]

            ticks_goal = 6
            span_seconds = max(span.total_seconds(), 1)
            span_hours = span_seconds / 3600.0
            span_days = span_seconds / 86400.0

            ticks_goal = max(ticks_goal, 3)
            total_points = len(time_index)
            step = max(1, math.ceil(total_points / ticks_goal))

            selected = time_index[::step]
            last_point = time_index.iloc[[-1]]
            if selected.empty:
                selected = last_point
            elif selected.iloc[-1] != last_point[0]:
                selected = selected.union(last_point)

            if min_step is not None and min_step <= pd.Timedelta(hours=6):
                label_builder = lambda dt: dt.strftime("%d/%m\n%H:%M")
            elif span <= pd.Timedelta(days=120):
                label_builder = lambda dt: dt.strftime("%d/%m")
            else:
                label_builder = lambda dt: dt.strftime("%d/%m/%y")

            tick_positions = mdates.date2num(selected.to_pydatetime())
            tick_labels = [label_builder(dt) for dt in selected]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.tick_params(axis="x", pad=4)
        except Exception as axis_error:
            logger.debug(f"Falha ao configurar eixo de datas: {axis_error}")

    @staticmethod
    def generate_chart_image(
        df: pd.DataFrame,
        title: str = "",
        width: int = 800,
        height: int = 400,
        operation: Optional[Operation] = None,
    ) -> str:
        try:
            if df is None or df.empty:
                return ""

            df_plot = df.copy()

            try:
                enhanced_image = ChartGenerator._generate_simple_chart(
                    df_plot,
                    title,
                    operation,
                    width,
                    height,
                )
                if enhanced_image:
                    return enhanced_image
            except Exception as enhanced_error:
                logger.warning(f"Falha no renderizador aprimorado do gráfico: {enhanced_error}")

            if "time" in df_plot.columns:
                df_plot.set_index("time", inplace=True)

            if df_plot.index.tz is not None:
                df_plot.index = df_plot.index.tz_localize(None)

            ema_21 = df_plot["EMA_21"].copy() if "EMA_21" in df_plot.columns else None
            ema_200 = df_plot["EMA_200"].copy() if "EMA_200" in df_plot.columns else None

            df_plot = df_plot.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                }
            )

            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df_plot.columns for col in required_cols):
                return ""

            df_plot = df_plot[list(required_cols)].copy()
            df_plot = df_plot.dropna(subset=required_cols)

            if len(df_plot) < 2:
                return ""

            mc = mpf.make_marketcolors(
                up="#26a69a",
                down="#ef5350",
                edge="inherit",
                wick={"up": "#26a69a", "down": "#ef5350"},
                volume={"up": "#26a69a", "down": "#ef5350"},
            )

            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle="-",
                gridcolor="#ecf0f1",
                facecolor="white",
                figcolor="white",
                rc={"font.size": 9},
            )

            addplot = []
            for ema, color, width_line in [
                (ema_21, "#1E88E5", 1.2),
                (ema_200, "#FF8F00", 1.4),
            ]:
                if ema is not None and hasattr(ema, "notna") and ema.notna().sum() > 5:
                    ema_aligned = ema.reindex(df_plot.index).dropna()
                    if len(ema_aligned) > 0:
                        addplot.append(mpf.make_addplot(ema_aligned, color=color, width=width_line))

            fig, axes = mpf.plot(
                df_plot,
                type="candle",
                style=s,
                title=title,
                ylabel="Preço (R$)",
                volume=False,
                addplot=addplot if addplot else None,
                figsize=(width / 80, height / 80),
                returnfig=True,
                warn_too_much_data=1000,
            )

            price_ax = None

            if isinstance(axes, (list, tuple)) and len(axes) > 0:
                price_ax = axes[0]
            elif hasattr(axes, "__len__") and len(axes) > 0:
                try:
                    price_ax = axes[0]
                except Exception:
                    price_ax = None
            elif isinstance(axes, dict):
                price_ax = axes.get("main") or axes.get("price") or axes.get("volume")
            else:
                price_ax = axes

            if price_ax is None and operation is not None:
                logger.warning("Não foi possível identificar o eixo principal para anotar níveis da operação")

            if operation is not None and price_ax is not None:
                entry_band_color = "#1E88E5"
                target_color = "#0B8043"
                stop_color = "#FF0000"
                partial_color = "#F4B400"
                guide_color = "#0066CC"

                price_levels: List[Tuple[str, float, str]] = []

                try:
                    if operation.entrada_min is not None and operation.entrada_max is not None:
                        low_entry = float(min(operation.entrada_min, operation.entrada_max))
                        high_entry = float(max(operation.entrada_min, operation.entrada_max))
                        price_levels.append(("Entrada mínima", low_entry, entry_band_color))
                        if high_entry - low_entry > 1e-6:
                            price_levels.append(("Entrada máxima", high_entry, entry_band_color))
                    elif operation.entrada_min is not None:
                        price_levels.append(("Entrada faixa", float(operation.entrada_min), entry_band_color))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.entrada is not None:
                        price_levels.append(("Entrada guia", float(operation.entrada), guide_color))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.parcial_preco is not None:
                        price_levels.append(("Saída parcial", float(operation.parcial_preco), partial_color))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.alvo is not None:
                        price_levels.append(("Alvo final", float(operation.alvo), target_color))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.stop is not None:
                        price_levels.append(("Stop loss", float(operation.stop), stop_color))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.support_level is not None:
                        price_levels.append(("Suporte", float(operation.support_level), "#0F9D58"))
                except (TypeError, ValueError):
                    pass

                try:
                    if operation.resistance_level is not None:
                        price_levels.append(("Resistência", float(operation.resistance_level), "#DB4437"))
                except (TypeError, ValueError):
                    pass

                if price_ax is not None:
                    try:
                        current_ylim = price_ax.get_ylim()
                        level_values = [level for _, level, _ in price_levels if level is not None]
                        if level_values:
                            min_level = min(level_values)
                            max_level = max(level_values)
                            lower_bound = min(current_ylim[0], min_level)
                            upper_bound = max(current_ylim[1], max_level)
                            price_ax.set_ylim(lower_bound, upper_bound)

                        entry_range = None
                        try:
                            if operation.entrada_min is not None and operation.entrada_max is not None:
                                entry_min = float(min(operation.entrada_min, operation.entrada_max))
                                entry_max = float(max(operation.entrada_min, operation.entrada_max))
                                if entry_max > entry_min:
                                    entry_range = (entry_min, entry_max)
                        except (TypeError, ValueError):
                            entry_range = None

                        if entry_range:
                            price_ax.axhspan(
                                entry_range[0],
                                entry_range[1],
                                color="#1E88E5",
                                alpha=0.05,
                                label="Faixa de entrada",
                            )

                        x_position = df_plot.index[-1]
                        for label, level, color in price_levels:
                            price_ax.axhline(y=level, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
                            pontos_text = ""
                            if label == "Alvo final" and operation.pontos_alvo:
                                pontos_text = f" ({operation.pontos_alvo:.1f} ticks)"
                            elif label == "Stop loss" and operation.pontos_stop:
                                pontos_text = f" ({operation.pontos_stop:.1f} ticks)"
                            elif label == "Saída parcial" and operation.parcial_pontos:
                                pontos_text = f" ({operation.parcial_pontos:.1f} ticks)"

                            price_ax.text(
                                x_position,
                                level,
                                f" {label}: R$ {level:.2f}{pontos_text}",
                                color=color,
                                fontsize=8,
                                va="center",
                                ha="left",
                                bbox={
                                    "facecolor": "white",
                                    "edgecolor": color,
                                    "alpha": 0.65,
                                    "boxstyle": "round,pad=0.2",
                                },
                            )
                    except Exception as axis_error:
                        logger.warning(f"Falha ao anotar níveis da operação no gráfico: {axis_error}")
            ChartGenerator._configure_date_axis(price_ax, df_plot.index)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as exc:
            logger.error(f"Erro ao gerar imagem: {exc}")
            try:
                return ChartGenerator._generate_simple_chart(df, title, operation, width, height)
            except Exception as fallback_error:
                logger.error(f"Erro no fallback simples do gráfico: {fallback_error}")
                return ""

    @staticmethod
    def _generate_simple_chart(
        df: pd.DataFrame,
        title: str,
        operation: Optional[Operation] = None,
        width: int = 800,
        height: int = 400,
    ) -> str:
        if df is None or df.empty or "close" not in df.columns:
            return ""

        df_plot = df.copy()
        if "time" in df_plot.columns:
            df_plot["time"] = pd.to_datetime(df_plot["time"], errors="coerce")
        elif "time_str" in df_plot.columns:
            df_plot["time"] = pd.to_datetime(df_plot["time_str"], errors="coerce")
        else:
            df_plot["time"] = pd.to_datetime(df_plot.index, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_plot.columns:
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

        df_plot = df_plot.dropna(subset=["time", "open", "high", "low", "close"])
        if df_plot.empty:
            return ""

        times = df_plot["time"]
        opens = df_plot["open"]
        highs = df_plot["high"]
        lows = df_plot["low"]
        closes = df_plot["close"]
        ema_21 = pd.to_numeric(df_plot.get("EMA_21"), errors="coerce") if "EMA_21" in df_plot.columns else None
        ema_200 = pd.to_numeric(df_plot.get("EMA_200"), errors="coerce") if "EMA_200" in df_plot.columns else None

        if len(times) < 2:
            return ""

        fig_width = max(width / 115, 6.8)
        fig_height = max(height / 140, 4.6)

        fig, ax_price = plt.subplots(
            nrows=1,
            figsize=(fig_width, fig_height),
            dpi=140,
        )

        date_vals = mdates.date2num(times)
        bar_width = (date_vals[1] - date_vals[0]) * 0.6 if len(date_vals) > 1 else 0.6
        colors = ["#26a69a" if close >= open_ else "#ef5350" for close, open_ in zip(closes, opens)]

        ax_price.set_facecolor("#ffffff")
        ax_price.vlines(date_vals, lows, highs, color=colors, linewidth=1)
        bodies_bottom = np.minimum(opens, closes)
        bodies_height = np.maximum(np.abs(closes - opens), 0.001)
        ax_price.bar(
            date_vals,
            bodies_height,
            width=bar_width,
            bottom=bodies_bottom,
            color=colors,
            align="center",
            alpha=0.9,
            linewidth=0,
        )

        if ema_21 is not None and ema_21.notna().sum() > 5:
            ema21_clean = ema_21.reindex(df_plot.index).fillna(method="ffill")
            ax_price.plot(date_vals, ema21_clean, color="#1E88E5", linewidth=1.2, label="EMA 21")
        if ema_200 is not None and ema_200.notna().sum() > 5:
            ema200_clean = ema_200.reindex(df_plot.index).fillna(method="ffill")
            ax_price.plot(date_vals, ema200_clean, color="#FF8F00", linewidth=1.2, label="EMA 200")

        if operation is not None:
            entry_band_color = "#1E88E5"
            guide_color = "#0066CC"
            partial_color = "#F4B400"
            target_color = "#0B8043"
            stop_color = "#FF0000"

            entry_min = operation.entrada_min if operation.entrada_min is not None else operation.entrada
            entry_max = operation.entrada_max if operation.entrada_max is not None else operation.entrada
            try:
                if entry_min is not None and entry_max is not None and entry_max > entry_min:
                    ax_price.axhspan(
                        min(entry_min, entry_max),
                        max(entry_min, entry_max),
                        color=entry_band_color,
                        alpha=0.06,
                        label="Faixa de entrada",
                    )
            except (TypeError, ValueError):
                pass

            level_specs: List[Tuple[str, Optional[float], str, Optional[float]]] = []

            try:
                if entry_min is not None and entry_max is not None:
                    low_entry = float(min(entry_min, entry_max))
                    high_entry = float(max(entry_min, entry_max))
                    level_specs.append(("Entrada mínima", low_entry, entry_band_color, None))
                    if high_entry - low_entry > 1e-6:
                        level_specs.append(("Entrada máxima", high_entry, entry_band_color, None))
                elif entry_min is not None:
                    level_specs.append(("Entrada faixa", float(entry_min), entry_band_color, None))
            except (TypeError, ValueError):
                pass

            if operation.entrada is not None:
                try:
                    level_specs.append(("Entrada guia", float(operation.entrada), guide_color, None))
                except (TypeError, ValueError):
                    pass

            if operation.parcial_preco is not None:
                try:
                    level_specs.append(("Saída parcial", float(operation.parcial_preco), partial_color, operation.parcial_pontos))
                except (TypeError, ValueError):
                    pass

            if operation.alvo is not None:
                try:
                    level_specs.append(("Alvo final", float(operation.alvo), target_color, operation.pontos_alvo))
                except (TypeError, ValueError):
                    pass

            if operation.stop is not None:
                try:
                    level_specs.append(("Stop loss", float(operation.stop), stop_color, operation.pontos_stop))
                except (TypeError, ValueError):
                    pass

            if operation.support_level is not None:
                try:
                    level_specs.append(("Suporte", float(operation.support_level), "#0F9D58", None))
                except (TypeError, ValueError):
                    pass

            if operation.resistance_level is not None:
                try:
                    level_specs.append(("Resistência", float(operation.resistance_level), "#DB4437", None))
                except (TypeError, ValueError):
                    pass

            for label, price_level, color, pontos in level_specs:
                if price_level is None:
                    continue
                try:
                    ax_price.axhline(price_level, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
                    pontos_text = f" ({pontos:.1f} ticks)" if pontos else ""
                    ax_price.text(
                        date_vals[-1],
                        price_level,
                        f" {label}: R$ {price_level:.2f}{pontos_text}",
                        color=color,
                        fontsize=8,
                        va="center",
                        ha="left",
                        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.7, "pad": 0.2},
                    )
                except (TypeError, ValueError):
                    continue
        if len(date_vals) > 1:
            margin = (date_vals[-1] - date_vals[0]) * 0.03
            ax_price.set_xlim(date_vals[0] - margin, date_vals[-1] + margin)

        ax_price.set_title(title)
        ax_price.set_ylabel("Preço (R$)")
        ax_price.grid(True, linestyle="--", alpha=0.2)
        handles, labels = ax_price.get_legend_handles_labels()
        if handles:
            ax_price.legend(handles, labels, loc="upper left", fontsize=8)

        ax_price.xaxis_date()
        ChartGenerator._configure_date_axis(ax_price, times)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
