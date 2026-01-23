"""PDF report generation for trading operations."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from weasyprint import HTML

from .config import BR_TZ, PLACEHOLDER_IMAGE_DATA_URL
from .models import Operation


class ReportGenerator:
    def __init__(self, reports_dir: str) -> None:
        self.reports_dir = reports_dir

    def generate_pdf_report(self, operation: Operation, charts: List[Dict[str, Any]]) -> str:
        html_content = self._create_html_content(operation, charts)
        filename = f"report_{operation.symbol}_{int(time.time())}.pdf"
        path = os.path.join(self.reports_dir, filename)
        HTML(string=html_content).write_pdf(path)
        return path

    def _create_html_content(self, operation: Operation, charts: List[Dict[str, Any]]) -> str:
        def normalize_tf(tf: Optional[str]) -> str:
            if not tf:
                return ""
            tf_norm = tf.lower().strip()
            replacements = {
                "15min": "15m",
                "15": "15m",
                "60min": "1h",
                "60m": "1h",
                "diario": "1d",
                "daily": "1d",
                "dia": "1d",
                "weekly": "1w",
                "semanal": "1w",
            }
            return replacements.get(tf_norm, tf_norm)

        def label_for(tf: str) -> str:
            mapping = {
                "15m": "15M",
                "1h": "60M",
                "1d": "DIÁRIO",
                "1w": "SEMANAL",
            }
            return mapping.get(tf, tf.upper())

        def fmt_price(value: Optional[float]) -> str:
            if value is None:
                return "-"
            return f"R$ {value:.2f}"

        def fmt_ticks(value: Optional[float]) -> str:
            if value is None:
                return "-"
            return f"{value:.1f} ticks"

        def fmt_tick_size(value: Optional[float]) -> str:
            if value is None:
                return "-"
            return f"{value:.2f} pontos"

        charts_map: Dict[str, Dict[str, Any]] = {}
        for chart in charts:
            tf_key = normalize_tf(chart.get("timeframe"))
            if tf_key and chart.get("image"):
                charts_map[tf_key] = chart

        def chart_for(tf: str) -> Dict[str, Any]:
            key = normalize_tf(tf)
            return charts_map.get(key, {})

        chart_primary = chart_for("15m") or chart_for(operation.timeframe)
        chart_60 = chart_for("1h")
        chart_daily = chart_for("1d")
        chart_weekly = chart_for("1w")

        faixa_text = "-"
        faixa_min = operation.entrada_min
        faixa_max = operation.entrada_max
        if faixa_min is not None and faixa_max is not None:
            if abs(faixa_max - faixa_min) < 1e-6:
                faixa_text = fmt_price(faixa_min)
            else:
                faixa_text = f"{fmt_price(min(faixa_min, faixa_max))} a {fmt_price(max(faixa_min, faixa_max))}"

        direction_word = "compra" if operation.tipo == "COMPRA" else "venda"
        direction_word_upper = direction_word.upper()
        bias_word = "positivo" if operation.tipo == "COMPRA" else "negativo"
        main_label = label_for(normalize_tf("15m")) if chart_primary else label_for(operation.timeframe)
        summary_text = operation.narrative_text or (
            f"{operation.symbol} opera com viés técnico {bias_word} no {main_label}, favorecendo {direction_word_upper} entre {faixa_text}."
        )

        resumo_dt = datetime.now(BR_TZ).strftime("%d/%m/%Y %H:%M")

        def fmt_points(value: Optional[float]) -> str:
            if value is None:
                return "-"
            return f"{abs(value):.1f}"

        def colored_price(value: Optional[float], css_class: str) -> str:
            price_text = fmt_price(value)
            return f"<span class=\"{css_class}\">{price_text}</span>" if value is not None else price_text

        risk_reward_text = "-"
        if operation.risco_retorno:
            rr_value = abs(operation.risco_retorno)
            risk_reward_text = f"1:{rr_value:.2f}"

        detail_rows = [
            ("Ticker", operation.symbol),
            ("Tipo", operation.tipo.upper()),
            ("Timeframe base", operation.timeframe.upper()),
            ("Entrada guia", colored_price(operation.entrada, "text-entry")),
            ("Entrada mínima", colored_price(faixa_min, "text-entry")),
            ("Entrada máxima", colored_price(faixa_max, "text-entry")),
            ("Saída parcial", colored_price(operation.parcial_preco, "text-target")),
            ("Parcial pts", fmt_points(operation.parcial_pontos)),
            ("Alvo final", colored_price(operation.alvo, "text-target")),
            ("Alvo pts", fmt_points(operation.pontos_alvo)),
            ("Stop loss", colored_price(operation.stop, "text-stop")),
            ("Stop loss pts", fmt_points(operation.pontos_stop)),
            ("Quantidade", str(operation.quantidade)),
            ("Tick size", fmt_tick_size(operation.tick_size)),
            ("Risco / Retorno", risk_reward_text),
        ]

        details_table_rows = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in detail_rows)

        metrics = [
            ("Status", operation.status.title()),
            ("Preço atual", fmt_price(operation.preco_atual)),
            ("Suporte", fmt_price(operation.support_level)),
            ("Resistência", fmt_price(operation.resistance_level)),
            ("Alvo pts", fmt_points(operation.pontos_alvo)),
            ("Stop pts", fmt_points(operation.pontos_stop)),
            ("Gerado", resumo_dt),
        ]

        def chart_card(chart_data: Dict[str, Any], title: str, size_class: str) -> str:
            if not chart_data:
                return f"""
                <div class="chart-box {size_class} empty">
                    <div class="chart-title">{title}</div>
                    <div class="chart-placeholder">Nenhum dado disponível</div>
                </div>
                """
            return f"""
            <div class="chart-box {size_class}">
                <div class="chart-title">{title}</div>
                <img src="{chart_data.get('image', PLACEHOLDER_IMAGE_DATA_URL)}" alt="{title}">
            </div>
            """

        chart_15_html = chart_card(chart_primary, f"{operation.symbol} · {label_for('15m')}", "chart-15")
        chart_60_html = chart_card(chart_60, f"{operation.symbol} · {label_for('1h')}", "chart-60")
        chart_daily_html = chart_card(chart_daily, f"{operation.symbol} · {label_for('1d')}", "chart-1d")
        chart_weekly_html = chart_card(chart_weekly, f"{operation.symbol} · {label_for('1w')}", "chart-1w")

        observations_text = operation.observacoes.strip() if operation.observacoes else "Sem observações adicionais."

        metrics_html = "".join(
            f"<div class=\"metric-item\"><strong>{label}</strong><span>{value}</span></div>"
            for label, value in metrics
        )

        charts_grid_html = f"{chart_15_html}{chart_60_html}{chart_daily_html}{chart_weekly_html}"

        return f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <title>Boletim de Execução - {operation.symbol}</title>
            <style>
                @page {{
                    size: A4;
                    margin: 18mm;
                }}
                :root {{
                    --color-entry: #1565C0;
                    --color-partial: #17A589;
                    --color-target: #2ECC71;
                    --color-stop: #E74C3C;
                }}
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    background: #f5f7fb;
                    color: #243042;
                    font-size: 10.5px;
                    line-height: 1.4;
                }}
                .wrapper {{
                    background: #fff;
                    border-radius: 12px;
                    padding: 18px 22px;
                    box-shadow: 0 8px 24px rgba(15, 35, 95, 0.12);
                }}
                .header {{
                    border-left: 4px solid var(--color-target);
                    padding-left: 14px;
                    margin-bottom: 14px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 20px;
                    letter-spacing: 0.4px;
                }}
                .header span {{
                    display: block;
                    font-size: 11px;
                    color: #5f6b7c;
                    margin-top: 3px;
                }}
                .comment {{
                    font-size: 12.5px;
                    font-weight: 600;
                    margin: 8px 0 0 0;
                    color: #1f2d3d;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: 1.15fr 0.85fr;
                    gap: 12px;
                    margin-bottom: 14px;
                    align-items: start;
                }}
                .details-card {{
                    background: linear-gradient(180deg, #fefaf3 0%, #fdf4d7 100%);
                    border: 1px solid #f2c97d;
                    border-radius: 10px;
                    padding: 12px 14px;
                }}
                .details-card h2 {{
                    margin: 0 0 8px 0;
                    text-transform: uppercase;
                    font-size: 13px;
                    letter-spacing: 0.5px;
                    color: #9c6b06;
                }}
                .details-card table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 12px;
                }}
                .details-card td {{
                    padding: 4px 3px;
                    border-bottom: 1px dashed rgba(156, 107, 6, 0.25);
                }}
                .details-card tr:last-child td {{
                    border-bottom: none;
                }}
                .details-card td:first-child {{
                    font-weight: 600;
                    color: #825406;
                    width: 42%;
                }}
                .details-card td:last-child {{
                    font-weight: 600;
                    color: #1f2733;
                    text-align: left;
                }}
                .metrics-card {{
                    background: #f1f4fb;
                    border: 1px solid #c9d4ee;
                    border-radius: 10px;
                    padding: 12px;
                }}
                .metrics-card h2 {{
                    margin: 0 0 6px 0;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    color: #2c3e55;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 6px;
                }}
                .metric-item {{
                    background: #fff;
                    border: 1px solid #d7e0f2;
                    border-radius: 6px;
                    padding: 6px 8px;
                    font-size: 10.5px;
                }}
                .metric-item strong {{
                    display: block;
                    color: #374868;
                    margin-bottom: 2px;
                    letter-spacing: 0.3px;
                    text-transform: uppercase;
                    font-size: 10px;
                }}
                .metric-item span {{
                    color: #1f2733;
                    font-weight: 600;
                }}
                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 10px;
                    margin-bottom: 14px;
                }}
                .chart-box {{
                    background: #f8fafc;
                    border: 1px solid #d9e3f5;
                    border-radius: 10px;
                    padding: 10px 12px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    min-height: 220px;
                }}
                .chart-title {{
                    font-size: 12px;
                    font-weight: 700;
                    color: #1f2a3c;
                    margin-bottom: 6px;
                }}
                .chart-box img {{
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                    border: 1px solid #cbd5e1;
                    background: #fff;
                    object-fit: contain;
                }}
                .chart-box.empty {{
                    min-height: 160px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #8b98ab;
                    font-size: 12px;
                    border: 1px dashed #cbd5e1;
                }}
                .text-entry {{
                    color: #0066CC;
                    font-weight: 600;
                }}
                .text-target {{
                    color: #00AA00;
                    font-weight: 600;
                }}
                .text-stop {{
                    color: #FF0000;
                    font-weight: 600;
                }}
                .notes {{
                    font-size: 11px;
                    background: #f4f7ff;
                    border: 1px solid #dbe5ff;
                    border-radius: 9px;
                    padding: 10px 12px;
                }}
                .notes strong {{
                    display: block;
                    margin-bottom: 4px;
                    color: #1f2a3c;
                }}
                footer {{
                    margin-top: 10px;
                    text-align: right;
                    font-size: 9.5px;
                    color: #7b889c;
                }}
            </style>
        </head>
        <body>
            <div class="wrapper">
                <div class="header">
                    <h1>Boletim de Execução · {operation.symbol}</h1>
                    <span>{datetime.now(BR_TZ).strftime('%d/%m/%Y %H:%M:%S')}</span>
                    <p class="comment">{summary_text}</p>
                </div>
                <div class="summary-grid">
                    <div class="details-card">
                        <h2>Detalhes da Operação</h2>
                        <table>{details_table_rows}</table>
                    </div>
                    <div class="metrics-card">
                        <h2>Contexto Técnico</h2>
                        <div class="metrics-grid">{metrics_html}</div>
                    </div>
                </div>
                <div class="charts-grid">{charts_grid_html}</div>
                <div class="notes">
                    <strong>Observações</strong>
                    <span>{observations_text}</span>
                </div>
                <footer>Relatório gerado pelo Sistema de Execução • {datetime.now(BR_TZ).strftime('%d/%m/%Y %H:%M:%S')}</footer>
            </div>
        </body>
        </html>
        """
