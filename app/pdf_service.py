from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

from flask import render_template

from .charts import ChartGenerator
from .pdf_models import (
    AssetCard,
    BrandingAssets,
    ComplianceData,
    EditorialContent,
    PdfInput,
    PerformanceStats,
    PortfolioSection,
    ReportIdentity,
    SeriesData,
)
from .utils import logger


def _format_date_str(raw: Optional[str]) -> str:
    try:
        if not raw:
            return ""
        return datetime.fromisoformat(str(raw)).strftime("%d/%m/%Y")
    except Exception:
        return str(raw or "")


def _safe_float(value: Optional[float], default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def default_weasyprint_engine(base_url: Optional[str] = None):
    """Configura motor WeasyPrint, se disponivel, respeitando base_url."""
    try:
        from weasyprint import HTML
    except Exception:
        return None

    def _engine(html: str) -> bytes:
        import os
        # Garante que o base_url seja sempre a raiz do projeto (onde está a pasta static)
        root_dir = base_url or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        css_path = os.path.join(root_dir, "static", "pdf", "pdf.css")
        logger.info(f"[PDF] CSS esperado em: {css_path} (existe: {os.path.exists(css_path)})")
        return HTML(string=html, base_url=root_dir).write_pdf()

    return _engine


def build_pdf_payload_from_weekly_portfolio(
    portfolio_record: Dict,
    *,
    branding: Optional[BrandingAssets] = None,
    editorial_text: Optional[str] = None,
    performance_data: Optional[Dict] = None,
    series_data: Optional[Dict] = None,
    compliance_data: Optional[Dict] = None,
    disclaimer_text: Optional[str] = None,
) -> PdfInput:
    branding_assets = branding or BrandingAssets(**(portfolio_record.get("branding") or {}))

    start_date = _format_date_str(portfolio_record.get("start_date"))
    end_date = _format_date_str(portfolio_record.get("end_date"))

    identity = ReportIdentity(
        title="Carteira Semanal",
        period_start=start_date,
        period_end=end_date,
        strategy_name="Carteira Semanal",
        report_type="semanal",
        branding=branding_assets,
    )

    editorial_body = editorial_text or portfolio_record.get("editorial_text") or (
        "Bem-vindos a Carteira Semanal da Castling.me! Operacoes curtas, disciplina rigida "
        "e movimentos rapidos para buscar vantagem diante do mercado."
    )
    editorial = EditorialContent(
        headline="Bem-vindos a Carteira Semanal da Castling.me!",
        body=editorial_body,
        assets_count=len(portfolio_record.get("assets", []) or []),
        history_since="janeiro de 2019",
        benchmark="IBOV",
        link="https://www.castling.me",
    )

    assets_cards = []
    for asset in portfolio_record.get("assets", []) or []:
        assets_cards.append(
            AssetCard(
                symbol=str(asset.get("symbol", "")).upper(),
                entrada=_safe_float(asset.get("entry"), 0.0) or 0.0,
                entrada_maxima=_safe_float(asset.get("entry_maxima")),
                entrada_minima=_safe_float(asset.get("entry_minima")),
                objetivo=_safe_float(asset.get("objective"), 0.0) or 0.0,
                stop_loss=_safe_float(asset.get("stop_loss"), 0.0) or 0.0,
                ultimo_preco=_safe_float(asset.get("ultimo_preco")),
                retorno_pct=_safe_float(asset.get("retorno_pct"), 0.0) or 0.0,
                risco_pct=_safe_float(asset.get("risco_pct"), 0.0) or 0.0,
                risco_zero_preco=_safe_float(asset.get("risco_zero_preco"), 0.0) or 0.0,
            )
        )

    weight_pct = round(100 / len(assets_cards), 2) if assets_cards else 0.0
    portfolio_section = PortfolioSection(
        assets=assets_cards,
        weights=[weight_pct for _ in assets_cards],
        entered=portfolio_record.get("entered", []) or [],
        exited=portfolio_record.get("exited", []) or [],
        technical_notes=portfolio_record.get("technical_notes", []) or [],
    )

    perf = performance_data or portfolio_record.get("performance") or {}
    performance = PerformanceStats(
        win_rate=float(perf.get("win_rate", 77.8)),
        avg_gain=float(perf.get("avg_gain", 1.65)),
        avg_loss=float(perf.get("avg_loss", -1.59)),
        vol_annualized=float(perf.get("vol_annualized", 11.27)),
        sharpe=float(perf.get("sharpe", 3.07)),
        alpha_ibov=float(perf.get("alpha_ibov", 2.57)),
        profit_factor=float(perf.get("profit_factor", 3.65)),
        return_annualized=float(perf.get("return_annualized", 6.56)),
    )

    series_cfg = series_data or portfolio_record.get("series") or {}
    series = SeriesData(
        cumulative_castling=series_cfg.get("cumulative_castling", []) or [],
        cumulative_ibov=series_cfg.get("cumulative_ibov", []) or [],
        weekly_returns=series_cfg.get("weekly_returns", []) or [],
        weekly_labels=series_cfg.get("weekly_labels", []) or [],
        cumulative_labels=series_cfg.get("cumulative_labels", []) or [],
    )
    if not series.cumulative_castling or not series.cumulative_ibov:
        demo_labels = ["S1", "S2", "S3", "S4", "S5", "S6"]
        series.cumulative_castling = [100, 101.5, 102.2, 104.8, 106.0, 107.3]
        series.cumulative_ibov = [100, 100.7, 100.9, 102.0, 102.8, 103.1]
        series.cumulative_labels = demo_labels
    if not series.weekly_returns:
        series.weekly_returns = [0.6, -0.2, 1.1, 0.4, 0.9, -0.3]
        series.weekly_labels = series.weekly_labels or [f"S{i}" for i in range(1, len(series.weekly_returns) + 1)]

    comp_cfg = compliance_data or portfolio_record.get("compliance") or {}
    compliance = ComplianceData(
        analyst_name=comp_cfg.get("analyst_name", "Analista Castling"),
        analyst_cnpi=comp_cfg.get("analyst_cnpi", "CNPI"),
        link=comp_cfg.get("link", "https://www.castling.me"),
    )

    disclaimer_default = (
        portfolio_record.get("disclaimer_text")
        or "Este relatorio e informativo e nao constitui oferta de compra ou venda de ativos. "
        "As opinioes refletem o cenario na data de emissao e podem mudar sem aviso previo."
    )

    return PdfInput(
        identity=identity,
        editorial=editorial,
        portfolio=portfolio_section,
        performance=performance,
        series=series,
        compliance=compliance,
        disclaimer_text=disclaimer_text or disclaimer_default,
    )


class PdfReportBuilder:
    def __init__(self, engine: Optional[object] = None, base_url: Optional[str] = None):
        self.engine = engine or default_weasyprint_engine(base_url=base_url)

    def render_html(self, payload: PdfInput) -> str:
        try:
            self._attach_charts(payload)
            import os
            from pathlib import Path
            context = asdict(payload)
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            css_path = os.path.join(root_dir, "static", "pdf", "pdf.css")
            context["css_href"] = Path(css_path).resolve().as_uri()
            return render_template("pdf/document.html", **context)
        except Exception:
            logger.exception("Falha ao renderizar HTML do relatorio PDF")
            raise

    def build_pdf(self, payload: PdfInput) -> bytes:
        if self.engine is None:
            raise RuntimeError("Engine de PDF nao configurada. Configure WeasyPrint ou similar.")
        html = self.render_html(payload)
        # Log base_url usado
        import os
        base_url = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logger.info(f"[PDF] base_url usado: {base_url}")
        try:
            return self.engine(html)
        except Exception as e:
            logger.exception(f"Erro ao gerar PDF com WeasyPrint: {e}")
            raise

    def _attach_charts(self, payload: PdfInput) -> None:
        series = getattr(payload, "series", None)
        if series is None:
            return
        try:
            has_base100 = bool(getattr(series, "cumulative_castling", None) and getattr(series, "cumulative_ibov", None))
            if has_base100 and not getattr(series, "base100_chart", None):
                series.base100_chart = ChartGenerator.render_base100_comparison(
                    series.cumulative_castling,
                    series.cumulative_ibov,
                    labels=getattr(series, "cumulative_labels", None) or None,
                    width=820,
                    height=320,
                )

            has_weekly = bool(getattr(series, "weekly_returns", None))
            if has_weekly and not getattr(series, "weekly_chart", None):
                series.weekly_chart = ChartGenerator.render_weekly_returns(
                    series.weekly_returns,
                    labels=getattr(series, "weekly_labels", None) or None,
                    width=820,
                    height=280,
                )
        except Exception:
            logger.exception("Falha ao gerar graficos de performance para PDF")
