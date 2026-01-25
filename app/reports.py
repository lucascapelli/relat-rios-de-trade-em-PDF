from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import BR_TZ, INSTITUTION_NAME, INSTITUTION_TEXT
from .models import Operation
from .utils import logger


ChartPayload = Dict[str, Any]


def _fmt_price(value: Optional[float]) -> str:
    return f"R$ {value:.2f}" if value is not None else "-"


def _fmt_points(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _fmt_ratio(value: Optional[float]) -> str:
    return f"1:{value:.2f}" if value else "-"


def _parse_b3_future_expiration(symbol: str) -> Optional[str]:
    """Parse B3 future code like WING26 / WDOG26 -> 'Fevereiro/2026'."""
    if not symbol:
        return None

    match = re.match(r"^([A-Z]{2,4})([FGHJKMNQUVXZ])(\d{2})$", symbol.strip().upper())
    if not match:
        return None

    month_code = match.group(2)
    year_full = 2000 + int(match.group(3))

    month_map = {
        "F": "Janeiro",
        "G": "Fevereiro",
        "H": "Março",
        "J": "Abril",
        "K": "Maio",
        "M": "Junho",
        "N": "Julho",
        "Q": "Agosto",
        "U": "Setembro",
        "V": "Outubro",
        "X": "Novembro",
        "Z": "Dezembro",
    }
    month_name = month_map.get(month_code)
    if not month_name:
        return None
    return f"{month_name}/{year_full}"


class ReportGenerator:
    """Gera relatórios em PDF aproveitando os gráficos renderizados pelo sistema."""

    def __init__(self, reports_dir: str) -> None:
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self._log = logger.getChild("reports")

    def generate_pdf_report(self, operation: Operation, charts: List[ChartPayload]) -> str:
        timestamp = datetime.now(BR_TZ)
        filename = f"report_{operation.symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(self.reports_dir, filename)

        decoded_charts = self._decode_charts(charts)
        story = self._build_story(operation, decoded_charts, timestamp)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"Relatório {operation.symbol}",
            author=INSTITUTION_NAME,
        )

        try:
            doc.build(story, onFirstPage=self._draw_footer, onLaterPages=self._draw_footer)
            self._log.info("PDF gerado: %s", output_path)
        except Exception:
            self._log.error("Erro ao gerar PDF", exc_info=True)
            raise

        return output_path

    def generate_day_trade_pdf(self, session: Dict[str, Any]) -> str:
        """Gera PDF para uma sessão de day trade (tabela de entradas)."""
        trade_date = str(session.get("trade_date") or "")
        timestamp = datetime.now(BR_TZ)
        safe_date = trade_date.replace("/", "-") if trade_date else timestamp.strftime("%Y-%m-%d")
        filename = f"daytrade_{safe_date}_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(self.reports_dir, filename)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, leading=26, alignment=1))
        styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=14, leading=18))
        styles.add(ParagraphStyle("InfoText", parent=styles["BodyText"], fontSize=10, leading=14))

        story: List[Any] = []
        self._add_institution_block(story, styles)
        story.append(Paragraph("Relatório de Day Trade", styles["ReportTitle"]))
        story.append(Spacer(1, 6 * mm))

        tf_major = str(session.get("timeframe_major") or "-")
        tf_minor = str(session.get("timeframe_minor") or "-")
        risk_amount = session.get("risk_amount")
        risk_percent = session.get("risk_percent")
        meta = (
            f"Sessão: <b>{trade_date or '-'}</b> • Timeframes: <b>{tf_major}</b> / <b>{tf_minor}</b> • "
            f"Gerado em {timestamp.strftime('%d/%m/%Y %H:%M:%S')} (Horário de Brasília)."
        )
        story.append(Paragraph(meta, styles["InfoText"]))
        if risk_amount is not None or risk_percent is not None:
            risk_line = ""
            if risk_amount is not None:
                try:
                    risk_line += f"Risco (R$): <b>{_fmt_price(float(risk_amount))}</b>"
                except Exception:
                    risk_line += f"Risco (R$): <b>{risk_amount}</b>"
            if risk_percent is not None:
                if risk_line:
                    risk_line += " • "
                try:
                    risk_line += f"Risco (%): <b>{float(risk_percent):.2f}%</b>"
                except Exception:
                    risk_line += f"Risco (%): <b>{risk_percent}</b>"
            if risk_line:
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(risk_line, styles["InfoText"]))

        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Entradas", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))

        entries = session.get("entries") or []
        header = [
            "Ativo",
            "Dir",
            "Entrada",
            "Var. Max",
            "Alvo",
            "Stop",
            "Alvo%",
            "Stop%",
        ]
        rows: List[List[str]] = [header]
        for entry in entries:
            symbol = str(entry.get("symbol") or "-")
            direction = str(entry.get("direction") or "-")
            entry_price = entry.get("entry")
            max_var = entry.get("max_entry_variation")
            target = entry.get("target")
            stop = entry.get("stop")
            target_pct = entry.get("target_percent")
            stop_pct = entry.get("stop_percent")

            def _fmt_num(value: Any) -> str:
                try:
                    if value is None or value == "":
                        return "-"
                    return f"{float(value):.2f}"
                except Exception:
                    return str(value)

            def _fmt_pct(value: Any) -> str:
                try:
                    if value is None or value == "":
                        return "-"
                    return f"{float(value):.2f}%"
                except Exception:
                    return str(value)

            rows.append(
                [
                    symbol,
                    direction,
                    _fmt_num(entry_price),
                    _fmt_pct(max_var),
                    _fmt_num(target),
                    _fmt_num(stop),
                    _fmt_pct(target_pct),
                    _fmt_pct(stop_pct),
                ]
            )

        table = Table(rows, hAlign="LEFT", repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"Day Trade {trade_date}",
            author=INSTITUTION_NAME,
        )
        try:
            doc.build(story, onFirstPage=self._draw_footer, onLaterPages=self._draw_footer)
            self._log.info("PDF gerado: %s", output_path)
        except Exception:
            self._log.error("Erro ao gerar PDF de day trade", exc_info=True)
            raise

        return output_path

    def generate_portfolio_pdf(self, portfolio: Dict[str, Any]) -> str:
        """Gera PDF para uma carteira (tabela de ativos + análise)."""
        portfolio_type = str(portfolio.get("portfolio_type") or "CARTEIRA")
        state = str(portfolio.get("state") or "-")
        timestamp = datetime.now(BR_TZ)
        filename = f"portfolio_{portfolio_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(self.reports_dir, filename)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, leading=26, alignment=1))
        styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=14, leading=18))
        styles.add(ParagraphStyle("InfoText", parent=styles["BodyText"], fontSize=10, leading=14))

        story: List[Any] = []
        self._add_institution_block(story, styles)
        story.append(Paragraph(f"Carteira {portfolio_type}", styles["ReportTitle"]))
        story.append(Spacer(1, 6 * mm))

        start_date = portfolio.get("start_date")
        end_date = portfolio.get("end_date")
        version = portfolio.get("version")
        meta_parts = [f"Estado: <b>{state}</b>"]
        if version is not None:
            meta_parts.append(f"Versão: <b>{version}</b>")
        if start_date:
            meta_parts.append(f"Período: <b>{start_date}</b> até <b>{end_date or '-'} </b>")
        meta_parts.append(f"Gerado em {timestamp.strftime('%d/%m/%Y %H:%M:%S')} (Horário de Brasília).")
        story.append(Paragraph(" • ".join(meta_parts), styles["InfoText"]))
        story.append(Spacer(1, 6 * mm))

        story.append(Paragraph("Ativos", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))

        assets = portfolio.get("assets") or []
        header = ["Ativo", "Entrada", "Entrada Máx", "Risco Zero", "Alvo", "Stop"]
        rows: List[List[str]] = [header]

        def _fmt_num(value: Any) -> str:
            try:
                if value is None or value == "":
                    return "-"
                return f"{float(value):.2f}"
            except Exception:
                return str(value)

        for asset in assets:
            rows.append(
                [
                    str(asset.get("symbol") or "-"),
                    _fmt_num(asset.get("entry")),
                    _fmt_num(asset.get("entry_max")),
                    _fmt_num(asset.get("risk_zero")),
                    _fmt_num(asset.get("target")),
                    _fmt_num(asset.get("stop")),
                ]
            )

        table = Table(rows, hAlign="LEFT", repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)

        analytical_text = str(portfolio.get("analytical_text") or "").strip()
        if analytical_text:
            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph("Análise", styles["SectionHeading"]))
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(analytical_text, styles["InfoText"]))

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"Carteira {portfolio_type}",
            author=INSTITUTION_NAME,
        )
        try:
            doc.build(story, onFirstPage=self._draw_footer, onLaterPages=self._draw_footer)
            self._log.info("PDF gerado: %s", output_path)
        except Exception:
            self._log.error("Erro ao gerar PDF de carteira", exc_info=True)
            raise

        return output_path

    def _decode_charts(self, charts: Iterable[ChartPayload]) -> List[ChartPayload]:
        decoded: List[ChartPayload] = []
        for chart in charts:
            timeframe = str(chart.get("timeframe", "")).strip().lower()
            if not timeframe:
                self._log.warning("Chart ignorado por timeframe ausente")
                continue

            image_data = chart.get("image") or ""
            buffer: Optional[BytesIO] = None
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                try:
                    payload = image_data.split(",", 1)[1]
                    binary = base64.b64decode(payload)
                    if len(binary) > 200:
                        buffer = BytesIO(binary)
                    else:
                        self._log.warning("Chart %s ignorado: payload muito pequeno", timeframe)
                except Exception:
                    self._log.warning("Falha ao decodificar imagem do timeframe %s", timeframe, exc_info=True)
            elif isinstance(image_data, str) and image_data:
                try:
                    binary = base64.b64decode(image_data)
                    if len(binary) > 200:
                        buffer = BytesIO(binary)
                except Exception:
                    self._log.warning("Falha ao interpretar imagem crua do timeframe %s", timeframe, exc_info=True)

            decoded.append(
                {
                    "timeframe": timeframe,
                    "buffer": buffer,
                    "source": str(chart.get("source", "desconhecido")),
                    "last_time": chart.get("last_time"),
                    "last_close": chart.get("last_close"),
                }
            )

        order = {"15m": 0, "1h": 1, "1d": 2, "6m": 10, "30m": 11, "45m": 12, "4h": 13, "1w": 14}
        decoded.sort(key=lambda item: (order.get(str(item.get("timeframe", "")), 99), str(item.get("timeframe", ""))))
        return decoded

    def _build_story(
        self,
        operation: Operation,
        charts: List[ChartPayload],
        generated_at: datetime,
    ) -> List[Any]:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, leading=26, alignment=1))
        styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=14, leading=18))
        styles.add(ParagraphStyle("InfoText", parent=styles["BodyText"], fontSize=10, leading=14))

        story: List[Any] = []
        self._add_institution_block(story, styles)
        story.append(Paragraph(f"Relatório de Operação - {operation.symbol}", styles["ReportTitle"]))
        story.append(Spacer(1, 6 * mm))

        viés = "positivo" if operation.tipo.upper() == "COMPRA" else "negativo"
        resumo = (
            f"Análise do ativo <b>{operation.symbol}</b> com viés <b>{viés}</b>. "
            f"Operação do tipo <b>{operation.tipo}</b> no timeframe <b>{operation.timeframe.upper()}</b>, "
            f"gerada em {generated_at.strftime('%d/%m/%Y %H:%M:%S')} (Horário de Brasília)."
        )
        story.append(Paragraph(resumo, styles["InfoText"]))
        story.append(Spacer(1, 4 * mm))

        faixa_text = "-"
        if operation.entrada_min is not None and operation.entrada_max is not None:
            faixa_min = min(operation.entrada_min, operation.entrada_max)
            faixa_max = max(operation.entrada_min, operation.entrada_max)
            if abs(faixa_max - faixa_min) < 0.01:
                faixa_text = _fmt_price(faixa_min)
            else:
                faixa_text = f"{_fmt_price(faixa_min)} a {_fmt_price(faixa_max)}"

        details_rows = [
            ("Ticker", operation.symbol),
            ("Tipo", operation.tipo.upper()),
            ("Timeframe", operation.timeframe.upper()),
            ("Preço atual", _fmt_price(operation.preco_atual) if operation.preco_atual else "-"),
            ("Entrada sugerida", _fmt_price(operation.entrada)),
            ("Entrada mínima", _fmt_price(operation.entrada_min)),
            ("Entrada máxima", _fmt_price(operation.entrada_max)),
            ("Alvo", _fmt_price(operation.alvo)),
            ("Stop", _fmt_price(operation.stop)),
            ("Saída Parcial", _fmt_price(operation.parcial_preco)),
            ("Pontos Parciais", _fmt_points(operation.parcial_pontos)),
            ("Pontos Alvo", _fmt_points(operation.pontos_alvo)),
            ("Pontos Stop", _fmt_points(operation.pontos_stop)),
            ("Risco / Retorno", _fmt_ratio(operation.risco_retorno)),
            ("Quantidade", str(operation.quantidade)),
        ]

        table = Table(
            details_rows,
            hAlign="LEFT",
            colWidths=[55 * mm, 100 * mm],
            repeatRows=0,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f2f5")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#34495e")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 6 * mm))

        observacoes = operation.observacoes or "Sem observações adicionais."
        expiration = _parse_b3_future_expiration(operation.symbol)
        if expiration:
            story.append(Paragraph("Informações do Contrato", styles["SectionHeading"]))
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(f"Vencimento: <b>{expiration}</b>", styles["InfoText"]))
            story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Indicadores", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("MME 21 e MME 200 (Médias Móveis Exponenciais).", styles["InfoText"]))
        story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Contexto", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))
        if operation.narrative_text:
            story.append(Paragraph(operation.narrative_text, styles["InfoText"]))
        else:
            story.append(Paragraph("Contexto automático indisponível.", styles["InfoText"]))
        if operation.support_level is not None or operation.resistance_level is not None:
            story.append(Spacer(1, 2 * mm))
            story.append(
                Paragraph(
                    f"Suporte: <b>{_fmt_price(operation.support_level)}</b> | "
                    f"Resistência: <b>{_fmt_price(operation.resistance_level)}</b>",
                    styles["InfoText"],
                )
            )
        story.append(Spacer(1, 6 * mm))

        story.append(Paragraph("Observações", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(observacoes, styles["InfoText"]))
        story.append(Spacer(1, 6 * mm))

        valid_charts = [chart for chart in charts if chart.get("buffer")]
        if valid_charts:
            story.append(Paragraph("Gráficos", styles["SectionHeading"]))
            story.append(Spacer(1, 4 * mm))
            timeframe_labels = {
                "1m": "1 Minuto",
                "5m": "5 Minutos",
                "6m": "6 Minutos",
                "15m": "15 Minutos",
                "30m": "30 Minutos",
                "45m": "45 Minutos",
                "1h": "60 Minutos",
                "4h": "4 Horas",
                "1d": "Diário",
                "1w": "Semanal",
            }

            def _fmt_source(value: str) -> str:
                raw = (value or "").strip().lower()
                if not raw or raw == "desconhecido":
                    return "desconhecido"
                if raw.startswith("fallback"):
                    return "fallback"
                return raw

            rendered = 0
            for idx, chart in enumerate(valid_charts):
                timeframe = str(chart.get("timeframe", "")).strip().lower()
                buffer = chart.get("buffer")
                source = str(chart.get("source", "desconhecido"))
                last_time = chart.get("last_time")
                last_close = chart.get("last_close")
                try:
                    buffer.seek(0)
                    reader = ImageReader(buffer)
                    width_px, height_px = reader.getSize()
                    width_mm = 160 * mm
                    aspect = height_px / float(width_px) if width_px else 0
                    height_mm = width_mm * aspect if aspect else 90 * mm
                    buffer.seek(0)
                    img = Image(buffer, width=width_mm, height=height_mm)
                except Exception:
                    self._log.warning("Falha ao preparar imagem do timeframe %s", timeframe, exc_info=True)
                    continue

                img.hAlign = "CENTER"

                label = timeframe_labels.get(timeframe, timeframe.upper())
                update_line = ""
                if last_time:
                    update_line = f"Atualizado: {last_time}"
                elif last_close is not None:
                    update_line = f"Último preço: {_fmt_price(float(last_close))}"
                chart_block: List[Any] = [
                    Paragraph(f"{operation.symbol} - {label}", styles["InfoText"]),
                    Spacer(1, 2 * mm),
                    img,
                    Spacer(1, 1 * mm),
                    Paragraph(f"Fonte: {_fmt_source(source)}", styles["InfoText"]),
                ]

                if update_line:
                    chart_block.append(Paragraph(update_line, styles["InfoText"]))

                story.append(KeepTogether(chart_block))
                story.append(Spacer(1, 6 * mm))

                rendered += 1
                if rendered % 2 == 0 and rendered < len(valid_charts):
                    story.append(PageBreak())
        else:
            story.append(Paragraph("Nenhuma imagem de gráfico foi fornecida.", styles["InfoText"]))

        return story

    def _draw_footer(self, canvas, doc) -> None:
        canvas.saveState()
        page_width, _ = doc.pagesize
        canvas.setFont("Helvetica", 9)
        canvas.drawString(18 * mm, 12 * mm, INSTITUTION_NAME)
        canvas.drawRightString(page_width - 18 * mm, 12 * mm, f"Página {doc.page}")
        canvas.restoreState()

    def _add_institution_block(self, story: List[Any], styles) -> None:
        name = (INSTITUTION_NAME or "").strip()
        text = (INSTITUTION_TEXT or "").strip()

        if not name and not text:
            return

        if name:
            story.append(Paragraph(f"<b>{name}</b>", styles["InfoText"]))
        if text:
            story.append(Paragraph(text, styles["InfoText"]))
        story.append(Spacer(1, 4 * mm))


__all__ = ["ReportGenerator"]
