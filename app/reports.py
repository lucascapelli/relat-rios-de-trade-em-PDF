from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
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

# Cores do layout Castling
DARK_HEADER = colors.HexColor("#1e2530")  
GOLD_ACCENT = colors.HexColor("#c5a065")
TEXT_DARK = colors.HexColor("#1e2530")
TEXT_LIGHT = colors.white


def _fmt_price(value: Optional[float]) -> str:
    return f"R$ {value:.2f}" if value is not None else "-"


def _fmt_points(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _fmt_ratio(value: Optional[float]) -> str:
    return f"1:{value:.2f}" if value else "-"


class ModernReportGenerator:
    """Gerador de relatórios em PDF com layout estilo Castling.me."""

    def __init__(self, reports_dir: str, logo_path: Optional[str] = None) -> None:
        self.reports_dir = reports_dir
        self.logo_path = logo_path
        os.makedirs(self.reports_dir, exist_ok=True)
        self._log = logger.getChild("reports")

    def _draw_header_footer(self, c: canvas.Canvas, doc: SimpleDocTemplate) -> None:
        """Callback para desenhar header e footer na página."""
        c.saveState()
        page_width, page_height = doc.pagesize

        # --- HEADER ---
        header_height = 40 * mm
        header_y = page_height - header_height
        
        # Fundo header
        c.setFillColor(DARK_HEADER)
        c.rect(0, header_y, page_width, header_height, fill=1, stroke=0)
        
        # Linha dourada inferior
        c.setFillColor(GOLD_ACCENT)
        c.rect(0, header_y, page_width, 1 * mm, fill=1, stroke=0)
        
        # Título esquerda
        # O titulo do report é setado no doc.title
        title_text = doc.title or "Relatório"
        date_text = datetime.now(BR_TZ).strftime("%d/%m/%Y")
        
        c.setFont("Helvetica", 24)
        c.setFillColor(colors.white) # ou cinza claro conforme imagem
        c.drawString(15 * mm, header_y + 20 * mm, title_text)
        
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.gray)
        c.drawString(15 * mm, header_y + 10 * mm, date_text)
        
        # Logo direita
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                # Caixa separadora dourada vertical (opcional, estilo da imagem)
                c.setStrokeColor(GOLD_ACCENT)
                c.setLineWidth(0.5)
                c.line(page_width - 40*mm, page_height, page_width - 40*mm, header_y)
                
                logo_size = 20 * mm
                c.drawImage(
                    self.logo_path,
                    page_width - 32 * mm,
                    header_y + (header_height - logo_size) / 2,
                    width=logo_size,
                    height=logo_size,
                    mask="auto",
                    preserveAspectRatio=True
                )
            except Exception:
                pass


        # --- FOOTER ---
        footer_height = 10 * mm
        
        # Fundo Footer
        c.setFillColor(DARK_HEADER)
        c.rect(0, 0, page_width, footer_height, fill=1, stroke=0)
        
        # Linha dourada superior
        c.setFillColor(GOLD_ACCENT)
        c.rect(0, footer_height, page_width, 0.5 * mm, fill=1, stroke=0)
        
        # Link
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.white)
        c.drawCentredString(page_width / 2, 3.5 * mm, "www.castling.me")
        
        c.restoreState()

    def _draw_institutional_bg(self, c: canvas.Canvas, doc: SimpleDocTemplate) -> None:
        """Background da página institucional (Pág 2)."""
        c.saveState()
        w, h = doc.pagesize
        
        # Fundo escuro total
        c.setFillColor(DARK_HEADER)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        
        # Footer igual
        footer_h = 10 * mm
        c.setFillColor(GOLD_ACCENT)
        c.rect(0, footer_h, w, 0.5 * mm, fill=1, stroke=0)
        
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.white)
        c.drawCentredString(w / 2, 3.5 * mm, "www.castling.me")

        # Badge/Seal no canto inferior direito (opcional conforme imagem)
        # Vamos desenhar um círculo dourado fake se não houver imagem
        c.setFillColor(GOLD_ACCENT)
        # c.circle(w - 25*mm, 25*mm, 12*mm, fill=1, stroke=0) 
        
        c.restoreState()

    def generate_day_trade_pdf(
        self, session: Dict[str, Any], charts: Optional[List[ChartPayload]] = None
    ) -> str:
        """Gera PDF day trade layout Castling."""
        trade_date = str(session.get("trade_date") or datetime.now(BR_TZ).strftime("%d/%m/%Y"))
        symbol = str(session.get("symbol", "")).upper() or "TRADE"
        timestamp = datetime.now(BR_TZ)
        safe_date = trade_date.replace("/", "-")
        filename = f"castling_{symbol}_{safe_date}_{timestamp.strftime('%H%M%S')}.pdf"
        output_path = os.path.join(self.reports_dir, filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=50 * mm, 
            bottomMargin=15 * mm,
            title=f"{symbol} Day Trade",
            author=INSTITUTION_NAME,
        )

        styles = self._get_styles()
        story: List[Any] = []

        # -- PÁGINA 1
        
        # Intro
        story.append(Paragraph("Texto feito pela ia referente ao trade...", styles["Intro"]))
        story.append(Spacer(1, 10 * mm))

        # Layout: Tabela Detalhes (Col 1) | Gráfico Principal (Col 2)
        details_table = self._create_details_table(session, styles)
        
        decoded = self._decode_charts(charts or [])
        # Procurar 15m
        main_chart_data = next((c for c in decoded if "15m" in c["timeframe"]), None)
        # Se não tiver, pega o primeiro
        if not main_chart_data and decoded:
            main_chart_data = decoded[0]
            
        col2_content = []
        if main_chart_data:
            img = self._create_chart_image(main_chart_data, 110*mm, 70*mm)
            if img:
                col2_content.append(Paragraph("Velas de 15 minutos", styles["ChartLabel"]))
                col2_content.append(img)
        
        # Tabela Container
        # Se não tiver gráfico para col 2, a tabela ocupa tudo?
        # Layout fixo 2 colunas
        layout_data = [[details_table, col2_content if col2_content else ""]]
        layout = Table(layout_data, colWidths=[65*mm, 115*mm], vAlign="TOP")
        layout.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ]))
        story.append(layout)
        story.append(Spacer(1, 8 * mm))

        # Stats strip
        stats_data = [
            ["Vencimento", "Fechamento", "Ajuste", "Variação", "Mínima", "Máxima", "Volume Financeiro"],
            ["Fevereiro", "-", "-", "N/A", "-", "-", "-"]
        ]
        stats_t = Table(stats_data, colWidths=[25*mm]*7)
        stats_t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME", (0,1), (-1,1), "Helvetica"),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("TEXTCOLOR", (0,0), (-1,0), DARK_HEADER),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))
        story.append(stats_t)
        story.append(Spacer(1, 10 * mm))

        # Secondary Charts (Diário + 60m side by side, Weekly below)
        # Filter used charts
        used_charts = [main_chart_data] if main_chart_data else []
        remaining = [c for c in decoded if c not in used_charts]
        
        # Mapeando por prioridade
        chart_d = next((c for c in remaining if "1d" in c["timeframe"]), None)
        chart_h = next((c for c in remaining if "60m" in c["timeframe"] or "1h" in c["timeframe"]), None)
        chart_w = next((c for c in remaining if "1w" in c["timeframe"]), None)
        
        # Row 1: Daily + Hourly
        row1_cells = []
        if chart_d:
            img = self._create_chart_image(chart_d, 85*mm, 50*mm)
            if img: row1_cells.append([Paragraph("Velas diárias", styles["ChartLabel"]), img])
            else: row1_cells.append("")
        else: row1_cells.append("")
            
        if chart_h:
            img = self._create_chart_image(chart_h, 85*mm, 50*mm)
            if img: row1_cells.append([Paragraph("Velas de 60 minutos", styles["ChartLabel"]), img])
            else: row1_cells.append("")
        else:
             row1_cells.append("")
             
        if any(row1_cells):
            t_charts1 = Table([row1_cells], colWidths=[90*mm, 90*mm], vAlign="TOP")
            story.append(t_charts1)
            story.append(Spacer(1, 5 * mm))
            
        # Row 2: Weekly
        if chart_w:
            img = self._create_chart_image(chart_w, 85*mm, 50*mm)
            if img:
                story.append(Paragraph("Velas semanais", styles["ChartLabel"]))
                story.append(img)


        story.append(PageBreak())

        # -- PÁGINA 2
        # Título
        story.append(Paragraph("Quem somos?", styles["InverseTitle"]))
        story.append(Spacer(1, 5 * mm))
        
        # Texto
        txt = """No jogo de xadrez, o termo "castling" (roque) se refere a uma jogada que permite que o rei e a torre troquem de lugar no tabuleiro. Esta jogada é projetada para proteger o rei e também pode ser usada como uma manobra estratégica para melhorar a posição da torre.<br/><br/>No mundo dos investimentos, o conceito de "castling" também pode ser aplicado à ideia de proteger seus ativos e fazer movimentos estratégicos para melhorar a carteira geral."""
        story.append(Paragraph(txt, styles["InverseBody"]))
        story.append(Spacer(1, 30 * mm))
        
        # Brand area (Logo image + Text)
        # Se tiver imagem de peça de xadrez grande, seria aqui.
        # Vamos colocar Texto Brand
        story.append(Paragraph("Castling.me", styles["InverseBrand"]))
        story.append(Paragraph("Defesa e ataque em um só movimento.", styles["InverseTagline"]))
        
        # Espaço até o fim para o Disclaimer
        story.append(Spacer(1, 40 * mm))
        
        story.append(Paragraph("Disclaimer", styles["InverseDisclaimerTitle"]))
        disc = """Este relatório de análise foi elaborado pela Castling LTDA (Castling), de acordo com todas as exigências previstas na Resolução CVM nº 20/2021. As informações contidas neste relatório são consideradas válidas na data de sua divulgação e foram obtidas de fontes públicas. A Castling não se responsabiliza por qualquer decisão tomada pelo cliente com base no presente relatório. O(s) signatário(s) deste relatório declara(m) que as recomendações refletem única e exclusivamente suas análises e opiniões pessoais, que foram produzidas de forma independente, inclusive em relação à Castling e que estão sujeitas a modificações sem aviso prévio."""
        story.append(Paragraph(disc, styles["InverseDisclaimerText"]))

        # Build
        doc.build(
            story,
            onFirstPage=self._draw_header_footer,
            onLaterPages=self._draw_institutional_bg
        )
        return output_path

    def _create_details_table(self, session: Dict[str, Any], styles: Any) -> Table:
        # Extrair dados
        ticker = str(session.get("symbol") or "WDOG26").upper()
        entries = session.get("entries", [])
        entry_val = entries[0]["entry"] if entries else session.get("entry_price")
        
        rows = [
            ["DETALHES DA OPERAÇÃO", ""],
            ["TICKER", ticker],
            ["ENTRADA", _fmt_price(float(entry_val)) if entry_val else "-"],
            ["ENTRADA MÍNIMA", _fmt_price(session.get("min_entry"))],
            ["ENTRADA MÁXIMA", _fmt_price(session.get("max_entry"))],
            ["SAÍDA PARCIAL", _fmt_price(session.get("partial_exit"))],
            ["PARCIAL pts", _fmt_points(session.get("partial_points"))],
            ["ALVO", _fmt_price(session.get("target"))],
            ["ALVO pts", _fmt_points(session.get("target_points"))],
            ["STOP LOSS", _fmt_price(session.get("stop_loss"))],
            ["STOP LOSS pts", _fmt_points(session.get("stop_loss_points"))],
        ]
        
        t = Table(rows, colWidths=[35*mm, 30*mm], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("SPAN", (0,0), (1,0)),
            ("BACKGROUND", (0,0), (1,0), GOLD_ACCENT),
            ("TEXTCOLOR", (0,0), (1,0), DARK_HEADER),
            ("FONTNAME", (0,0), (1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (1,0), "CENTER"),
            
            ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (1,1), (1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,1), (-1,-1), colors.white),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        return t

    def _create_chart_image(self, chart: ChartPayload, w: float, h: float) -> Optional[Image]:
        buf = chart.get("buffer")
        if not buf: return None
        try:
            buf.seek(0)
            return Image(buf, width=w, height=h)
        except:
            return None

    def _get_styles(self) -> Dict[str, ParagraphStyle]:
        s = getSampleStyleSheet()
        
        s.add(ParagraphStyle(
            "Intro", parent=s["Normal"],
            fontName="Helvetica-Oblique", fontSize=11, textColor=TEXT_DARK,
            spaceAfter=6
        ))
        
        s.add(ParagraphStyle(
            "ChartLabel", parent=s["Normal"],
            fontName="Helvetica", fontSize=8, textColor=colors.gray,
            alignment=1, spaceAfter=2
        ))
        
        # Inverse styles (Page 2)
        s.add(ParagraphStyle(
            "InverseTitle", parent=s["Heading1"],
            fontName="Helvetica-Bold", fontSize=26, textColor=GOLD_ACCENT,
            spaceAfter=10
        ))
        s.add(ParagraphStyle(
            "InverseBody", parent=s["Normal"],
            fontName="Helvetica", fontSize=11, textColor=colors.white,
            leading=15, alignment=4
        ))
        s.add(ParagraphStyle(
            "InverseBrand", parent=s["Heading1"],
            fontName="Helvetica", fontSize=24, textColor=GOLD_ACCENT,
            alignment=1, spaceAfter=4
        ))
        s.add(ParagraphStyle(
            "InverseTagline", parent=s["Normal"],
            fontName="Helvetica-Oblique", fontSize=10, textColor=colors.white,
            alignment=1, spaceAfter=10
        ))
        s.add(ParagraphStyle(
            "InverseDisclaimerTitle", parent=s["Normal"],
            fontName="Helvetica-Bold", fontSize=9, textColor=GOLD_ACCENT,
            alignment=1, spaceAfter=2
        ))
        s.add(ParagraphStyle(
            "InverseDisclaimerText", parent=s["Normal"],
            fontName="Helvetica", fontSize=6, textColor=colors.white,
            alignment=4, leading=8
        ))
        return s

    def _decode_charts(self, charts: Iterable[ChartPayload]) -> List[ChartPayload]:
        decoded = []
        for c in charts:
            tf = str(c.get("timeframe", "")).strip().lower()
            if not tf: continue
            
            img = c.get("image")
            buf = None
            if isinstance(img, str) and img:
                try:
                    payload = img.split(",", 1)[1] if "," in img else img
                    binary = base64.b64decode(payload)
                    if len(binary) > 100:
                        buf = BytesIO(binary)
                except:
                    pass
            decoded.append({"timeframe": tf, "buffer": buf, "symbol": c.get("symbol")})
        return decoded
        
    def generate_pdf_report(self, operation: Operation, charts: List[ChartPayload]) -> str:
        # Wrapper
        session = {
            "symbol": operation.symbol,
            "trade_date": datetime.now(BR_TZ).strftime("%d/%m/%Y"),
            "entry_price": operation.entrada,
            "min_entry": operation.entrada_min,
            "max_entry": operation.entrada_max,
            "target": operation.alvo,
            "stop_loss": operation.stop,
            "partial_exit": operation.parcial_preco,
            "partial_points": operation.parcial_pontos,
            "target_points": operation.pontos_alvo,
            "stop_loss_points": operation.pontos_stop,
            "risk_zero": operation.risco_retorno
        }
        return self.generate_day_trade_pdf(session, charts)

    def generate_portfolio_pdf(self, portfolio: Dict[str, Any]) -> str:
        # Generic fallback
        return self.generate_day_trade_pdf(portfolio)


ReportGenerator = ModernReportGenerator
__all__ = ["ReportGenerator", "ModernReportGenerator"]
