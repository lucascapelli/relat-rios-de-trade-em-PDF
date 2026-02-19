from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BrandingAssets:
    logo_path: Optional[str] = None
    watermark_path: Optional[str] = None
    icon_path: Optional[str] = None
    cover_top_image: Optional[str] = None
    cover_bottom_image: Optional[str] = None
    disclaimer_side_image: Optional[str] = None
    selo_apimec_path: Optional[str] = None
    selo_cvm_path: Optional[str] = None
    color_primary: str = "#0F1B2D"  # fundo azul-marinho/acinzentado
    color_accent: str = "#C9A646"   # dourado
    color_text: str = "#FFFFFF"    # branco


@dataclass
class ReportIdentity:
    title: str
    period_start: str  # DD/MM/AAAA
    period_end: str    # DD/MM/AAAA
    strategy_name: str = "Carteira Semanal"
    report_type: str = "semanal"  # ex: semanal, mensal
    branding: BrandingAssets = field(default_factory=BrandingAssets)


@dataclass
class EditorialContent:
    headline: str
    body: str
    assets_count: int
    history_since: str  # ex: "janeiro de 2019"
    benchmark: str = "IBOV"
    link: str = "https://www.castling.me"


@dataclass
class AssetCard:
    symbol: str
    entrada: float
    entrada_maxima: Optional[float]
    entrada_minima: Optional[float]
    objetivo: float
    stop_loss: float
    ultimo_preco: Optional[float]
    retorno_pct: float
    risco_pct: float
    risco_zero_preco: float


@dataclass
class PortfolioSection:
    assets: List[AssetCard]
    weights: List[float]
    distribution_chart: Optional[str] = None
    entered: List[str] = field(default_factory=list)
    exited: List[str] = field(default_factory=list)
    technical_notes: List[str] = field(default_factory=list)


@dataclass
class PerformanceStats:
    # ESTATÍSTICA
    num_finalized: int
    num_castling_greater_ibov: int
    num_castling_positive: int
    num_castling_negative: int
    win_rate: float
    avg_overall: float
    avg_gain: float
    avg_loss: float
    vol_annualized: float
    
    # PERFORMANCE
    risk_return: float
    return_accumulated: float
    return_ibov: float
    alpha_pp: float
    sharpe: float
    profit_factor: float
    return_annualized: float
    # MÉTRICAS AVANÇADAS
    max_drawdown: float = 0.0
    expectancy_per_trade: float = 0.0
    payoff_medio: float = 0.0
    win_loss_ratio: float = 0.0
    operations_per_week: float = 0.0
    operations_per_month: float = 0.0
    return_std: float = 0.0
    ulcer_index: float = 0.0
    excess_vs_benchmark: float = 0.0
    # METADADOS INSTITUCIONAIS
    generated_at: str = ""
    model_version: str = "Modelo Quantitativo v1.0"
    data_source: str = "B3 / Market Data"
    executive_summary: List[str] = field(default_factory=list)


@dataclass
class SeriesData:
    cumulative_castling: List[float]
    cumulative_ibov: List[float]
    weekly_returns: List[float]  # últimas 12 semanas Castling
    weekly_returns_ibov: List[float] = field(default_factory=list)  # últimas 12 semanas IBOV
    weekly_labels: List[str] = field(default_factory=list)     # datas/labels das 12 semanas
    cumulative_labels: List[str] = field(default_factory=list)
    base100_chart: Optional[str] = None
    drawdown_chart: Optional[str] = None
    weekly_chart: Optional[str] = None


@dataclass
class ComplianceData:
    analyst_name: str
    analyst_cnpi: str
    link: str = "https://www.castling.me"


@dataclass
class PdfInput:
    identity: ReportIdentity
    editorial: EditorialContent
    portfolio: PortfolioSection
    performance: PerformanceStats
    series: SeriesData
    compliance: ComplianceData
    disclaimer_text: str
