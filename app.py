# ==================== IMPORTAÇÕES ====================
# Bibliotecas principais do Flask para construção da aplicação web
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading  # Para executar tarefas em paralelo
import webbrowser  # Para abrir o navegador automaticamente
import time  # Para controle de tempo e delays
import os  # Para manipulação de arquivos e diretórios
import sqlite3  # Banco de dados embutido para armazenar operações
import base64  # Para codificar imagens em formato texto
import io  # Para manipulação de streams de dados
import json  # Para manipulação de dados JSON
import random  # Para gerar dados de fallback
from datetime import datetime, timedelta  # Para manipulação de datas

# Bibliotecas de análise de dados
import pandas as pd  # Manipulação de dados tabulares
import numpy as np  # Cálculos numéricos

# Bibliotecas financeiras
import yfinance as yf  # Obter dados de mercado (Yahoo Finance)
import mplfinance as mpf  # Gráficos financeiros com matplotlib
import matplotlib  # Gráficos estáticos
matplotlib.use('Agg')  # Usa backend não-interativo (para servidor)
import matplotlib.pyplot as plt

# Bibliotecas de visualização interativa
import plotly.graph_objs as go  # Gráficos interativos
import plotly.io as pio  # Input/Output do Plotly
from weasyprint import HTML  # Gerar PDFs a partir de HTML
import plotly  # Biblioteca principal do Plotly

# WebSocket para comunicação em tempo real
from flask_socketio import SocketIO, emit, join_room, leave_room

# Outras utilidades
import requests  # Requisições HTTP
import pytz  # Timezones (fuso horário Brasil)
import pandas_ta as ta  # Indicadores técnicos para pandas

# Programação orientada a objetos avançada
from dataclasses import dataclass, asdict  # Classes de dados imutáveis
from typing import Dict, List, Optional, Any, Tuple  # Type hints (tipagem)
from abc import ABC, abstractmethod  # Classes abstratas
import logging  # Sistema de logs (registro de eventos)

# ==================== CONFIGURAÇÃO DE LOGGING ====================
# Configura o sistema de registro de eventos da aplicação
logging.basicConfig(
    level=logging.INFO,  # Nível mínimo de log: INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Cria logger principal

# ==================== CONSTANTES E CONFIGURAÇÕES ====================
# Define caminhos e configurações fixas da aplicação
APP_DIR = os.path.abspath(os.path.dirname(__file__))  # Diretório da aplicação
REPORTS_DIR = os.path.join(APP_DIR, 'reports')  # Pasta para relatórios PDF
DB_PATH = os.path.join(APP_DIR, 'database.db')  # Caminho do banco SQLite
CACHE_DIR = os.path.join(APP_DIR, 'cache')  # Pasta para cache
os.makedirs(REPORTS_DIR, exist_ok=True)  # Cria pasta se não existir
os.makedirs(CACHE_DIR, exist_ok=True)

BR_TZ = pytz.timezone('America/Sao_Paulo')  # Fuso horário do Brasil

# ==================== DATACLASSES ====================
# Classes de dados imutáveis para estruturar informações

@dataclass
class TickerInfo:
    """Informações de um ativo (ação)"""
    symbol: str  # Símbolo do ativo (ex: PETR4)
    price: float  # Preço atual
    change: float  # Variação absoluta
    change_percent: float  # Variação percentual
    open: float  # Preço de abertura
    high: float  # Preço máximo do dia
    low: float  # Preço mínimo do dia
    volume: int  # Volume negociado
    previous_close: float  # Fechamento anterior
    name: str  # Nome da empresa
    currency: str = 'BRL'  # Moeda (padrão Real)
    market_cap: float = 0  # Valor de mercado
    timestamp: str = None  # Horário da última atualização
    
    def __post_init__(self):
        """Executado após inicialização, define timestamp atual"""
        if self.timestamp is None:
            self.timestamp = datetime.now(BR_TZ).strftime('%H:%M:%S')

@dataclass
class Operation:
    """Dados de uma operação de trading (compra/venda)"""
    symbol: str  # Símbolo negociado
    tipo: str  # 'COMPRA' ou 'VENDA'
    entrada: float  # Preço de entrada
    stop: float  # Preço do stop loss
    alvo: float  # Preço do alvo (take profit)
    quantidade: int  # Quantidade de ações
    observacoes: str = ''  # Anotações do trader
    preco_atual: float = 0  # Preço atual do ativo
    pontos_alvo: float = 0  # Pontos até o alvo
    pontos_stop: float = 0  # Pontos até o stop
    status: str = 'ABERTA'  # Status da operação
    timeframe: str = '15m'  # Período do gráfico analisado
    created_at: str = None  # Data de criação
    
    def __post_init__(self):
        """Calcula pontos e define data de criação"""
        if self.created_at is None:
            self.created_at = datetime.now(BR_TZ).isoformat()
        
        # Calcula distância em pontos (preço) entre entrada e alvo/stop
        self.pontos_alvo = self.alvo - self.entrada
        self.pontos_stop = self.entrada - self.stop

@dataclass
class Candle:
    """Dados de um candle (vela) do gráfico"""
    time: datetime  # Data/hora do candle
    open: float  # Preço de abertura
    high: float  # Preço máximo
    low: float  # Preço mínimo
    close: float  # Preço de fechamento
    volume: int  # Volume no período
    
    def to_dict(self):
        """Converte para dicionário (serialização)"""
        return {
            'time': self.time.isoformat(),
            'time_str': self.time.strftime('%Y-%m-%d %H:%M:%S'),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }

# ==================== CACHE ====================

class CacheManager:
    """Gerenciador de cache com expiração para melhor performance"""
    
    def __init__(self, expiry_seconds: int = 60):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}  # {chave: (valor, timestamp)}
        self.expiry = expiry_seconds  # Tempo de expiração em segundos
    
    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache se não expirou"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.expiry:
                return data
        return None
    
    def set(self, key: str, value: Any):
        """Armazena valor no cache"""
        self.cache[key] = (value, datetime.now())
    
    def clear_expired(self):
        """Remove itens expirados do cache (limpeza periódica)"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if (now - timestamp).seconds >= self.expiry
        ]
        for key in expired_keys:
            del self.cache[key]

# ==================== INDICADORES TÉCNICOS ====================

class TechnicalIndicators:
    """Classe para cálculo de indicadores técnicos de análise gráfica"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos no DataFrame"""
        df = df.copy()  # Trabalha com cópia para não alterar original
        
        # Médias móveis simples (Simple Moving Average)
        df['SMA_9'] = TechnicalIndicators._safe_calc(ta.sma, df['close'], 9)
        df['SMA_21'] = TechnicalIndicators._safe_calc(ta.sma, df['close'], 21)
        df['SMA_50'] = TechnicalIndicators._safe_calc(ta.sma, df['close'], 50)
        
        # Médias móveis exponenciais (Exponential Moving Average)
        df['EMA_12'] = TechnicalIndicators._safe_calc(ta.ema, df['close'], 12)
        df['EMA_26'] = TechnicalIndicators._safe_calc(ta.ema, df['close'], 26)
        
        # RSI (Relative Strength Index) - indica sobrecompra/venda
        df['RSI'] = TechnicalIndicators._safe_calc(ta.rsi, df['close'], 14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = TechnicalIndicators._safe_calc(ta.macd, df['close'])
        if macd is not None and not macd.empty:
            TechnicalIndicators._add_macd_columns(df, macd)
        
        # Bollinger Bands (Bandas de Bollinger)
        bb = TechnicalIndicators._safe_calc(ta.bbands, df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            TechnicalIndicators._add_bb_columns(df, bb)
        
        # ATR (Average True Range) - mede volatilidade
        df['ATR'] = TechnicalIndicators._safe_calc(
            ta.atr, df['high'], df['low'], df['close'], 14
        )
        
        # Volume médio
        df['Volume_SMA'] = TechnicalIndicators._safe_calc(ta.sma, df['volume'], 20)
        
        # Suporte e resistência (máximos e mínimos recentes)
        if len(df) > 20:
            df['Resistance'] = df['high'].rolling(window=20).max()
            df['Support'] = df['low'].rolling(window=20).min()
        
        return df
    
    @staticmethod
    def _safe_calc(func, *args, **kwargs):
        """Executa cálculo de indicador com tratamento de erros (fail-safe)"""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.debug(f"Erro ao calcular indicador {func.__name__}: {e}")
            return None
    
    @staticmethod
    def _add_macd_columns(df: pd.DataFrame, macd: pd.DataFrame):
        """Adiciona colunas MACD ao DataFrame"""
        for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']:
            if col in macd.columns:
                df[col.replace('_12_26_9', '')] = macd[col]
    
    @staticmethod
    def _add_bb_columns(df: pd.DataFrame, bb: pd.DataFrame):
        """Adiciona colunas Bollinger Bands ao DataFrame"""
        column_map = {
            'BBU_20_2.0': 'BB_upper',  # Banda superior
            'BBM_20_2.0': 'BB_middle', # Banda média (SMA 20)
            'BBL_20_2.0': 'BB_lower'   # Banda inferior
        }
        for old_col, new_col in column_map.items():
            if old_col in bb.columns:
                df[new_col] = bb[old_col]

# ==================== DADOS FINANCEIROS ====================

class FinanceData:
    """Classe responsável por obter dados financeiros de fontes externas"""
    
    def __init__(self):
        self.session = requests.Session()  # Session para reutilizar conexões HTTP
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.tickers = self._load_tickers()  # Carrega lista de tickers da B3
        self.indicators = TechnicalIndicators()  # Instância para calcular indicadores
    
    def _load_tickers(self) -> Dict[str, str]:
        """Carrega lista de tickers da B3 de um repositório GitHub"""
        try:
            url = "https://raw.githubusercontent.com/guilhermecappi/b3-tickers/master/data/tickers.csv"
            df = pd.read_csv(url)
            return dict(zip(df['ticker'], df['company']))  # {symbol: nome}
        except Exception as e:
            logger.warning(f"Erro ao carregar tickers: {e}")
            return self._get_default_tickers()  # Fallback se falhar
    
    def _get_default_tickers(self) -> Dict[str, str]:
        """Retorna tickers padrão como fallback (backup)"""
        return {
            'PETR4.SA': 'Petrobras PN',
            'VALE3.SA': 'Vale ON',
            'ITUB4.SA': 'Itaú Unibanco PN',
            'BBDC4.SA': 'Bradesco PN',
            'BBAS3.SA': 'Banco do Brasil ON',
            'ABEV3.SA': 'Ambev ON',
            'WEGE3.SA': 'Weg ON',
            'MGLU3.SA': 'Magazine Luiza ON',
            'VIIA3.SA': 'Via ON',
            'BOVA11.SA': 'ETF Ibovespa'
        }
    
    def get_ticker_info(self, symbol: str) -> TickerInfo:
        """Obtém informações de um ativo específico"""
        try:
            # Normaliza símbolo (adiciona .SA se for ação brasileira)
            if not symbol.endswith('.SA') and symbol[-1].isdigit():
                symbol = f"{symbol}.SA"
            
            ticker = yf.Ticker(symbol)  # Objeto do yfinance
            info = ticker.info  # Informações gerais
            hist = ticker.history(period='2d', interval='5m')  # Histórico recente
            
            if hist.empty:  # Se não encontrar dados
                return self._create_fallback_info(symbol)
            
            return self._parse_ticker_info(symbol, info, hist)
            
        except Exception as e:
            logger.error(f"Erro ao buscar info para {symbol}: {e}")
            return self._create_fallback_info(symbol)  # Fallback em caso de erro
    
    def _parse_ticker_info(self, symbol: str, info: Dict, hist: pd.DataFrame) -> TickerInfo:
        """Parseia informações do ativo para a estrutura TickerInfo"""
        last_close = float(hist['Close'].iloc[-1])  # Último preço
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last_close  # Fechamento anterior
        
        return TickerInfo(
            symbol=symbol.replace('.SA', ''),  # Remove .SA para exibição
            price=last_close,
            change=last_close - prev_close,  # Variação absoluta
            change_percent=((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0,
            open=float(hist['Open'].iloc[-1]) if 'Open' in hist.columns else last_close,
            high=float(hist['High'].iloc[-1]) if 'High' in hist.columns else last_close,
            low=float(hist['Low'].iloc[-1]) if 'Low' in hist.columns else last_close,
            volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            previous_close=prev_close,
            name=info.get('longName', info.get('shortName', symbol)),  # Nome longo ou curto
            currency=info.get('currency', 'BRL'),
            market_cap=info.get('marketCap', 0)  # Valor de mercado
        )
    
    def _create_fallback_info(self, symbol: str) -> TickerInfo:
        """Cria informações de fallback quando dados reais não estão disponíveis"""
        return TickerInfo(
            symbol=symbol.replace('.SA', ''),
            price=100.0,  # Valores padrão
            change=0,
            change_percent=0,
            open=100.0,
            high=101.0,
            low=99.0,
            volume=1000000,
            previous_close=100.0,
            name=symbol,
            currency='BRL'
        )
    
    def get_candles(self, symbol: str, interval: str = '15m', periods: int = 100) -> pd.DataFrame:
        """Obtém dados de candles para um símbolo e intervalo específicos"""
        try:
            # Normaliza símbolo
            if not symbol.endswith('.SA') and symbol[-1].isdigit():
                symbol_yf = f"{symbol}.SA"
            else:
                symbol_yf = symbol
            
            # Busca dados do yfinance
            df = self._fetch_yfinance_data(symbol_yf, interval, periods)
            
            if df is None or df.empty:
                logger.warning(f"Sem dados reais para {symbol} ({interval}), usando fallback")
                return self._generate_fallback_data(symbol, interval, periods, reason='fallback_no_data')
            
            # Processa dados
            df = self._process_candle_data(df, symbol_yf, periods)
            
            # Adiciona indicadores técnicos
            df = self.indicators.calculate_all(df)

            # Garante dataset utilizável; fallback se insuficiente
            valid = df[['open', 'high', 'low', 'close']].dropna()
            if len(valid) < 2:
                logger.warning(f"Candles insuficientes para {symbol} ({interval}): {len(valid)} disponíveis; usando fallback")
                return self._generate_fallback_data(symbol, interval, periods, reason='fallback_insufficient')

            df.attrs['source'] = 'yfinance'
            self._log_dataset_snapshot(df, symbol, interval, source='yfinance')
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar candles para {symbol}: {e}")
            return self._generate_fallback_data(symbol, interval, periods, reason='fallback_error')
    
    def _fetch_yfinance_data(self, symbol: str, interval: str, periods: int) -> Optional[pd.DataFrame]:
        """Busca dados do yfinance com mapeamento de intervalos"""
        # Mapeia intervalos do sistema para intervalos do yfinance
        interval_map = {
            '1m': ('1m', '1d'), '5m': ('5m', '5d'), '15m': ('15m', '5d'),
            '30m': ('30m', '10d'), '1h': ('60m', '30d'), '1d': ('1d', '3mo'),
            '15min': ('15m', '5d'), '60min': ('60m', '30d'), 'daily': ('1d', '3mo')
        }
        
        yf_interval, yf_period = interval_map.get(interval, ('15m', '5d'))
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=yf_period, interval=yf_interval, timeout=10)
            return df if df is not None and not df.empty else None
        except Exception as e:
            logger.error(f"Erro yfinance para {symbol}: {e}")
            return None
    
    def _process_candle_data(self, df: pd.DataFrame, symbol: str, periods: int) -> pd.DataFrame:
        """Processa dados de candles: formata, ajusta timezone, limita"""
        # Renomeia colunas para padrão lowercase
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 
            'Volume': 'volume'
        })
        
        # Ajusta timezone para Brasil
        if df.index.tz is not None:
            df.index = df.index.tz_convert(BR_TZ)
        else:
            df.index = df.index.tz_localize('UTC').tz_convert(BR_TZ)
        
        # Limita número de períodos
        if len(df) > periods:
            df = df.iloc[-periods:]
        
        # Adiciona colunas de tempo formatadas
        df['time'] = df.index
        df['time_str'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Valida candles (garante que high >= low, etc)
        df = self._validate_candles(df)
        
        logger.info(f"Dados carregados para {symbol}: R$ {df['close'].iloc[-1]:.2f} ({len(df)} candles)")
        
        return df
    
    def _validate_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida integridade dos candles"""
        # Converte tipos para numérico
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # coerce: transforma erros em NaN
        
        # Filtra candles inválidos (garante relações lógicas entre preços)
        mask = (
            (df['high'] >= df['low']) &  # Máxima >= Mínima
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        )
        
        return df[mask].copy()  # Retorna apenas candles válidos
    
    def _log_dataset_snapshot(self, df: pd.DataFrame, symbol: str, interval: str, source: str):
        """Registra amostra dos dados obtidos para inspeção"""
        if df is None or df.empty:
            logger.warning(f"[MarketData] {source} {symbol} ({interval}): dataset vazio")
            return

        first = df.iloc[0]
        last = df.iloc[-1]

        def safe_time(row: pd.Series) -> str:
            if 'time_str' in row.index:
                return str(row['time_str'])
            if 'time' in row.index:
                return str(row['time'])
            return str(row.name)

        def safe_value(row: pd.Series, key: str) -> Optional[float]:
            if key in row.index and not pd.isna(row[key]):
                value = row[key]
                if isinstance(value, (int, float, np.floating)):
                    return float(value)
            return None

        def fmt(value: Optional[float]) -> str:
            return f"{value:.2f}" if value is not None else "-"

        msg = (
            f"[MarketData] {source} {symbol} ({interval}) -> registros={len(df)} "
            f"primeiro={safe_time(first)} O={fmt(safe_value(first, 'open'))} C={fmt(safe_value(first, 'close'))} "
            f"último={safe_time(last)} O={fmt(safe_value(last, 'open'))} C={fmt(safe_value(last, 'close'))}"
        )
        logger.info(msg)

    def _generate_fallback_data(self, symbol: str, interval: str, periods: int, reason: str = 'fallback') -> pd.DataFrame:
        """Gera dados de fallback realistas quando API falha"""
        # Configurações por intervalo (volatilidade e frequência)
        config = {
            '1m': {'freq': '1min', 'vol': 0.002},
            '5m': {'freq': '5min', 'vol': 0.005},
            '15m': {'freq': '15min', 'vol': 0.008},
            '30m': {'freq': '30min', 'vol': 0.012},
            '1h': {'freq': '1h', 'vol': 0.015},
            '1d': {'freq': '1D', 'vol': 0.02}
        }
        
        cfg = config.get(interval, config['15m'])  # Usa 15m como padrão
        
        # Gera timestamps
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=cfg['freq'])
        
        # Gera preços com tendência e ruído
        base_price = 100 + (hash(symbol) % 100)  # Preço base único por símbolo
        trend_direction = 1 if (hash(symbol) % 2) == 0 else -1  # Tendência aleatória
        trend = np.linspace(0, trend_direction * cfg['vol'] * 20, periods)  # Tendência linear
        noise = np.random.normal(0, cfg['vol'], periods)  # Ruído aleatório
        close_prices = base_price * (1 + trend + noise.cumsum())  # Preços de fechamento
        
        # Gera dados OHLCV (Open, High, Low, Close, Volume)
        data = []
        for i in range(periods):
            open_price = close_prices[i-1] if i > 0 else close_prices[i] * (1 + np.random.uniform(-0.005, 0.005))
            close_price_val = close_prices[i]
            
            # Determina se candle é de alta ou baixa
            is_bullish = close_price_val >= open_price
            body_range = abs(close_price_val - open_price)  # Tamanho do corpo
            wick_range = body_range + base_price * cfg['vol'] * 2  # Tamanho das sombras
            
            # Gera high e low baseado na direção
            if is_bullish:  # Candle de alta (verde)
                low = open_price - np.random.uniform(0, wick_range * 0.3)  # Sombra inferior menor
                high = close_price_val + np.random.uniform(0, wick_range * 0.7)  # Sombra superior maior
            else:  # Candle de baixa (vermelho)
                low = close_price_val - np.random.uniform(0, wick_range * 0.7)
                high = open_price + np.random.uniform(0, wick_range * 0.3)
            
            # Garante validade dos preços
            high = max(high, low + 0.01)
            open_price = max(low, min(high, open_price))
            close_price_val = max(low, min(high, close_price_val))
            
            # Volume proporcional à volatilidade
            volume = np.random.randint(10000, 1000000) * (1 + body_range/base_price)
            
            data.append({
                'time': dates[i],
                'time_str': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price_val, 2),
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data)
        df = self.indicators.calculate_all(df)  # Adiciona indicadores
        df.attrs['source'] = reason
        self._log_dataset_snapshot(df, symbol, interval, source=reason)
        logger.info(f"Fallback gerado para {symbol} ({interval}) com razão '{reason}': {len(df)} candles")
        
        return df

# ==================== GERADOR DE GRÁFICOS ====================

class ChartGenerator:
    """Classe para geração de gráficos interativos e estáticos"""
    
    @staticmethod
    def create_plotly_chart(df: pd.DataFrame, title: str = "", 
                           show_volume: bool = True, show_indicators: bool = True) -> go.Figure:
        """Cria gráfico Plotly interativo com candles e indicadores"""
        if df is None or df.empty:
            logger.warning("DataFrame vazio recebido para gráfico")
            return go.Figure()
        
        df = ChartGenerator._prepare_dataframe(df)  # Prepara dados
        
        if len(df) < 2:
            logger.warning(f"Poucos dados válidos: {len(df)} candles")
            return go.Figure()
        
        # Cria traço de candlestick
        candlestick = go.Candlestick(
            x=df['time'],  # Eixo X: tempo
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Preço',
            increasing_line_color='#26a69a',  # Verde para alta
            decreasing_line_color='#ef5350',   # Vermelho para baixa
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            line=dict(width=1),
            whiskerwidth=0.8,  # Largura das sombras
            hoverinfo='text',   # Informação ao passar mouse
            text=ChartGenerator._generate_hover_text(df)  # Texto do hover
        )
        
        data = [candlestick]  # Lista de traços
        
        # Adiciona indicadores se solicitado
        if show_indicators:
            data.extend(ChartGenerator._add_indicators(df))
        
        # Cria layout do gráfico
        layout = ChartGenerator._create_layout(title, show_volume)
        
        # Adiciona volume se necessário
        if show_volume and 'volume' in df.columns:
            data.append(ChartGenerator._create_volume_trace(df))
            layout = ChartGenerator._add_volume_axis(layout)
        
        fig = go.Figure(data=data, layout=layout)
        
        # Adiciona seletor de range (zoom temporal)
        fig = ChartGenerator._add_range_selector(fig)
        
        return fig
    
    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Prepara DataFrame para plotagem: formata, converte, valida"""
        df = df.copy()
        
        # Garante coluna time como datetime
        if 'time_str' in df.columns and 'time' not in df.columns:
            df['time'] = pd.to_datetime(df['time_str'], errors='coerce')
        elif 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Remove timezone para compatibilidade
        if hasattr(df['time'].dt, 'tz') and df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
        
        # Converte tipos numéricos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN e valida candles
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[
            (df['high'] >= df['low']) & 
            (df['close'] <= df['high']) & (df['close'] >= df['low']) &
            (df['open'] <= df['high']) & (df['open'] >= df['low'])
        ]
        
        return df
    
    @staticmethod
    def _generate_hover_text(df: pd.DataFrame) -> List[str]:
        """Gera texto para hover (tooltip) de cada candle"""
        return [
            f"Abertura: R$ {o:.2f}<br>Máxima: R$ {h:.2f}<br>Mínima: R$ {l:.2f}<br>Fechamento: R$ {c:.2f}<br>Volume: {v:,.0f}"
            for o, h, l, c, v in zip(
                df['open'], df['high'], df['low'], 
                df['close'], df.get('volume', [0]*len(df))
            )
        ]
    
    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> List[go.Scatter]:
        """Adiciona traços de indicadores técnicos ao gráfico"""
        traces = []
        
        # SMAs (linhas)
        for period, color in [(9, '#2196F3'), (21, '#FF9800')]:
            col = f'SMA_{period}'
            if col in df.columns and df[col].notna().sum() > 0:
                traces.append(go.Scatter(
                    x=df['time'], y=df[col],
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(color=color, width=1.5),
                    opacity=0.7
                ))
        
        # Bollinger Bands (bandas)
        if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
            if df['BB_upper'].notna().sum() > 0:
                traces.extend([
                    go.Scatter(
                        x=df['time'], y=df['BB_upper'],
                        mode='lines',
                        name='BB Superior',
                        line=dict(color='rgba(158, 158, 158, 0.5)', width=1, dash='dash'),
                        showlegend=False
                    ),
                    go.Scatter(
                        x=df['time'], y=df['BB_lower'],
                        mode='lines',
                        name='BB Inferior',
                        fill='tonexty',  # Preenche entre as bandas
                        fillcolor='rgba(158, 158, 158, 0.1)',
                        line=dict(color='rgba(158, 158, 158, 0.5)', width=1, dash='dash'),
                        showlegend=False
                    )
                ])
        
        return traces
    
    @staticmethod
    def _create_layout(title: str, show_volume: bool) -> go.Layout:
        """Cria layout do gráfico com configurações visuais"""
        return go.Layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#2c3e50', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                type='date',
                gridcolor='#ecf0f1',
                showgrid=True,
                rangeslider=dict(visible=False),  # Range slider escondido (adicionado depois)
                nticks=15,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title='Preço (R$)',
                gridcolor='#ecf0f1',
                showgrid=True,
                side='right',
                tickformat='.2f',
                tickprefix='R$ ',
                tickfont=dict(size=10)
            ),
            height=550,
            margin=dict(l=50, r=60, t=70, b=50),
            hovermode='x unified',  # Hover unificado para todos os traços
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=9)
            ),
            font=dict(family='Arial, sans-serif')
        )
    
    @staticmethod
    def _create_volume_trace(df: pd.DataFrame) -> go.Bar:
        """Cria traço de volume (barras)"""
        # Cores baseadas na direção do candle
        colors = [
            'rgba(38, 166, 154, 0.6)' if c >= o else 'rgba(239, 83, 80, 0.6)'
            for c, o in zip(df['close'], df['open'])
        ]
        
        return go.Bar(
            x=df['time'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            yaxis='y2',  # Usa segundo eixo Y
            hoverinfo='y'
        )
    
    @staticmethod
    def _add_volume_axis(layout: go.Layout) -> go.Layout:
        """Adiciona eixo de volume ao layout"""
        layout.update(
            yaxis2=dict(
                title='Volume',
                overlaying='y',  # Sobrepoe ao eixo Y principal
                side='left',     # Lado esquerdo
                showgrid=False,
                tickformat=',.0f',
                tickfont=dict(size=9)
            )
        )
        return layout
    
    @staticmethod
    def _add_range_selector(fig: go.Figure) -> go.Figure:
        """Adiciona seletor de range (botões de zoom temporal)"""
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=30, label="30min", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]),
                bgcolor='rgba(255, 255, 255, 0.9)',
                activecolor='#2196F3',
                x=0.01, y=1.05,
                font=dict(size=9)
            )
        )
        return fig
    
    @staticmethod
    def generate_chart_image(df: pd.DataFrame, title: str = "", 
                            width: int = 800, height: int = 400) -> str:
        """Gera imagem estática do gráfico para PDF (base64)"""
        try:
            if df is None or df.empty:
                return ""
            
            df_plot = df.copy()
            
            # Prepara DataFrame para mplfinance
            if 'time' in df_plot.columns:
                df_plot.set_index('time', inplace=True)
            
            if df_plot.index.tz is not None:
                df_plot.index = df_plot.index.tz_localize(None)
            
            # Salva indicadores para plotagem
            sma_9 = df_plot['SMA_9'].copy() if 'SMA_9' in df_plot.columns else None
            sma_21 = df_plot['SMA_21'].copy() if 'SMA_21' in df_plot.columns else None
            
            # Renomeia colunas para padrão do mplfinance
            df_plot = df_plot.rename(columns={
                'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close',
                'volume': 'Volume'
            })
            
            # Verifica colunas necessárias
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df_plot.columns for col in required_cols):
                return ""
            
            # Remove NaN
            df_plot = df_plot[required_cols + (['Volume'] if 'Volume' in df_plot.columns else [])].dropna()
            
            if len(df_plot) < 2:
                return ""
            
            # Configura estilo visual
            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick={'up': '#26a69a', 'down': '#ef5350'},
                volume={'up': '#26a69a', 'down': '#ef5350'},
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#ecf0f1',
                facecolor='white',
                figcolor='white',
                rc={'font.size': 9}
            )
            
            # Adiciona médias móveis ao gráfico
            addplot = []
            for sma, color in [(sma_9, 'blue'), (sma_21, 'orange')]:
                if sma is not None and sma.notna().sum() > 5:
                    sma_aligned = sma.reindex(df_plot.index).dropna()
                    if len(sma_aligned) > 0:
                        addplot.append(mpf.make_addplot(sma_aligned, color=color, width=1.2))
            
            # Cria figura com mplfinance
            fig, _ = mpf.plot(
                df_plot,
                type='candle',
                style=s,
                title=title,
                ylabel='Preço (R$)',
                volume=True if 'Volume' in df_plot.columns else False,
                addplot=addplot if addplot else None,
                figsize=(width/80, height/80),  # Converte pixels para polegadas
                returnfig=True,
                warn_too_much_data=1000
            )
            
            # Salva em buffer de memória como PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Fecha figura para liberar memória
            buf.seek(0)
            
            # Converte para base64 (string)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{b64}"  # Data URL para uso em HTML
            
        except Exception as e:
            logger.error(f"Erro ao gerar imagem: {e}")
            return ""

# ==================== GERENCIADOR DE TEMPO REAL ====================

class RealTimeManager:
    """Gerenciador de atualizações em tempo real via WebSocket"""
    
    def __init__(self, socketio: SocketIO, finance_data: FinanceData, update_interval: int = 5):
        self.socketio = socketio  # Instância do SocketIO
        self.finance_data = finance_data  # Fonte de dados
        self.update_interval = update_interval  # Intervalo entre atualizações (segundos)
        self.active_symbols = set()  # Símbolos sendo monitorados
        self.symbol_data = {}  # Cache de dados por símbolo
        self.running = False  # Flag de controle do loop

    def _normalize_symbol(self, symbol: str) -> str:
        """Normaliza símbolo para formato aceito pelo yfinance"""
        if symbol and not symbol.endswith('.SA') and symbol[-1].isdigit():
            return f"{symbol}.SA"
        return symbol
    
    def update_symbol(self, symbol: str):
        """Atualiza dados de um símbolo específico"""
        try:
            info = self.finance_data.get_ticker_info(symbol)
            
            if info:
                room_symbol = symbol
                display_symbol = room_symbol.replace('.SA', '')
                # Armazena dados com timestamp
                self.symbol_data[symbol] = {
                    **asdict(info),
                    'timestamp': datetime.now(BR_TZ).isoformat(),
                    'update_count': self.symbol_data.get(symbol, {}).get('update_count', 0) + 1
                }
                
                # Emite atualização via WebSocket para sala específica
                self.socketio.emit('price_update', {
                    'symbol': display_symbol,
                    'data': asdict(info)
                }, room=f"symbol_{room_symbol}")
                
                return info
                
        except Exception as e:
            logger.error(f"Erro ao atualizar {symbol}: {e}")
        
        return None
    
    def start_updates(self):
        """Inicia loop de atualizações em tempo real (executa em thread)"""
        self.running = True
        while self.running:
            try:
                symbols = list(self.active_symbols)
                
                if symbols:
                    for symbol in symbols:
                        self.update_symbol(symbol)
                
                time.sleep(self.update_interval)  # Aguarda intervalo
                
            except Exception as e:
                logger.error(f"Erro no loop de atualizações: {e}")
                time.sleep(10)  # Aguarda mais em caso de erro
    
    def stop_updates(self):
        """Para as atualizações"""
        self.running = False
    
    def subscribe(self, symbol: str):
        """Inscreve em um símbolo (adiciona à lista de monitoramento)"""
        normalized = self._normalize_symbol(symbol)
        self.active_symbols.add(normalized)
        return normalized
    
    def unsubscribe(self, symbol: str):
        """Cancela inscrição em um símbolo"""
        normalized = self._normalize_symbol(symbol)
        if normalized in self.active_symbols:
            self.active_symbols.remove(normalized)
        return normalized

# ==================== BANCO DE DADOS ====================

class Database:
    """Classe para operações de banco de dados SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()  # Inicializa tabelas
    
    def _init_db(self):
        """Inicializa banco de dados, cria tabela se não existir"""
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute('''
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                tipo TEXT,
                entrada REAL,
                stop REAL,
                alvo REAL,
                quantidade INTEGER,
                observacoes TEXT,
                preco_atual REAL,
                pontos_alvo REAL,
                pontos_stop REAL,
                status TEXT,
                created_at TEXT,
                pdf_path TEXT,
                timeframe TEXT,
                indicators TEXT
            )
            ''')
            con.commit()
    
    def insert_operation(self, operation: Operation, pdf_path: str = None, 
                        indicators: Dict = None) -> int:
        """Insere operação no banco de dados e retorna ID"""
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute('''
                INSERT INTO operations (
                    symbol, tipo, entrada, stop, alvo, quantidade, observacoes,
                    preco_atual, pontos_alvo, pontos_stop, status, created_at,
                    pdf_path, timeframe, indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                operation.symbol,
                operation.tipo,
                operation.entrada,
                operation.stop,
                operation.alvo,
                operation.quantidade,
                operation.observacoes,
                operation.preco_atual,
                operation.pontos_alvo,
                operation.pontos_stop,
                operation.status,
                operation.created_at,
                pdf_path,
                operation.timeframe,
                json.dumps(indicators) if indicators else '{}'  # Serializa indicadores
            ))
            con.commit()
            return cur.lastrowid  # Retorna ID da operação inserida
    
    def get_operations(self, limit: int = 50) -> List[Dict]:
        """Obtém histórico de operações ordenado por data (mais recente primeiro)"""
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute('''
                SELECT id, symbol, tipo, entrada, stop, alvo, quantidade,
                       observacoes, preco_atual, status, created_at, pdf_path
                FROM operations 
                ORDER BY id DESC 
                LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            
            # Converte para lista de dicionários
            operations = []
            for row in rows:
                operations.append({
                    'id': row[0],
                    'symbol': row[1],
                    'tipo': row[2],
                    'entrada': row[3],
                    'stop': row[4],
                    'alvo': row[5],
                    'quantidade': row[6],
                    'observacoes': row[7],
                    'preco_atual': row[8],
                    'status': row[9],
                    'created_at': row[10],
                    'pdf_path': row[11]
                })
            
            return operations

# ==================== GERADOR DE RELATÓRIOS ====================

class ReportGenerator:
    """Classe para geração de relatórios PDF das operações"""
    
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
    
    def generate_pdf_report(self, operation: Operation, images: Dict[str, str]) -> str:
        """Gera relatório PDF da operação com gráficos"""
        html_content = self._create_html_content(operation, images)
        
        # Nome único para o arquivo
        filename = f"report_{operation.symbol}_{int(time.time())}.pdf"
        path = os.path.join(self.reports_dir, filename)
        
        # Converte HTML para PDF
        HTML(string=html_content).write_pdf(path)
        
        return path
    
    def _create_html_content(self, operation: Operation, images: Dict[str, str]) -> str:
        """Cria conteúdo HTML para o relatório"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Relatório - {operation.symbol}</title>
            <style>
                /* Estilos CSS para o relatório */
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0; }}
                .stat-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-card h3 {{ margin-top: 0; color: #2c3e50; }}
                .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                .observations {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .status-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .status-aberta {{ background: #ffc107; color: #000; }}  /* Amarelo */
                .status-alvo {{ background: #28a745; color: white; }}   /* Verde */
                .status-stop {{ background: #dc3545; color: white; }}   /* Vermelho */
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Relatório de Operação</h1>
                <p>
                    <strong>Símbolo:</strong> {operation.symbol} | 
                    <strong>Tipo:</strong> {operation.tipo} | 
                    <strong>Status:</strong> 
                    <span class="status-badge status-{operation.status.lower().replace(' ', '-')}">
                        {operation.status}
                    </span>
                </p>
                <p><strong>Data:</strong> {datetime.now(BR_TZ).strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>🎯 Entrada</h3>
                    <div class="stat-value">R$ {operation.entrada:.2f}</div>
                </div>
                <div class="stat-card">
                    <h3>🛑 Stop</h3>
                    <div class="stat-value">R$ {operation.stop:.2f}</div>
                    <div>Pontos: {operation.pontos_stop:.2f}</div>
                </div>
                <div class="stat-card">
                    <h3>🚀 Alvo</h3>
                    <div class="stat-value">R$ {operation.alvo:.2f}</div>
                    <div>Pontos: {operation.pontos_alvo:.2f}</div>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>📈 Quantidade</h3>
                    <div class="stat-value">{operation.quantidade}</div>
                </div>
                <div class="stat-card">
                    <h3>💰 Preço Atual</h3>
                    <div class="stat-value">R$ {operation.preco_atual:.2f}</div>
                </div>
                <div class="stat-card">
                    <h3>⚖️ Risco/Retorno</h3>
                    <div class="stat-value">
                        {operation.pontos_alvo/operation.pontos_stop if operation.pontos_stop > 0 else 0:.2f}:1
                    </div>
                </div>
            </div>
            
            <div class="charts">
                <h2>📈 Análise Técnica</h2>
                {"".join([f'''
                <div class="chart-container">
                    <h3>{tf.upper()} - {operation.symbol}</h3>
                    <img src="{img}">
                </div>
                ''' for tf, img in images.items()])}
            </div>
            
            <div class="observations">
                <h2>📝 Observações</h2>
                <p>{operation.observacoes or 'Sem observações registradas.'}</p>
            </div>
            
            <footer style="margin-top: 40px; text-align: center; color: #6c757d; font-size: 12px;">
                <p>Relatório gerado automaticamente pelo Sistema de Trading</p>
                <p>Data: {datetime.now(BR_TZ).strftime('%d/%m/%Y %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """

# ==================== APLICAÇÃO FLASK ====================

# Inicialização da aplicação Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua-chave-secreta-aqui'  # Chave para sessões
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Instâncias globais (singletons)
finance_data = FinanceData()
chart_generator = ChartGenerator()
database = Database(DB_PATH)
report_generator = ReportGenerator(REPORTS_DIR)
realtime_manager = RealTimeManager(socketio, finance_data)

# Caches para melhor performance
price_cache = CacheManager(expiry_seconds=30)
chart_cache = CacheManager(expiry_seconds=15)

# ==================== ROTAS API ====================

@app.route('/')
def index():
    """Página principal com dashboard de ações e gráfico inicial"""
    symbols = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4']  # Ações para exibir no dashboard
    stocks = []
    
    for sym in symbols:
        try:
            info = finance_data.get_ticker_info(sym)
            if info:
                stocks.append(asdict(info))
        except Exception as e:
            logger.error(f"Erro ao buscar {sym}: {e}")
    
    # Gráfico inicial (PETR4 como exemplo)
    try:
        df = finance_data.get_candles('PETR4', '15m', 50)
        fig = chart_generator.create_plotly_chart(df, 'PETR4 - 15 Minutos')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)  # Serializa gráfico
    except Exception as e:
        logger.error(f"Erro no gráfico inicial: {e}")
        graphJSON = "{}"
    
    # Renderiza template HTML
    return render_template('index.html', 
                         stocks=stocks[:4],  # Máximo 4 ações
                         graphJSON=graphJSON,
                         timeframe='15m')

@app.route('/api/quote/<symbol>')
def api_quote(symbol):
    """API para cotação em tempo real (REST)"""
    try:
        info = finance_data.get_ticker_info(symbol)
        return jsonify(asdict(info))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _build_chart_components(df: pd.DataFrame, limit: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Serializa candles e séries completas para consumo no front-end"""
    df_tail = df.iloc[-limit:].copy()
    candles: List[Dict[str, Any]] = []
    series: Dict[str, List[Any]] = {
        'time': [],
        'time_str': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    indicator_columns = {
        'sma_9': 'SMA_9',
        'sma_21': 'SMA_21',
        'ema_12': 'EMA_12',
        'ema_26': 'EMA_26',
        'rsi_series': 'RSI',
        'bb_upper': 'BB_upper',
        'bb_middle': 'BB_middle',
        'bb_lower': 'BB_lower'
    }

    for key, col in indicator_columns.items():
        if col in df_tail.columns:
            series[key] = []

    for _, row in df_tail.iterrows():
        volume_value = row.get('volume', 0)
        if pd.isna(volume_value):
            volume_value = 0

        iso_time = row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time'])
        time_str = row['time_str'] if 'time_str' in row.index else iso_time

        candles.append({
            'time': iso_time,
            'time_str': time_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(volume_value)
        })

        series['time'].append(iso_time)
        series['time_str'].append(time_str)
        series['open'].append(float(row['open']))
        series['high'].append(float(row['high']))
        series['low'].append(float(row['low']))
        series['close'].append(float(row['close']))
        series['volume'].append(int(volume_value))

        for key, col in indicator_columns.items():
            if key in series:
                value = row.get(col)
                series[key].append(float(value) if value is not None and not pd.isna(value) else None)

    return candles, series


@app.route('/api/chart/<symbol>/<interval>')
def api_chart(symbol, interval):
    """API para dados do gráfico (candles + indicadores)"""
    try:
        df = finance_data.get_candles(symbol, interval, 100)
        fig = chart_generator.create_plotly_chart(df, f'{symbol} - {interval}')

        candles_payload, series = _build_chart_components(df, limit=100)

        source = df.attrs.get('source', 'unknown')
        if candles_payload:
            last_candle = candles_payload[-1]
            last_open = float(last_candle.get('open', 0) or 0)
            last_close = float(last_candle.get('close', 0) or 0)
            logger.info(
                f"/api/chart {symbol}/{interval} -> source={source} candles={len(candles_payload)} "
                f"último={last_candle.get('time_str')} O={last_open:.2f} C={last_close:.2f}"
            )
        else:
            logger.warning(f"/api/chart {symbol}/{interval} -> source={source} sem candles retornados")

        indicators_snapshot = {
            'sma_9': float(df['SMA_9'].iloc[-1]) if 'SMA_9' in df.columns else None,
            'sma_21': float(df['SMA_21'].iloc[-1]) if 'SMA_21' in df.columns else None,
            'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
        }

        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'source': source,
            'candles': candles_payload,
            'series': series,
            'indicators': indicators_snapshot,
            'chart_data': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/operacao', methods=['POST'])
def api_operation():
    """Registra nova operação de trading via API"""
    try:
        data = request.get_json()  # Dados da operação
        
        # Busca preço atual do ativo
        symbol = data.get('ativo') or data.get('symbol')
        info = finance_data.get_ticker_info(symbol)
        
        if not info:
            return jsonify({'error': 'Símbolo não encontrado'}), 400
        
        # Cria objeto Operation com dados recebidos
        operation = Operation(
            symbol=symbol,
            tipo=data.get('tipo', 'COMPRA'),
            entrada=float(data['entrada']),
            stop=float(data['stop']),
            alvo=float(data['alvo']),
            quantidade=int(data.get('quantidade', 100)),
            observacoes=data.get('observacoes', ''),
            preco_atual=info.price,
            timeframe=data.get('timeframe', '15m')
        )
        
        # Calcula status baseado no preço atual
        if operation.preco_atual >= operation.alvo:
            operation.status = 'ALVO ATINGIDO'
        elif operation.preco_atual <= operation.stop:
            operation.status = 'STOP ATINGIDO'
        
        # Gera gráficos para PDF (múltiplos timeframes)
        images = {}
        for tf in ['15m', '1h', '1d']:
            df = finance_data.get_candles(operation.symbol, tf, 80)
            img = chart_generator.generate_chart_image(df, f"{operation.symbol} - {tf}")
            if img:
                images[tf] = img
        
        # Gera relatório PDF
        pdf_path = report_generator.generate_pdf_report(operation, images)
        
        # Extrai indicadores atuais
        indicators = {
            'sma_9': float(df['SMA_9'].iloc[-1]) if 'SMA_9' in df.columns else None,
            'sma_21': float(df['SMA_21'].iloc[-1]) if 'SMA_21' in df.columns else None,
            'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
        }
        
        # Salva no banco de dados
        op_id = database.insert_operation(operation, pdf_path, indicators)
        
        return jsonify({
            'id': op_id,
            'status': 'success',
            'pdf_url': f'/reports/{os.path.basename(pdf_path)}',
            'operation': asdict(operation)
        })
        
    except Exception as e:
        logger.error(f"Erro ao registrar operação: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<query>')
def api_search(query):
    """Busca símbolos por nome ou código"""
    results = []
    
    for symbol, name in finance_data.tickers.items():
        if query.upper() in symbol or query.upper() in name.upper():
            results.append({
                'symbol': symbol.replace('.SA', ''),
                'name': name,
                'type': 'Ação' if '.SA' in symbol else 'ETF'
            })
    
    return jsonify(results[:10])  # Limita a 10 resultados

@app.route('/api/history')
def api_history():
    """Retorna histórico de operações registradas"""
    try:
        operations = database.get_operations(50)
        return jsonify(operations)
    except Exception as e:
        logger.error(f"Erro ao buscar histórico: {e}")
        return jsonify([])

@app.route('/reports/<filename>')
def serve_report(filename):
    """Serve arquivos de relatório PDF"""
    return send_from_directory(REPORTS_DIR, filename)

# ==================== WEBSOCKET HANDLERS ====================

@socketio.on('subscribe')
def handle_subscribe(data):
    """Cliente se inscreve para atualizações em tempo real de um símbolo"""
    symbol = data.get('symbol')
    
    if symbol:
        room_symbol = realtime_manager.subscribe(symbol)
        join_room(f"symbol_{room_symbol}")  # Entra na sala normalizada
        
        # Envia dados atuais imediatamente
        info = finance_data.get_ticker_info(symbol)
        if info:
            emit('price_update', {
                'symbol': info.symbol,
                'data': asdict(info)
            })
        
        # Confirma inscrição
        emit('subscription_confirmed', {
            'symbol': symbol,
            'message': f'Inscrito em {symbol}'
        })

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Cliente cancela inscrição em um símbolo"""
    symbol = data.get('symbol')
    
    if symbol:
        room_symbol = realtime_manager.unsubscribe(symbol)
        leave_room(f"symbol_{room_symbol}")  # Sai da sala normalizada
        
        emit('unsubscription_confirmed', {
            'symbol': symbol,
            'message': f'Inscrição cancelada para {symbol}'
        })

@socketio.on('request_chart')
def handle_chart_request(data):
    """Envia dados do gráfico via WebSocket (push)"""
    try:
        symbol = data.get('symbol')
        interval = data.get('interval', '15m')
        
        df = finance_data.get_candles(symbol, interval, 100)
        fig = chart_generator.create_plotly_chart(df, f'{symbol} - {interval}')
        candles_payload, series = _build_chart_components(df, limit=100)
        source = df.attrs.get('source', 'unknown')

        emit('chart_data', {
            'symbol': symbol,
            'interval': interval,
            'source': source,
            'candles': candles_payload,
            'series': series,
            'indicators': {
                'sma_9': float(df['SMA_9'].iloc[-1]) if 'SMA_9' in df.columns else None,
                'sma_21': float(df['SMA_21'].iloc[-1]) if 'SMA_21' in df.columns else None,
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
            },
            'chart': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        })
        
    except Exception as e:
        emit('chart_error', {'error': str(e)})

# ==================== TAREFAS EM BACKGROUND ====================

def start_background_tasks():
    """Inicia tarefas em background (threads)"""
    logger.info("🚀 Iniciando sistema de dados financeiros...")
    
    # Thread para atualizações em tempo real
    update_thread = threading.Thread(target=realtime_manager.start_updates, daemon=True)
    update_thread.start()
    
    # Thread para limpeza periódica de cache
    def cache_cleaner():
        while True:
            time.sleep(300)  # 5 minutos
            price_cache.clear_expired()
            chart_cache.clear_expired()
            logger.debug("🧹 Cache limpo")
    
    cleaner_thread = threading.Thread(target=cache_cleaner, daemon=True)
    cleaner_thread.start()

# ==================== INICIALIZAÇÃO ====================

if __name__ == '__main__':
    # Inicia tarefas em background
    start_background_tasks()
    
    # Abre navegador automaticamente (opcional)
    threading.Thread(
        target=lambda: (time.sleep(3), webbrowser.open('http://127.0.0.1:5000')),
        daemon=True
    ).start()
    
    logger.info("✅ Servidor iniciado em http://127.0.0.1:5000")
    logger.info("📊 Sistema de dados financeiros ativo")
    
    # Inicia servidor Flask com SocketIO
    socketio.run(app, 
                debug=True,  # Modo debug (desativar em produção)
                port=5000, 
                allow_unsafe_werkzeug=True,
                log_output=False)