// Trading System Frontend Application
class TradingApp {
    constructor() {
        this.socket = null;
        this.currentSymbol = 'PETR4';
        this.allowedIntervals = ['1m', '15m', '60m'];
        this.intervalDurations = {
            '1m': 1,
            '15m': 15,
            '60m': 60,
            '1h': 60,
            '1d': 1440
        };
        this.scaleConfig = {
            '1d': {
                label: '1D',
                durationMinutes: 1440,
                preferredIntervals: ['1m', '15m', '60m'],
                defaultInterval: '15m'
            },
            '5d': {
                label: '5D',
                durationMinutes: 7200,
                preferredIntervals: ['15m', '60m'],
                defaultInterval: '60m'
            },
            '1m': {
                label: '1M',
                durationMinutes: 43200,
                preferredIntervals: ['60m', '15m'],
                defaultInterval: '60m'
            }
        };
        this.maxCandlesPerRequest = 1500;
        this.currentScale = '1d';
        this.currentInterval = this.enforceIntervalForScale(this.currentScale, '15m', { silent: true }).interval;
        this.charts = {};
        this.chartConfig = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d']
        };
        this.lastSeries = null;
        this.lastIndicators = null;
        this.lastChartTitle = '';
        this.subscribedSymbols = new Set();
        this.socketSubscriptions = new Set();
        this.operations = [];
        this.lastDataSource = null;
        this.portfolioUI = null;
        
        this.init();
    }
    
    init() {
        // Initialize Socket.IO
        this.initSocket();
        
        // Load initial data
        this.loadDashboardData();
        
        // Setup event listeners before rendering charts
        this.setupEventListeners();
        this.updateIntervalButtons();
        this.updateScaleButtons();
        this.loadChart(this.currentSymbol, this.currentInterval);
        this.loadOperationsHistory();

        // Init portfolio UI (Carteiras)
        this.initPortfolioUI();
        
        // Show welcome message
        setTimeout(() => {
            this.showToast('Sistema de trading carregado com sucesso!', 'success');
        }, 1000);
    }

    initPortfolioUI() {
        try {
            this.portfolioUI = new PortfolioUI({ showToast: this.showToast.bind(this) });
        } catch (err) {
            console.error('Erro ao inicializar carteiras:', err);
        }
    }
    
    inferTickSize(symbol) {
        const clean = (symbol || '').toUpperCase();
        if (clean.startsWith('WIN') || clean.startsWith('IND')) {
            return 5;
        }
        if (clean.startsWith('WDO') || clean.startsWith('DOL')) {
            return 0.5;
        }
        return 0.01;
    }

    formatCurrency(value) {
        if (typeof value !== 'number' || Number.isNaN(value)) {
            return 'R$ -';
        }
        return `R$ ${value.toFixed(2)}`;
    }

    formatTicks(value) {
        if (typeof value !== 'number' || Number.isNaN(value)) {
            return '-';
        }
        return `${value.toFixed(1)} ticks`;
    }

    sanitizeNumber(value) {
        const parsed = parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    computeDirectionalTicks(tipo, referencia, priceLevel, tickSize) {
        if (!Number.isFinite(referencia) || !Number.isFinite(priceLevel) || !Number.isFinite(tickSize) || tickSize <= 0) {
            return 0;
        }
        const isCompra = (tipo || 'COMPRA').toUpperCase() === 'COMPRA';
        const diff = isCompra ? (priceLevel - referencia) : (referencia - priceLevel);
        const ticks = diff / tickSize;
        return Number.isFinite(ticks) ? Math.abs(ticks) : 0;
    }

    initSocket() {
        // Connect to WebSocket server
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Conectado ao servidor WebSocket');
            this.updateConnectionStatus(true);
            this.showToast('Conectado ao servidor em tempo real', 'success');
            this.socketSubscriptions.clear();
            
            // Subscribe to default symbols
            if (this.subscribedSymbols.size === 0) {
                this.subscribeToSymbol(this.currentSymbol);
            } else {
                Array.from(this.subscribedSymbols).forEach(symbol => {
                    this.subscribeToSymbol(symbol);
                });
            }
        });
        
        this.socket.on('disconnect', () => {
            console.log('Desconectado do servidor WebSocket');
            this.updateConnectionStatus(false);
            this.showToast('Conexão perdida. Tentando reconectar...', 'warning');
            this.socketSubscriptions.clear();
        });
        
        this.socket.on('price_update', (data) => {
            this.handlePriceUpdate(data);
        });
        
        this.socket.on('chart_data', (data) => {
            this.handleChartData(data);
        });
        
        this.socket.on('subscription_confirmed', (data) => {
            console.log('Inscrição confirmada:', data);
        });
        
        this.socket.on('unsubscription_confirmed', (data) => {
            console.log('Inscrição cancelada:', data);
        });
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i>Conectado';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-danger me-1"></i>Desconectado';
        }
    }
    
    async loadDashboardData() {
        try {
            // Load main stocks
            const symbols = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4'];
            const stocksList = document.getElementById('stocks-list');
            
            for (const symbol of symbols) {
                const response = await fetch(`/api/quote/${symbol}`);
                const data = await response.json();
                
                const stockItem = this.createStockItem(data);
                stocksList.appendChild(stockItem);
                
                // Subscribe to real-time updates
                this.subscribeToSymbol(symbol);
            }
            
            // Load recent operations
            await this.loadRecentOperations();

            this.updateSymbolButtons(this.currentSymbol);
            
        } catch (error) {
            console.error('Erro ao carregar dados do dashboard:', error);
            this.showToast('Erro ao carregar dados do dashboard', 'danger');
        }
    }
    
    createStockItem(stock) {
        const div = document.createElement('div');
        div.className = 'list-group-item list-group-item-action';
        div.setAttribute('data-symbol', stock.symbol);
        div.innerHTML = `
            <div class="d-flex w-100 justify-content-between align-items-center">
                <div>
                    <h6 class="mb-1">${stock.symbol}</h6>
                    <small class="text-muted">${stock.name}</small>
                </div>
                <div class="text-end">
                    <div class="h6 mb-1 price-display" data-symbol="${stock.symbol}">
                        R$ ${stock.price.toFixed(2)}
                    </div>
                    <small class="${stock.change >= 0 ? 'positive' : 'negative'}">
                        ${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)} 
                        (${stock.change >= 0 ? '+' : ''}${stock.change_percent.toFixed(2)}%)
                    </small>
                </div>
            </div>
        `;

        div.addEventListener('click', () => {
            this.changeSymbol(stock.symbol);
        });
        return div;
    }
    
    async loadChart(symbol, interval) {
        try {
            const normalizedSymbol = (symbol || '').toUpperCase();
            let targetInterval = interval || this.currentInterval;
            const enforced = this.enforceIntervalForScale(this.currentScale, targetInterval, { silent: true });
            targetInterval = enforced.interval;
            this.currentInterval = targetInterval;
            this.updateIntervalButtons();
            this.updateScaleButtons();

            const limit = this.resolveFetchLimit(this.currentScale, this.currentInterval);
            const response = await fetch(`/api/chart/${normalizedSymbol}/${this.currentInterval}?limit=${limit}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.lastSeries = data.series;
            this.lastIndicators = data.indicators;
            const activeSymbol = (data.symbol || normalizedSymbol || '').toUpperCase();
            const chartTitle = `${activeSymbol} - ${this.getIntervalName(this.currentInterval)}`;
            this.lastChartTitle = chartTitle;

            const sourceKey = (data.source || '').toLowerCase();
            if (sourceKey && sourceKey !== this.lastDataSource) {
                if (sourceKey !== 'yfinance') {
                    this.showToast(`Dados alternativos (${sourceKey}) carregados para ${activeSymbol}.`, 'warning');
                }
                this.lastDataSource = sourceKey;
            }

            this.renderSeriesChart(this.lastSeries, this.lastIndicators, 'main-chart', chartTitle, this.currentInterval, this.currentScale);
            this.renderSeriesChart(this.lastSeries, this.lastIndicators, 'analysis-chart', chartTitle, this.currentInterval, this.currentScale);
            this.updateIndicators(this.lastIndicators);
            if (Array.isArray(data.candles) && data.candles.length > 0) {
                this.updateCurrentPrice(data.candles[data.candles.length - 1]);
            }

            // Update labels with normalized símbolo
            this.currentSymbol = activeSymbol;
            document.getElementById('current-symbol').textContent = activeSymbol;
            const chartSymbolInput = document.getElementById('chart-symbol-input');
            if (chartSymbolInput) {
                chartSymbolInput.value = activeSymbol;
            }
            document.getElementById('chart-title').textContent = chartTitle;

            this.updateSymbolButtons(activeSymbol);
            this.syncSymbolInputs(activeSymbol);
            this.subscribeToSymbol(activeSymbol);

            if (this.socket) {
                this.socket.emit('request_chart', {
                    symbol: activeSymbol,
                    interval: this.currentInterval,
                    limit
                });
            }
                
        } catch (error) {
            console.error('Erro ao carregar gráfico:', error);
            this.showToast('Erro ao carregar gráfico', 'danger');
        }
    }
    
    renderSeriesChart(series, indicators, containerId, title, interval = this.currentInterval, scale = this.currentScale) {
        if (!series || !series.open || !series.open.length) {
            console.warn('Série vazia para renderização do gráfico', series);
            return;
        }

        const container = document.getElementById(containerId);
        if (!container) {
            return;
        }

        // Build a per-candle datetime axis.
        // NEVER fall back to numeric indexes for a date axis (it turns into 1970 in Plotly).
        const openLen = series.open.length;
        const rawTimeStr = Array.isArray(series.time_str) ? series.time_str : [];
        const rawTime = Array.isArray(series.time) ? series.time : [];

        const resolvedXAxis = new Array(openLen);
        const validMask = new Array(openLen).fill(false);
        for (let i = 0; i < openLen; i++) {
            const candidateStr = i < rawTimeStr.length ? rawTimeStr[i] : null;
            const candidateIso = i < rawTime.length ? rawTime[i] : null;
            let parsed = this.parseDateValue(candidateStr);
            if (!parsed) {
                parsed = this.parseDateValue(candidateIso);
            }
            if (parsed) {
                resolvedXAxis[i] = parsed;
                validMask[i] = true;
            } else {
                resolvedXAxis[i] = null;
            }
        }

        const scaleDurationMs = this.getScaleDurationMs(scale);
        if (scaleDurationMs) {
            let lastValidIndex = -1;
            for (let i = openLen - 1; i >= 0; i--) {
                if (validMask[i]) {
                    lastValidIndex = i;
                    break;
                }
            }
            if (lastValidIndex >= 0) {
                const lastDate = resolvedXAxis[lastValidIndex];
                if (lastDate instanceof Date && !Number.isNaN(lastDate.valueOf())) {
                    const cutoff = new Date(lastDate.getTime() - scaleDurationMs);
                    for (let i = 0; i < openLen; i++) {
                        if (validMask[i] && resolvedXAxis[i] instanceof Date && resolvedXAxis[i] < cutoff) {
                            validMask[i] = false;
                        }
                    }
                }
            }
        }

        const validCount = validMask.reduce((acc, ok) => acc + (ok ? 1 : 0), 0);
        if (validCount < 2) {
            console.warn('Não foi possível resolver datas válidas para o eixo X', {
                interval,
                openLen,
                rawTimeLen: rawTime.length,
                rawTimeStrLen: rawTimeStr.length,
                sampleTime: rawTime[0],
                sampleTimeStr: rawTimeStr[0]
            });
            this.showToast('Erro ao interpretar datas do gráfico (eixo X).', 'warning');
            return;
        }

        // Filter all series arrays to keep only points that have a valid datetime.
        const filterByMask = (arr) => {
            if (!Array.isArray(arr) || arr.length !== openLen) {
                return arr;
            }
            const out = [];
            for (let i = 0; i < openLen; i++) {
                if (validMask[i]) {
                    out.push(arr[i]);
                }
            }
            return out;
        };

        const xFiltered = filterByMask(resolvedXAxis);
        const openFiltered = filterByMask(series.open);
        const highFiltered = filterByMask(series.high);
        const lowFiltered = filterByMask(series.low);
        const closeFiltered = filterByMask(series.close);

        const lastDate = xFiltered.length ? xFiltered[xFiltered.length - 1] : null;
        let rangeStart = null;
        if (lastDate instanceof Date && scaleDurationMs) {
            const startCandidate = new Date(lastDate.getTime() - scaleDurationMs);
            if (xFiltered.length) {
                const firstDate = xFiltered[0];
                if (firstDate instanceof Date && firstDate > startCandidate) {
                    rangeStart = firstDate;
                } else {
                    rangeStart = startCandidate;
                }
            }
        }

        const candlestickTrace = {
            type: 'candlestick',
            x: xFiltered,
            open: openFiltered,
            high: highFiltered,
            low: lowFiltered,
            close: closeFiltered,
            name: 'Preço',
            increasing: {
                line: { color: '#26a69a' },
                fillcolor: '#26a69a'
            },
            decreasing: {
                line: { color: '#ef5350' },
                fillcolor: '#ef5350'
            },
            whiskerwidth: 0.8,
            hoverinfo: 'skip',
            hovertemplate: [
                '<b>%{x|%d/%m/%Y %H:%M}</b><br>',
                'Abertura: R$ %{open:.2f}<br>',
                'Máxima: R$ %{high:.2f}<br>',
                'Mínima: R$ %{low:.2f}<br>',
                'Fechamento: R$ %{close:.2f}<extra></extra>'
            ].join('')
        };

        const traces = [candlestickTrace];

        if (Array.isArray(series.volume) && series.volume.length === openLen) {
            const volumeFiltered = filterByMask(series.volume);
            traces.push({
                type: 'bar',
                x: xFiltered,
                y: volumeFiltered,
                name: 'Volume',
                marker: {
                    color: closeFiltered.map((close, idx) => {
                        const open = openFiltered[idx];
                        return close >= open ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)';
                    })
                },
                opacity: 0.7,
                yaxis: 'y2',
                hoverinfo: 'skip',
                hovertemplate: [
                    '<b>%{x|%d/%m/%Y %H:%M}</b><br>',
                    'Volume: %{y:,0f}<extra></extra>'
                ].join('')
            });
        }

        if (Array.isArray(series.sma_9) && series.sma_9.length === openLen) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.sma_9),
                name: 'SMA 9',
                line: { color: '#2196F3', width: 1.5 },
                opacity: 0.7,
                hoverinfo: 'skip',
                hovertemplate: 'SMA 9: R$ %{y:.2f}<extra></extra>'
            });
        }

        if (Array.isArray(series.sma_21) && series.sma_21.length === openLen) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.sma_21),
                name: 'SMA 21',
                line: { color: '#FF9800', width: 1.5 },
                opacity: 0.7,
                hoverinfo: 'skip',
                hovertemplate: 'SMA 21: R$ %{y:.2f}<extra></extra>'
            });
        }

        if (Array.isArray(series.ema_21) && series.ema_21.length === openLen) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.ema_21),
                name: 'MME 21',
                line: { color: '#1E88E5', width: 1.6, dash: 'dot' },
                opacity: 0.95,
                hoverinfo: 'skip',
                hovertemplate: 'MME 21: R$ %{y:.2f}<extra></extra>'
            });
        }

        if (Array.isArray(series.ema_200) && series.ema_200.length === openLen) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.ema_200),
                name: 'MME 200',
                line: { color: '#D32F2F', width: 1.8, dash: 'dot' },
                opacity: 0.95,
                hoverinfo: 'skip',
                hovertemplate: 'MME 200: R$ %{y:.2f}<extra></extra>'
            });
        }

        if (
            Array.isArray(series.bb_upper) && series.bb_upper.length === openLen &&
            Array.isArray(series.bb_middle) && series.bb_middle.length === openLen &&
            Array.isArray(series.bb_lower) && series.bb_lower.length === openLen
        ) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.bb_upper),
                name: 'BB Superior',
                line: { color: 'rgba(158, 158, 158, 0.5)', width: 1, dash: 'dash' },
                showlegend: false,
                hoverinfo: 'skip'
            });
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: xFiltered,
                y: filterByMask(series.bb_lower),
                name: 'BB Inferior',
                line: { color: 'rgba(158, 158, 158, 0.5)', width: 1, dash: 'dash' },
                fill: 'tonexty',
                fillcolor: 'rgba(158, 158, 158, 0.1)',
                showlegend: false,
                hoverinfo: 'skip'
            });
        }

        const layout = {
            title: {
                text: title,
                font: { size: 16, color: '#2c3e50', family: 'Arial Black' },
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: {
                type: 'date',
                gridcolor: '#ecf0f1',
                showgrid: true,
                rangeslider: { visible: false },
                nticks: 15,
                tickfont: { size: 10 },
                rangeselector: {
                    buttons: [
                        { count: 1, label: '1D', step: 'day', stepmode: 'backward' },
                        { count: 5, label: '5D', step: 'day', stepmode: 'backward' },
                        { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
                        { step: 'all', label: 'Tudo' }
                    ],
                    bgcolor: 'rgba(255, 255, 255, 0.9)',
                    activecolor: '#2196F3',
                    x: 0.01,
                    y: 1.05,
                    font: { size: 9 }
                }
            },
            yaxis: {
                title: 'Preço (R$)',
                gridcolor: '#ecf0f1',
                showgrid: true,
                side: 'right',
                tickformat: '.2f',
                tickprefix: 'R$ ',
                tickfont: { size: 10 },
                autorange: true
            },
            yaxis2: {
                title: 'Volume',
                overlaying: 'y',
                side: 'left',
                showgrid: false,
                tickformat: ',.0f',
                tickfont: { size: 9 },
                autorange: true
            },
            height: 550,
            margin: { l: 50, r: 60, t: 70, b: 50 },
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: '#f8f9fa',
            showlegend: true,
            legend: {
                x: 0.01,
                y: 0.99,
                bgcolor: 'rgba(255, 255, 255, 0.95)',
                bordercolor: '#bdc3c7',
                borderwidth: 1,
                font: { size: 9 }
            },
            font: { family: 'Arial, sans-serif' }
        };

        layout.xaxis = {
            ...layout.xaxis,
            autorange: !rangeStart || !lastDate,
        };

        if (rangeStart && lastDate) {
            layout.xaxis.range = [rangeStart, lastDate];
        }

        this.applyAxisFormats(layout, interval);

        if (this.charts[containerId]) {
            Plotly.react(container, traces, layout, this.chartConfig);
        } else {
            Plotly.newPlot(container, traces, layout, this.chartConfig);
            this.charts[containerId] = true;
        }
    }

    parseDateValue(value) {
        if (!value) {
            return null;
        }
        if (value instanceof Date && !Number.isNaN(value.valueOf())) {
            return value;
        }

        const raw = value;
        let parsed = new Date(raw);
        if (!Number.isNaN(parsed.valueOf())) {
            return parsed;
        }

        if (typeof raw === 'string') {
            const isoCandidate = raw.includes('T') ? raw : raw.replace(' ', 'T');
            parsed = new Date(isoCandidate);
            if (!Number.isNaN(parsed.valueOf())) {
                return parsed;
            }

            parsed = new Date(`${isoCandidate}Z`);
            if (!Number.isNaN(parsed.valueOf())) {
                return parsed;
            }
        }

        return null;
    }

    applyAxisFormats(layout, interval) {
        if (!layout || !layout.xaxis) {
            return;
        }

        const normalized = (interval || '').toLowerCase();
        const intraday = ['1m', '5m', '15m', '30m', '1h', '60m', '60min'].includes(normalized);
        const daily = normalized === '1d' || normalized === 'diario';
        const weekly = normalized === '1w' || normalized === 'semanal';

        if (intraday) {
            // Intraday often spans multiple days; showing only hours repeats and becomes ambiguous.
            // Default to day+hour (two-line label) and keep full datetime on hover.
            layout.xaxis.tickformat = '%d/%m<br>%H:%M';
            layout.xaxis.hoverformat = '%d/%m %H:%M';
            layout.xaxis.tickformatstops = [
                { dtickrange: [null, 60 * 60 * 1000], value: '%H:%M' },
                { dtickrange: [60 * 60 * 1000, 24 * 60 * 60 * 1000], value: '%d/%m %H:%M' },
                { dtickrange: [24 * 60 * 60 * 1000, null], value: '%d/%m' }
            ];
            layout.xaxis.rangebreaks = [
                { bounds: ['sat', 'mon'] },
                { bounds: [17, 10], pattern: 'hour' }
            ];
        } else if (daily) {
            layout.xaxis.tickformat = '%d/%m';
            layout.xaxis.hoverformat = '%d/%m/%Y';
            layout.xaxis.tickformatstops = [
                { dtickrange: [null, 7 * 24 * 60 * 60 * 1000], value: '%d/%m' },
                { dtickrange: [7 * 24 * 60 * 60 * 1000, null], value: '%d/%m/%Y' }
            ];
            layout.xaxis.rangebreaks = [
                { bounds: ['sat', 'mon'] }
            ];
        } else if (weekly) {
            layout.xaxis.tickformat = '%d/%m/%Y';
            layout.xaxis.hoverformat = '%d/%m/%Y';
            layout.xaxis.tickformatstops = [
                { dtickrange: [null, null], value: '%d/%m/%Y' }
            ];
            layout.xaxis.rangebreaks = [];
        } else {
            layout.xaxis.tickformat = '%d/%m/%Y';
            layout.xaxis.hoverformat = '%d/%m/%Y';
            layout.xaxis.tickformatstops = [
                { dtickrange: [null, null], value: '%d/%m/%Y' }
            ];
            layout.xaxis.rangebreaks = [];
        }
    }

    resizeChart(containerId) {
        const container = document.getElementById(containerId);
        if (container && this.charts[containerId]) {
            Plotly.Plots.resize(container);
        }
    }

    updateSymbolButtons(symbol) {
        if (!symbol) {
            return;
        }

        const normalized = symbol.toUpperCase();

        document.querySelectorAll('.symbol-toggle').forEach(btn => {
            const btnSymbol = (btn.getAttribute('data-symbol') || '').toUpperCase();
            const isActive = btnSymbol === normalized;
            btn.classList.toggle('active', isActive);
        });

        document.querySelectorAll('#stocks-list .list-group-item[data-symbol]').forEach(item => {
            const itemSymbol = (item.getAttribute('data-symbol') || '').toUpperCase();
            const isActive = itemSymbol === normalized;
            item.classList.toggle('active-symbol', isActive);
        });
    }

    syncSymbolInputs(symbol) {
        const normalized = (symbol || this.currentSymbol || '').toUpperCase();

        const chartInput = document.getElementById('chart-symbol-input');
        if (chartInput && document.activeElement !== chartInput) {
            chartInput.value = normalized;
        }

        const quickSymbolInput = document.getElementById('quick-symbol');
        if (quickSymbolInput && document.activeElement !== quickSymbolInput) {
            quickSymbolInput.value = normalized;
        }

        const operationSymbolInput = document.getElementById('operation-symbol');
        if (operationSymbolInput && !operationSymbolInput.value) {
            operationSymbolInput.value = normalized;
        }
    }

    changeSymbol(symbol) {
        const normalized = (symbol || '').trim().toUpperCase();
        if (!normalized) {
            return;
        }

        if (normalized === this.currentSymbol) {
            this.updateSymbolButtons(normalized);
            this.syncSymbolInputs(normalized);
            return Promise.resolve();
        }

        this.currentSymbol = normalized;
        this.updateSymbolButtons(normalized);
        this.syncSymbolInputs(normalized);
        this.subscribeToSymbol(normalized);

        return this.loadChart(normalized, this.currentInterval);
    }
    
    updateIndicators(indicators) {
        const panel = document.getElementById('indicators-panel');
        const cards = [
            {
                label: 'SMA 9',
                value: indicators.sma_9,
            },
            {
                label: 'SMA 21',
                value: indicators.sma_21,
            },
            {
                label: 'MME 21',
                value: indicators.ema_21,
            },
            {
                label: 'MME 200',
                value: indicators.ema_200,
            },
            {
                label: 'RSI',
                value: indicators.rsi,
                formatter: (val) => (this.isValidNumber(val) ? val.toFixed(2) : 'N/A'),
                extraClass: indicators.rsi > 70 ? 'text-danger' : (indicators.rsi < 30 ? 'text-success' : ''),
            },
            {
                label: 'Trend',
                value: indicators.sma_9 && indicators.sma_21
                    ? (indicators.sma_9 > indicators.sma_21 ? '📈 Alta' : '📉 Baixa')
                    : 'N/A',
                formatter: (val) => val,
            }
        ];

        panel.innerHTML = cards.map((card) => {
            const formatter = card.formatter || ((val) => this.formatIndicator(val));
            const displayValue = formatter(card.value);
            const extraClass = card.extraClass ? ` ${card.extraClass}` : '';
            return `
                <div class="col-xl-2 col-lg-3 col-md-4 col-sm-6 mb-3">
                    <div class="card bg-light h-100">
                        <div class="card-body text-center">
                            <small class="text-muted">${card.label}</small>
                            <div class="h5${extraClass}">${displayValue}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    formatIndicator(value) {
        if (!this.isValidNumber(value)) {
            return 'N/A';
        }
        return Number(value).toFixed(2);
    }

    isValidNumber(value) {
        return value !== null && value !== undefined && !Number.isNaN(value);
    }

    buildPdfUrl(path) {
        if (!path) return null;
        const filename = String(path).split(/[\\\/]/).pop();
        if (!filename) return null;
        return `/reports/${filename}`;
    }
    
    updateCurrentPrice(candle) {
        document.getElementById('current-price').textContent = 
            `R$ ${candle.close.toFixed(2)}`;
        document.getElementById('current-low').textContent = 
            `R$ ${candle.low.toFixed(2)}`;
        document.getElementById('current-high').textContent = 
            `R$ ${candle.high.toFixed(2)}`;
    }
    
    async loadOperationsHistory() {
        try {
            const response = await fetch('/api/history');
            this.operations = await response.json();
            
            this.renderOperationsTable();
            this.updateStatistics();
            
        } catch (error) {
            console.error('Erro ao carregar histórico:', error);
            this.showToast('Erro ao carregar histórico de operações', 'danger');
        }
    }
    
    renderOperationsTable() {
        const tbody = document.getElementById('history-table-body');
        tbody.innerHTML = '';

        const cardsContainer = document.getElementById('history-cards');
        if (cardsContainer) {
            cardsContainer.innerHTML = '';
        }
        
        this.operations.forEach(op => {
            const row = document.createElement('tr');
            const statusClass = (op.status || '').toLowerCase().replace(/\s+/g, '-');
            row.className = `operation-card ${statusClass}`;

            const pdfUrl = this.buildPdfUrl(op.pdf_url || op.pdf_path);

            const symbol = op.symbol || '-';
            const faixa = (op.entrada_min !== null && op.entrada_min !== undefined &&
                           op.entrada_max !== null && op.entrada_max !== undefined)
                ? `${this.formatCurrency(Math.min(op.entrada_min, op.entrada_max))} · ${this.formatCurrency(Math.max(op.entrada_min, op.entrada_max))}`
                : '-';

            const parcialInfo = (op.parcial_preco !== null && op.parcial_preco !== undefined)
                ? `${this.formatCurrency(op.parcial_preco)} | ${this.formatTicks(op.parcial_pontos)}`
                : '-';

            const alvoInfo = `${this.formatCurrency(op.alvo)} | ${this.formatTicks(op.pontos_alvo)}`;
            const stopInfo = `${this.formatCurrency(op.stop)} | ${this.formatTicks(op.pontos_stop)}`;
            const createdAt = op.created_at ? new Date(op.created_at).toLocaleDateString('pt-BR') : '-';
            const pdfIcon = pdfUrl ? '<i class="fas fa-file-pdf ms-1 text-danger" title="Possui PDF"></i>' : '';

            row.innerHTML = `
                <td>${op.id}</td>
                <td>
                    <strong>${symbol}</strong>
                    ${pdfIcon}
                </td>
                <td>
                    <span class="badge ${(op.tipo || '').toUpperCase() === 'COMPRA' ? 'bg-success' : 'bg-danger'}">
                        ${(op.tipo || '').toUpperCase()}
                    </span>
                </td>
                <td>${this.formatCurrency(op.entrada)}</td>
                <td>${faixa}</td>
                <td>${parcialInfo}</td>
                <td>${alvoInfo}</td>
                <td>${stopInfo}</td>
                <td>${op.quantidade || 0}</td>
                <td>
                    <span class="badge ${this.getStatusBadgeClass(op.status)}">
                        ${op.status || 'ABERTA'}
                    </span>
                </td>
                <td>${createdAt}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-operation" data-id="${op.id}">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${pdfUrl ? `
                    <a href="${pdfUrl}" class="btn btn-sm btn-outline-danger" target="_blank">
                        <i class="fas fa-file-pdf"></i>
                    </a>
                    ` : ''}
                </td>
            `;
            tbody.appendChild(row);

            if (cardsContainer) {
                const card = document.createElement('div');
                card.className = 'card shadow-sm';

                const directionBadge = (op.tipo || '').toUpperCase() === 'COMPRA'
                    ? '<span class="badge bg-success">COMPRA</span>'
                    : '<span class="badge bg-danger">VENDA</span>';
                const statusBadge = `<span class="badge ${this.getStatusBadgeClass(op.status)}">${op.status || 'ABERTA'}</span>`;
                const createdAt = op.created_at ? new Date(op.created_at).toLocaleDateString('pt-BR') : '-';

                card.innerHTML = `
                    <div class="card-body p-3">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <div>
                                <div class="fw-semibold">${symbol} ${pdfIcon}</div>
                                <div class="text-muted small">ID ${op.id} • ${createdAt}</div>
                            </div>
                            <div class="d-flex flex-column align-items-end gap-1">
                                ${directionBadge}
                                ${statusBadge}
                            </div>
                        </div>

                        <div class="row g-2 small">
                            <div class="col-6">
                                <div class="text-muted">Entrada</div>
                                <div class="fw-semibold">${this.formatCurrency(op.entrada)}</div>
                            </div>
                            <div class="col-6">
                                <div class="text-muted">Faixa</div>
                                <div class="fw-semibold">${faixa}</div>
                            </div>
                            <div class="col-6">
                                <div class="text-muted">Alvo</div>
                                <div class="fw-semibold">${alvoInfo}</div>
                            </div>
                            <div class="col-6">
                                <div class="text-muted">Stop</div>
                                <div class="fw-semibold">${stopInfo}</div>
                            </div>
                            <div class="col-6">
                                <div class="text-muted">Parcial</div>
                                <div class="fw-semibold">${parcialInfo}</div>
                            </div>
                            <div class="col-6">
                                <div class="text-muted">Qtd</div>
                                <div class="fw-semibold">${op.quantidade || 0}</div>
                            </div>
                        </div>

                        <div class="d-flex justify-content-end gap-2 mt-3">
                            <button class="btn btn-sm btn-outline-primary view-operation" data-id="${op.id}">
                                <i class="fas fa-eye"></i>
                            </button>
                            ${pdfUrl ? `
                            <a href="${pdfUrl}" class="btn btn-sm btn-outline-danger" target="_blank">
                                <i class="fas fa-file-pdf"></i>
                            </a>
                            ` : ''}
                        </div>
                    </div>
                `;

                cardsContainer.appendChild(card);
            }
        });
        
        // Add event listeners to view buttons
        document.querySelectorAll('.view-operation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const opId = e.currentTarget.getAttribute('data-id');
                this.showOperationDetails(opId);
            });
        });

        if (cardsContainer && this.operations.length === 0) {
            cardsContainer.innerHTML = `
                <div class="alert alert-secondary mb-0">Nenhuma operação encontrada. Toque em "Atualizar" para recarregar.</div>
            `;
        }
    }
    
    async loadRecentOperations() {
        const container = document.getElementById('recent-operations');
        container.innerHTML = '';
        
        const recentOps = this.operations.slice(0, 5);
        
        recentOps.forEach(op => {
            const div = document.createElement('div');
            div.className = 'list-group-item list-group-item-action';
            div.innerHTML = `
                <div class="d-flex w-100 justify-content-between">
                    <div>
                        <strong>${op.symbol}</strong>
                        <div class="small ${op.tipo === 'COMPRA' ? 'text-success' : 'text-danger'}">
                            ${op.tipo}
                        </div>
                    </div>
                    <div class="text-end">
                        <div>R$ ${op.entrada.toFixed(2)}</div>
                        <span class="badge ${this.getStatusBadgeClass(op.status)}">
                            ${op.status}
                        </span>
                    </div>
                </div>
            `;
            container.appendChild(div);
        });
    }
    
    updateStatistics() {
        // Função auxiliar para identificar Swing Trade
        function isSwing(op) {
            // Novo critério: se vier do backend como 'swing_trade', é swing
            if (op.source === 'swing_trade') return true;
            // Critérios antigos para compatibilidade
            const tradeType = (op.trade_type || op.tipo_operacao || '').toLowerCase();
            const tfMajor = (op.timeframe_major || op.timeframe || '').toLowerCase();
            if (op.is_swing === true || String(op.is_swing).toLowerCase() === 'true') return true;
            if (tradeType.startsWith('swing')) return true;
            if (tradeType.startsWith('day')) return false;
            if (["diario","daily","1d","semanal","weekly","1w","d","w"].includes(tfMajor)) return true;
            if (String(op.trade_type || op.tipo_operacao || '').toLowerCase().includes('swing')) return true;
            return false;
        }

        // Filtro de operações da semana atual
        function isThisWeek(op) {
            if (!op.created_at) return false;
            const opDate = new Date(op.created_at);
            const now = new Date();
            // Início da semana (segunda-feira)
            const weekStart = new Date(now);
            weekStart.setHours(0,0,0,0);
            weekStart.setDate(now.getDate() - ((now.getDay() + 6) % 7));
            // Fim da semana (domingo)
            const weekEnd = new Date(weekStart);
            weekEnd.setDate(weekStart.getDate() + 6);
            weekEnd.setHours(23,59,59,999);
            return opDate >= weekStart && opDate <= weekEnd;
        }

        // Separa operações da semana
        const swingOps = this.operations.filter(op => isSwing(op) && isThisWeek(op));
        const dayOps = this.operations.filter(op => !isSwing(op) && isThisWeek(op));

        // Função para calcular estatísticas
        function calcStats(ops) {
            const total = ops.length;
            const success = ops.filter(op => (op.status || '').includes('ALVO')).length;
            const stops = ops.filter(op => (op.status || '').includes('STOP')).length;
            const closed = success + stops;
            const winRate = closed > 0 ? ((success / closed) * 100).toFixed(1) : 0;
            let profit = 0;
            ops.forEach(op => {
                const qty = Number(op.quantidade) || 0;
                if (!qty) return;
                const entrada = Number(op.entrada) || 0;
                const alvo = Number(op.alvo) || 0;
                const stop = Number(op.stop) || 0;
                const tipo = (op.tipo || 'COMPRA').toUpperCase();
                const targetDiff = tipo === 'COMPRA' ? (alvo - entrada) : (entrada - alvo);
                const stopDiff = tipo === 'COMPRA' ? (entrada - stop) : (stop - entrada);
                if ((op.status || '').includes('ALVO')) {
                    profit += Math.max(targetDiff, 0) * qty;
                } else if ((op.status || '').includes('STOP')) {
                    profit -= Math.max(stopDiff, 0) * qty;
                }
            });
            const assets = new Set(ops.map(op => op.symbol)).size;
            return { total, profit, winRate, assets };
        }

        // Calcula estatísticas
        const swingStats = calcStats(swingOps);
        const dayStats = calcStats(dayOps);

        // Atualiza blocos Swing
        document.getElementById('stats-operations-swing').textContent = swingStats.total;
        document.getElementById('stats-profit-swing').textContent = this.formatCurrency(swingStats.profit);
        document.getElementById('stats-winrate-swing').textContent = `${swingStats.winRate}%`;
        document.getElementById('stats-assets-swing').textContent = swingStats.assets;

        // Atualiza blocos Day
        document.getElementById('stats-operations-day').textContent = dayStats.total;
        document.getElementById('stats-profit-day').textContent = this.formatCurrency(dayStats.profit);
        document.getElementById('stats-winrate-day').textContent = `${dayStats.winRate}%`;
        document.getElementById('stats-assets-day').textContent = dayStats.assets;
    }
    
    getStatusBadgeClass(status) {
        const normalized = (status || '').toUpperCase();
        if (normalized.includes('ALVO')) return 'bg-success';
        if (normalized.includes('STOP')) return 'bg-danger';
        if (normalized === 'ABERTA') return 'bg-warning';
        return 'bg-secondary';
    }
    
    async registerOperation(operationData) {
        try {
            const response = await fetch('/api/operacao', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(operationData)
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            this.showToast('Operação registrada com sucesso! PDF gerado.', 'success');
            
            // Reload history
            await this.loadOperationsHistory();
            
            // Show PDF link
            if (result.pdf_url) {
                setTimeout(() => {
                    if (confirm('Deseja visualizar o PDF da operação?')) {
                        window.open(result.pdf_url, '_blank');
                    }
                }, 1000);
            }
            
            return result;
            
        } catch (error) {
            console.error('Erro ao registrar operação:', error);
            this.showToast(`Erro: ${error.message}`, 'danger');
            throw error;
        }
    }
    
    async searchSymbol(query) {
        try {
            const response = await fetch(`/api/search/${query}`);
            const results = await response.json();
            
            return results;
        } catch (error) {
            console.error('Erro na busca:', error);
            return [];
        }
    }
    
    subscribeToSymbol(symbol) {
        const normalized = (symbol || '').trim().toUpperCase();
        if (!normalized) {
            return;
        }

        if (!this.subscribedSymbols.has(normalized)) {
            this.subscribedSymbols.add(normalized);
        }

        if (this.socket && this.socket.connected && !this.socketSubscriptions.has(normalized)) {
            this.socket.emit('subscribe', { symbol: normalized });
            this.socketSubscriptions.add(normalized);
        }
    }
    
    unsubscribeFromSymbol(symbol) {
        const normalized = (symbol || '').trim().toUpperCase();
        if (!normalized) {
            return;
        }

        if (this.subscribedSymbols.has(normalized)) {
            if (this.socket && this.socket.connected) {
                this.socket.emit('unsubscribe', { symbol: normalized });
            }
            this.subscribedSymbols.delete(normalized);
            this.socketSubscriptions.delete(normalized);
        }
    }
    
    handlePriceUpdate(data) {
        const incomingSymbol = (data.symbol || '').toUpperCase();
        const currentSymbol = (this.currentSymbol || '').toUpperCase();
        // Update price displays
        document.querySelectorAll(`.price-display[data-symbol="${incomingSymbol}"]`).forEach(el => {
            const oldPrice = parseFloat(el.textContent.replace('R$ ', ''));
            const newPrice = data.data.price;
            
            el.textContent = `R$ ${newPrice.toFixed(2)}`;
            
            // Add animation class
            if (newPrice > oldPrice) {
                el.classList.add('positive', 'pulse');
            } else if (newPrice < oldPrice) {
                el.classList.add('negative', 'pulse');
            }
            
            setTimeout(() => {
                el.classList.remove('positive', 'negative', 'pulse');
            }, 1000);
        });
        
        // Update current symbol if it's the one being viewed
        if (incomingSymbol === currentSymbol) {
            document.getElementById('current-price').textContent = 
                `R$ ${data.data.price.toFixed(2)}`;
            document.getElementById('price-change').textContent = 
                `${data.data.change >= 0 ? '+' : ''}${data.data.change_percent.toFixed(2)}%`;
            document.getElementById('price-change').className = 
                `badge ${data.data.change >= 0 ? 'bg-success' : 'bg-danger'}`;
        }
    }
    
    handleChartData(data) {
        // Update chart if it's for the current symbol and interval
        const incomingSymbol = (data.symbol || '').toUpperCase();
        const currentSymbol = (this.currentSymbol || '').toUpperCase();
        if (incomingSymbol === currentSymbol && data.interval === this.currentInterval) {
            if (!data.series) {
                console.warn('Atualização de chart sem série recebida', data);
                return;
            }

            if (data.interval && data.interval !== this.currentInterval) {
                this.currentInterval = data.interval;
            }

            this.updateIntervalButtons();

            this.lastSeries = data.series;
            this.lastIndicators = data.indicators || this.lastIndicators;
            const title = `${data.symbol} - ${this.getIntervalName(data.interval)}`;
            this.lastChartTitle = title;

            const sourceKey = (data.source || '').toLowerCase();
            if (sourceKey && sourceKey !== this.lastDataSource) {
                if (sourceKey !== 'yfinance') {
                    this.showToast(`Dados alternativos (${sourceKey}) carregados para ${data.symbol}.`, 'warning');
                }
                this.lastDataSource = sourceKey;
            }

            this.renderSeriesChart(this.lastSeries, this.lastIndicators, 'analysis-chart', title, data.interval, this.currentScale);
            this.renderSeriesChart(this.lastSeries, this.lastIndicators, 'main-chart', title, data.interval, this.currentScale);
            if (this.lastIndicators) {
                this.updateIndicators(this.lastIndicators);
            }
            this.resizeChart('analysis-chart');
            this.resizeChart('main-chart');
            this.updateSymbolButtons(data.symbol);
            this.syncSymbolInputs(data.symbol);
        }
    }

    handleSectionChange(section) {
        if (section === 'chart') {
            this.updateSymbolButtons(this.currentSymbol);
            this.syncSymbolInputs(this.currentSymbol);
            if (this.lastSeries) {
                this.renderSeriesChart(this.lastSeries, this.lastIndicators, 'analysis-chart', this.lastChartTitle, this.currentInterval, this.currentScale);
                this.resizeChart('analysis-chart');
            } else {
                this.loadChart(this.currentSymbol, this.currentInterval);
            }
            
            if (this.socket) {
                const limit = this.resolveFetchLimit(this.currentScale, this.currentInterval);
                this.socket.emit('request_chart', {
                    symbol: this.currentSymbol,
                    interval: this.currentInterval,
                    limit
                });
            }

            const chartTitleEl = document.getElementById('chart-title');
            if (chartTitleEl) {
                chartTitleEl.textContent = this.lastChartTitle || `${this.currentSymbol} - ${this.getIntervalName(this.currentInterval)}`;
            }
        }
    }
    
    showOperationDetails(operationId) {
        const operation = this.operations.find(op => op.id == operationId);
        if (!operation) return;
        
        const parseNumber = (value) => {
            if (value === null || value === undefined) {
                return null;
            }
            const num = Number(value);
            return Number.isFinite(num) ? num : null;
        };

        const entradaValue = parseNumber(operation.entrada);
        const stopValue = parseNumber(operation.stop);
        const alvoValue = parseNumber(operation.alvo);
        const entradaMinValue = parseNumber(operation.entrada_min);
        const entradaMaxValue = parseNumber(operation.entrada_max);
        const parcialValue = parseNumber(operation.parcial_preco);
        const ticksParcialValue = parseNumber(operation.parcial_pontos);
        const ticksAlvoValue = parseNumber(operation.pontos_alvo);
        const ticksStopValue = parseNumber(operation.pontos_stop);
        const tickSizeValue = parseNumber(operation.tick_size);
        const riscoRetornoValue = parseNumber(operation.risco_retorno);
        const qtyValue = parseNumber(operation.quantidade);
        const qtyNumber = qtyValue !== null ? qtyValue : 0;
        const precoAtualValue = parseNumber(operation.preco_atual);

        const faixaEntrada = (entradaMinValue !== null && entradaMaxValue !== null)
            ? `${this.formatCurrency(Math.min(entradaMinValue, entradaMaxValue))} · ${this.formatCurrency(Math.max(entradaMinValue, entradaMaxValue))}`
            : '-';
        const parcialDisplay = parcialValue !== null
            ? `${this.formatCurrency(parcialValue)} | ${ticksParcialValue !== null ? `${ticksParcialValue.toFixed(1)} ticks` : '-'}`
            : '-';
        const alvoDisplay = `${this.formatCurrency(alvoValue)} | ${ticksAlvoValue !== null ? `${ticksAlvoValue.toFixed(1)} ticks` : '-'}`;
        const stopDisplay = `${this.formatCurrency(stopValue)} | ${ticksStopValue !== null ? `${ticksStopValue.toFixed(1)} ticks` : '-'}`;
        const tickSizeDisplay = tickSizeValue !== null ? `${tickSizeValue.toFixed(2)} pontos` : '-';
        const riskRewardText = riscoRetornoValue !== null ? `${riscoRetornoValue.toFixed(2)}:1` : '-';
        const nocionalValue = (entradaValue !== null && qtyValue !== null) ? entradaValue * qtyNumber : null;
        const riskTotalValue = (ticksStopValue !== null && tickSizeValue !== null && qtyValue !== null)
            ? ticksStopValue * tickSizeValue * qtyNumber
            : null;
        const potentialValue = (ticksAlvoValue !== null && tickSizeValue !== null && qtyValue !== null)
            ? ticksAlvoValue * tickSizeValue * qtyNumber
            : null;
        const nocionalDisplay = nocionalValue !== null ? this.formatCurrency(nocionalValue) : 'R$ -';
        const riskTotalDisplay = riskTotalValue !== null ? this.formatCurrency(riskTotalValue) : 'R$ -';
        const potentialDisplay = potentialValue !== null ? this.formatCurrency(potentialValue) : 'R$ -';
        const statusText = operation.status || 'ABERTA';
        const createdAtText = operation.created_at ? new Date(operation.created_at).toLocaleString('pt-BR') : '-';
        const directionText = (operation.tipo || 'COMPRA').toUpperCase();
        const directionClass = directionText === 'COMPRA' ? 'bg-success' : 'bg-danger';

        const modalBody = document.getElementById('operation-details');
        modalBody.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Execução</h6>
                    <table class="table table-sm">
                        <tr><th>Ativo:</th><td>${operation.symbol || '-'}</td></tr>
                        <tr><th>Direção:</th><td><span class="badge ${directionClass}">${directionText}</span></td></tr>
                        <tr><th>Timeframe:</th><td>${(operation.timeframe || '').toUpperCase() || '-'}</td></tr>
                        <tr><th>Faixa de entrada:</th><td>${faixaEntrada}</td></tr>
                        <tr><th>Entrada guia:</th><td>${this.formatCurrency(entradaValue)}</td></tr>
                        <tr><th>Saída parcial:</th><td>${parcialDisplay}</td></tr>
                        <tr><th>Alvo final:</th><td>${alvoDisplay}</td></tr>
                        <tr><th>Stop loss:</th><td>${stopDisplay}</td></tr>
                        <tr><th>Tick size:</th><td>${tickSizeDisplay}</td></tr>
                        <tr><th>Risco/Retorno:</th><td>${riskRewardText}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Métricas</h6>
                    <table class="table table-sm">
                        <tr><th>Quantidade:</th><td>${qtyValue !== null ? qtyValue : '-'}</td></tr>
                        <tr><th>Ticks parcial:</th><td>${ticksParcialValue !== null ? ticksParcialValue.toFixed(1) : '-'}</td></tr>
                        <tr><th>Ticks alvo:</th><td>${ticksAlvoValue !== null ? ticksAlvoValue.toFixed(1) : '-'}</td></tr>
                        <tr><th>Ticks stop:</th><td>${ticksStopValue !== null ? ticksStopValue.toFixed(1) : '-'}</td></tr>
                        <tr><th>Valor nocional:</th><td>${nocionalDisplay}</td></tr>
                        <tr><th>Risco total:</th><td>${riskTotalDisplay}</td></tr>
                        <tr><th>Retorno potencial:</th><td>${potentialDisplay}</td></tr>
                        <tr><th>Preço atual:</th><td>${this.formatCurrency(precoAtualValue)}</td></tr>
                        <tr><th>Status:</th><td><span class="badge ${this.getStatusBadgeClass(statusText)}">${statusText}</span></td></tr>
                        <tr><th>Gerada em:</th><td>${createdAtText}</td></tr>
                    </table>
                </div>
            </div>
            ${operation.observacoes ? `
            <div class="mt-3">
                <h6>Observações</h6>
                <div class="alert alert-light">${operation.observacoes}</div>
            </div>
            ` : ''}
        `;
        
        // Update PDF download link
        if (operation.pdf_path) {
            const pdfBtn = document.getElementById('download-pdf-btn');
            const fileName = operation.pdf_path.split(/[/\\]/).pop();
            pdfBtn.href = fileName ? `/reports/${fileName}` : '#';
            pdfBtn.style.display = 'inline-block';
        } else {
            document.getElementById('download-pdf-btn').style.display = 'none';
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('operationModal'));
        modal.show();
    }
    
    showToast(message, type = 'info') {
        // Create toast element
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');
        
        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to container
        const container = document.getElementById('toast-container');
        container.appendChild(toastEl);
        
        // Initialize and show toast
        const toast = new bootstrap.Toast(toastEl, {
            autohide: true,
            delay: 3000
        });
        toast.show();
        
        // Remove after hide
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    }
    
    getIntervalName(interval) {
        const intervals = {
            '1m': '1 Minuto',
            '15m': '15 Minutos',
            '60m': '60 Minutos',
            '1h': '1 Hora',
            '1d': 'Diário'
        };
        return intervals[interval] || interval;
    }

    getScaleDurationMs(scale) {
        const config = this.scaleConfig[scale];
        if (!config || !Number.isFinite(config.durationMinutes)) {
            return null;
        }
        return config.durationMinutes * 60 * 1000;
    }

    enforceIntervalForScale(scale, requestedInterval, options = {}) {
        const config = this.scaleConfig[scale];
        const fallback = config?.defaultInterval || this.allowedIntervals[0] || requestedInterval;
        const requested = this.allowedIntervals.includes(requestedInterval) ? requestedInterval : fallback;

        if (!config) {
            return { interval: requested, adjusted: false };
        }

        const maxCandles = this.maxCandlesPerRequest;
        const buildPriority = () => {
            const sequence = [];
            if (requested) {
                sequence.push(requested);
            }
            if (config.preferredIntervals) {
                config.preferredIntervals.forEach(item => {
                    if (!sequence.includes(item)) {
                        sequence.push(item);
                    }
                });
            }
            if (config.defaultInterval && !sequence.includes(config.defaultInterval)) {
                sequence.push(config.defaultInterval);
            }
            this.allowedIntervals.forEach(item => {
                if (!sequence.includes(item)) {
                    sequence.push(item);
                }
            });
            return sequence;
        };

        const priorityList = buildPriority();
        for (const candidate of priorityList) {
            if (!this.allowedIntervals.includes(candidate)) {
                continue;
            }
            const minutes = this.intervalDurations[candidate];
            if (!Number.isFinite(minutes) || minutes <= 0) {
                continue;
            }
            const candlesNeeded = Math.ceil(config.durationMinutes / minutes);
            if (candlesNeeded <= maxCandles) {
                const adjusted = candidate !== requested;
                if (adjusted && !options.silent) {
                    this.showToast(
                        `Intervalo ajustado para ${this.getIntervalName(candidate)} para acompanhar a escala selecionada.`,
                        'info'
                    );
                }
                return { interval: candidate, adjusted };
            }
        }

        return { interval: requested, adjusted: false };
    }

    resolveFetchLimit(scale, interval) {
        const config = this.scaleConfig[scale];
        const minutes = this.intervalDurations[interval];
        if (!config || !Number.isFinite(minutes) || minutes <= 0) {
            return 300;
        }
        const candlesNeeded = Math.ceil(config.durationMinutes / minutes) + 20;
        const bounded = Math.min(Math.max(candlesNeeded, 120), this.maxCandlesPerRequest);
        return bounded;
    }

    updateIntervalButtons() {
        const interval = this.currentInterval;
        document.querySelectorAll('.timeframe-btn, .timeframe-btn-chart').forEach(btn => {
            const btnInterval = btn.getAttribute('data-interval');
            if (btnInterval === interval) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    updateScaleButtons() {
        const scale = this.currentScale;
        document.querySelectorAll('.scale-btn, .scale-btn-chart').forEach(btn => {
            const btnScale = btn.getAttribute('data-scale');
            if (btnScale === scale) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
    
    setupEventListeners() {
        // Symbol toggle buttons
        document.querySelectorAll('.symbol-toggle').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const targetSymbol = e.currentTarget.getAttribute('data-symbol');
                this.changeSymbol(targetSymbol);
            });
        });
        
        // Timeframe buttons
        document.querySelectorAll('.timeframe-btn, .timeframe-btn-chart').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const interval = e.currentTarget.getAttribute('data-interval');
                if (!interval) {
                    return;
                }
                const result = this.enforceIntervalForScale(this.currentScale, interval);
                this.currentInterval = result.interval;
                this.updateIntervalButtons();
                this.loadChart(this.currentSymbol, this.currentInterval);
            });
        });

        // Scale buttons
        document.querySelectorAll('.scale-btn, .scale-btn-chart').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const scale = e.currentTarget.getAttribute('data-scale');
                if (!scale) {
                    return;
                }
                this.currentScale = scale;
                const result = this.enforceIntervalForScale(scale, this.currentInterval);
                this.currentInterval = result.interval;
                this.updateScaleButtons();
                this.updateIntervalButtons();
                this.loadChart(this.currentSymbol, this.currentInterval);
            });
        });
        
        // Search button
        document.getElementById('search-btn').addEventListener('click', async () => {
            const query = document.getElementById('search-input').value.trim();
            if (query) {
                const results = await this.searchSymbol(query);
                if (results.length > 0) {
                    this.currentSymbol = results[0].symbol;
                    this.loadChart(this.currentSymbol, this.currentInterval);
                    this.showToast(`Ativo ${results[0].symbol} carregado`, 'success');
                } else {
                    this.showToast('Nenhum ativo encontrado', 'warning');
                }
            }
        });
        
        // Load chart button
        document.getElementById('load-chart-btn').addEventListener('click', () => {
            const symbol = document.getElementById('chart-symbol-input').value.trim();
            if (symbol) {
                this.currentSymbol = symbol;
                this.loadChart(symbol, this.currentInterval);
                this.subscribeToSymbol(symbol);
            }
        });
        
        // Quick trade button
        document.getElementById('quick-trade-btn').addEventListener('click', async () => {
            const symbol = document.getElementById('quick-symbol').value;
            const entradaBase = parseFloat(document.getElementById('quick-entrada').value);
            if (!Number.isFinite(entradaBase)) {
                this.showToast('Informe o preço de entrada para a operação rápida.', 'warning');
                return;
            }
            const tickSizeQuick = this.inferTickSize(symbol);
            const isCompra = document.getElementById('quick-type').value === 'COMPRA';
            const alvoPreco = isCompra ? entradaBase * 1.02 : entradaBase * 0.98;
            const stopPreco = isCompra ? entradaBase * 0.98 : entradaBase * 1.02;
            const parcialPreco = isCompra ? entradaBase + (tickSizeQuick * 10) : entradaBase - (tickSizeQuick * 10);
            const operation = {
                symbol,
                tipo: document.getElementById('quick-type').value,
                entrada: entradaBase,
                entrada_min: entradaBase,
                entrada_max: entradaBase,
                stop: stopPreco,
                alvo: alvoPreco,
                saida_parcial: parcialPreco,
                quantidade: parseInt(document.getElementById('quick-quantidade').value),
                timeframe: '15m',
                tick_size: tickSizeQuick
            };
            
            try {
                await this.registerOperation(operation);
                document.getElementById('quick-entrada').value = '';
            } catch (error) {
                // Error already handled by registerOperation
            }
        });
        
        // Fetch price button - COMMENTED OUT (old Nova Operação section removed)
        /* 
        document.getElementById('fetch-price-btn').addEventListener('click', async () => {
            const symbol = document.getElementById('operation-symbol').value.trim();
            if (symbol) {
                try {
                    const response = await fetch(`/api/quote/${symbol}`);
                    const data = await response.json();
                    
                    const price = Number(data.price) || 0;
                    document.getElementById('operation-entrada').value = price.toFixed(2);
                    document.getElementById('operation-entrada-min').value = price.toFixed(2);
                    document.getElementById('operation-entrada-max').value = price.toFixed(2);
                    document.getElementById('asset-name').textContent = data.name;
                    document.getElementById('operation-stop').value = (price * 0.98).toFixed(2);
                    document.getElementById('operation-alvo').value = (price * 1.02).toFixed(2);
                    document.getElementById('operation-parcial').value = '';
                    const inferredTick = this.inferTickSize(symbol);
                    document.getElementById('operation-tick-size').value = inferredTick;
                    
                    this.calculateOperationSummary();
                    
                } catch (error) {
                    this.showToast('Erro ao buscar preço do ativo', 'danger');
                }
            }
        });
        */
        
        // Operation form submit - COMMENTED OUT (old Nova Operação section removed)
        /* 
        document.getElementById('operation-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const symbol = document.getElementById('operation-symbol').value;
            const tipo = document.getElementById('operation-type').value;
            const entrada = this.sanitizeNumber(document.getElementById('operation-entrada').value);
            const entradaMinRaw = this.sanitizeNumber(document.getElementById('operation-entrada-min').value);
            const entradaMaxRaw = this.sanitizeNumber(document.getElementById('operation-entrada-max').value);
            const stop = this.sanitizeNumber(document.getElementById('operation-stop').value);
            const alvo = this.sanitizeNumber(document.getElementById('operation-alvo').value);
            const parcial = this.sanitizeNumber(document.getElementById('operation-parcial').value);
            const quantidade = parseInt(document.getElementById('operation-quantidade').value, 10) || 0;
            const timeframe = document.getElementById('operation-timeframe').value;
            const tickSizeInput = this.sanitizeNumber(document.getElementById('operation-tick-size').value);
            const tickSize = tickSizeInput !== null ? tickSizeInput : this.inferTickSize(symbol);

            let entradaMin = entradaMinRaw;
            let entradaMax = entradaMaxRaw;
            if (entradaMin === null && entrada !== null) entradaMin = entrada;
            if (entradaMax === null && entrada !== null) entradaMax = entrada;

            if (entrada === null || stop === null || alvo === null || quantidade <= 0) {
                this.showToast('Preencha os campos obrigatórios da operação.', 'warning');
                return;
            }

            const normalizedTickSize = tickSize > 0 ? tickSize : this.inferTickSize(symbol);

            const operation = {
                ativo: symbol,
                tipo,
                entrada,
                entrada_min: entradaMin,
                entrada_max: entradaMax,
                stop,
                alvo,
                saida_parcial: parcial,
                quantidade,
                timeframe,
                tick_size: normalizedTickSize,
                observacoes: document.getElementById('operation-observacoes').value
            };
            
            // Disable submit button
            const submitBtn = document.getElementById('submit-operation-btn');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processando...';
            submitBtn.disabled = true;
            
            try {
                await this.registerOperation(operation);
                
                // Reset form
                e.target.reset();
                document.getElementById('asset-name').textContent = '';
                this.calculateOperationSummary();
                
            } finally {
                // Re-enable submit button
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        });
        */
        
        // Real-time subscription button
        document.getElementById('subscribe-btn').addEventListener('click', () => {
            if (this.subscribedSymbols.has(this.currentSymbol)) {
                this.unsubscribeFromSymbol(this.currentSymbol);
                this.showToast(`Atualização em tempo real desativada para ${this.currentSymbol}`, 'warning');
                document.getElementById('subscribe-btn').innerHTML = 
                    '<i class="fas fa-play me-2"></i>Ativar Atualização';
            } else {
                this.subscribeToSymbol(this.currentSymbol);
                this.showToast(`Atualização em tempo real ativada para ${this.currentSymbol}`, 'success');
                document.getElementById('subscribe-btn').innerHTML = 
                    '<i class="fas fa-stop me-2"></i>Desativar Atualização';
            }
        });
        
        // Refresh history button
        document.getElementById('refresh-history-btn').addEventListener('click', () => {
            this.loadOperationsHistory();
            this.showToast('Histórico atualizado', 'info');
        });
        
        // Auto-calculate operation values - COMMENTED OUT (old Nova Operação section removed)
        /*
        [
            'operation-entrada-min',
            'operation-entrada',
            'operation-entrada-max',
            'operation-stop',
            'operation-parcial',
            'operation-alvo',
            'operation-quantidade',
            'operation-tick-size'
        ].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('input', () => {
                    this.calculateOperationSummary();
                });
            }
        });

        const typeElement = document.getElementById('operation-type');
        if (typeElement) {
            typeElement.addEventListener('change', () => this.calculateOperationSummary());
        }

        const symbolElement = document.getElementById('operation-symbol');
        if (symbolElement) {
            symbolElement.addEventListener('input', () => this.calculateOperationSummary());
        }

        this.calculateOperationSummary();
        */
    }
    
    calculateOperationSummary() {
        const symbol = document.getElementById('operation-symbol').value;
        const tipo = (document.getElementById('operation-type').value || 'COMPRA').toUpperCase();
        const entrada = this.sanitizeNumber(document.getElementById('operation-entrada').value);
        const entradaMin = this.sanitizeNumber(document.getElementById('operation-entrada-min').value);
        const entradaMax = this.sanitizeNumber(document.getElementById('operation-entrada-max').value);
        const stop = this.sanitizeNumber(document.getElementById('operation-stop').value);
        const alvo = this.sanitizeNumber(document.getElementById('operation-alvo').value);
        const parcial = this.sanitizeNumber(document.getElementById('operation-parcial').value);
        let tickSize = this.sanitizeNumber(document.getElementById('operation-tick-size').value);
        if (tickSize === null || tickSize <= 0) {
            tickSize = this.inferTickSize(symbol);
        }

        const quantidade = parseInt(document.getElementById('operation-quantidade').value, 10) || 0;

        let referencia = entrada;
        if (referencia === null && entradaMin !== null && entradaMax !== null) {
            referencia = (entradaMin + entradaMax) / 2;
        }
        if (referencia === null) {
            referencia = 0;
        }

        const ticksStop = this.computeDirectionalTicks(tipo, referencia, stop ?? referencia, tickSize);
        const ticksAlvo = this.computeDirectionalTicks(tipo, referencia, alvo ?? referencia, tickSize);
        const ticksParcial = parcial !== null ? this.computeDirectionalTicks(tipo, referencia, parcial, tickSize) : null;

        const riskReward = ticksStop > 0 ? (ticksAlvo / ticksStop).toFixed(2) : null;
        const totalValue = Number.isFinite(referencia) ? referencia * quantidade : 0;

        document.getElementById('tick-size-preview').textContent = tickSize ? tickSize.toFixed(2) : '0,00';
        document.getElementById('ticks-stop').textContent = ticksStop.toFixed(1);
        document.getElementById('ticks-alvo').textContent = ticksAlvo.toFixed(1);
        document.getElementById('ticks-parcial').textContent = ticksParcial !== null ? ticksParcial.toFixed(1) : '-';
        document.getElementById('risk-reward').textContent = riskReward ? `${riskReward}:1` : '-';
        document.getElementById('total-value').textContent = this.formatCurrency(totalValue);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Carregar notícias do mercado
    loadMarketNews();

    window.tradingApp = new TradingApp();

    // Preencher automaticamente o campo 'Ativo' de Day Trade ao clicar em um atalho de ativo global
    document.getElementById('asset-shortcuts')?.addEventListener('click', function(e) {
        const btn = e.target.closest('button');
        if (btn && btn.classList.contains('asset-shortcut-btn') && document.getElementById('dt-entry-symbol')) {
            document.getElementById('dt-entry-symbol').value = btn.textContent.trim();
        }
    });
});

// Função para carregar notícias do mercado
function loadMarketNews() {
    const container = document.getElementById('market-news');
    if (!container) return;

    fetch('/api/news')
        .then(response => response.json())
        .then(news => {
            if (!news || news.length === 0) {
                container.innerHTML = '<p class="text-muted">Nenhuma notícia encontrada.</p>';
                return;
            }
            container.innerHTML = news.map(item => `
                <div class="alert alert-info mb-2">
                    <small>
                        <strong><a href="${item.link}" target="_blank" class="text-decoration-none">${item.title}</a></strong>
                        ${item.published ? `<br><small class="text-muted">${item.published}</small>` : ''}
                    </small>
                </div>
            `).join('');
        })
        .catch(error => {
            console.error('Erro ao carregar notícias:', error);
            container.innerHTML = '<p class="text-muted">Erro ao carregar notícias.</p>';
        });
}

// Lightweight UI manager for Carteiras Semanais (v2)
class PortfolioUI {
    constructor({ showToast }) {
        this.maxAssets = 5;
        this.assets = [];
        this.showToast = showToast || ((msg, type) => alert(msg));
        this.tbody = document.getElementById('pf-asset-tbody');
        this.countBadge = document.getElementById('pf-asset-count');
        this.historyTbody = document.getElementById('pf-history-tbody');
        this.lastContainer = document.getElementById('pf-last-container');
        this.startInput = document.getElementById('pf-start-date');
        this.endInput = document.getElementById('pf-end-date');
        this.buildBtn = document.getElementById('pf-build-btn');
        this.addBtn = document.getElementById('pf-add-asset');
        this.refreshBtn = document.getElementById('pf-refresh-history');
        this._wireEvents();
        this._setDefaultDates();
        this._addAssetRow();
        this.loadHistory();
    }

    _wireEvents() {
        this.addBtn?.addEventListener('click', () => this._addAssetRow());
        this.buildBtn?.addEventListener('click', () => this.buildPortfolio());
        this.refreshBtn?.addEventListener('click', () => this.loadHistory());
    }

    _setDefaultDates() {
        const today = new Date();
        const day = today.getDay();
        const mondayOffset = (day === 0 ? -6 : 1 - day);
        const monday = new Date(today);
        monday.setDate(today.getDate() + mondayOffset);
        const friday = new Date(monday);
        friday.setDate(monday.getDate() + 4);
        const fmt = (d) => d.toISOString().split('T')[0];
        if (this.startInput && !this.startInput.value) this.startInput.value = fmt(monday);
        if (this.endInput && !this.endInput.value) this.endInput.value = fmt(friday);
    }

    _addAssetRow() {
        if (this.assets.length >= this.maxAssets) {
            this.showToast(`Máximo de ${this.maxAssets} ativos`, 'warning');
            return;
        }
        this.assets.push({ symbol: '', entrada: '', objetivo: '', stop: '', ultimo: '', entrada_maxima: '', entrada_minima: '' });
        this._render();
    }

    _removeAsset(index) {
        this.assets.splice(index, 1);
        this._render();
    }

    _render() {
        if (!this.tbody) return;
        const total = Math.max(this.assets.length, 1);
        this.tbody.innerHTML = this.assets
            .map((asset, idx) => {
                const lastVal = this.formatCurrency(asset.ultimo);
                const percentual = this.formatPercent(this._calcPercentual(asset));
                const retorno = this.formatPercent(this._calcRetorno(asset));
                const risco = this.formatPercent(this._calcRisco(asset));
                const riscoZero = this.formatCurrency(this._calcRiscoZero(asset).preco);
                const entradaMaxVal = this._toNumber(asset.entrada_maxima);
                const entradaMinVal = this._toNumber(asset.entrada_minima);
                const entradaMax = Number.isFinite(entradaMaxVal) ? this.formatCurrency(entradaMaxVal) : '';
                const entradaMin = Number.isFinite(entradaMinVal) ? this.formatCurrency(entradaMinVal) : '';
                const maxPct = this._formatPctFromEntry(asset.entrada, entradaMaxVal);
                const minPct = this._formatPctFromEntry(asset.entrada, entradaMinVal);
                return `
                <tr data-idx="${idx}">
                    <td><input class="form-control form-control-sm pf-symbol" value="${asset.symbol || ''}" placeholder="Ticker"></td>
                    <td><input type="number" step="0.01" class="form-control form-control-sm pf-entry" value="${asset.entrada || ''}"></td>
                    <td><input type="number" step="0.01" class="form-control form-control-sm pf-objetivo" value="${asset.objetivo || ''}"></td>
                    <td><input type="number" step="0.01" class="form-control form-control-sm pf-stop" value="${asset.stop || ''}"></td>
                    <td><input class="form-control form-control-sm pf-ultimo" value="${lastVal}" placeholder="auto" readonly></td>
                    <td><input class="form-control form-control-sm pf-percent" value="${percentual}" readonly></td>
                    <td><input class="form-control form-control-sm pf-retorno" value="${retorno}" readonly></td>
                    <td><input class="form-control form-control-sm pf-risco" value="${risco}" readonly></td>
                    <td><input class="form-control form-control-sm pf-rz-preco" value="${riscoZero}" readonly></td>
                    <td><input type="text" inputmode="decimal" class="form-control form-control-sm pf-entrada-max" value="${entradaMax}" placeholder="R$ 0,00"></td>
                    <td><input class="form-control form-control-sm pf-max-pct" value="${maxPct}" placeholder="auto" readonly></td>
                    <td><input type="text" inputmode="decimal" class="form-control form-control-sm pf-entrada-min" value="${entradaMin}" placeholder="R$ 0,00"></td>
                    <td><input class="form-control form-control-sm pf-min-pct" value="${minPct}" placeholder="auto" readonly></td>
                    <td><button class="btn btn-link text-danger p-0 pf-remove"><i class="fas fa-trash"></i></button></td>
                </tr>`;
            })
            .join('');

        this.tbody.querySelectorAll('tr').forEach((row) => {
            const idx = parseInt(row.getAttribute('data-idx'), 10);
            row.querySelector('.pf-symbol')?.addEventListener('change', (e) => this._handleSymbol(idx, e.target.value));
            ['pf-entry', 'pf-objetivo', 'pf-stop', 'pf-entrada-max', 'pf-entrada-min'].forEach(cls => {
                const el = row.querySelector(`.${cls}`);
                if (!el) return;
                el.addEventListener('input', () => this._updateAssetFromRow(idx, row));
                if (cls === 'pf-entrada-max' || cls === 'pf-entrada-min') {
                    el.addEventListener('focus', () => this._unformatCurrencyInput(el));
                    el.addEventListener('blur', () => this._formatCurrencyInput(el, idx, cls));
                }
            });
            row.querySelector('.pf-remove')?.addEventListener('click', () => this._removeAsset(idx));
        });

        if (this.countBadge) this.countBadge.textContent = `${this.assets.length}/${this.maxAssets} ativos`;
    }

    _updateAssetFromRow(idx, row) {
        const getVal = (cls) => row.querySelector(`.${cls}`)?.value;
        const asset = this.assets[idx];
        asset.entrada = getVal('pf-entry');
        asset.objetivo = getVal('pf-objetivo');
        asset.stop = getVal('pf-stop');
        asset.entrada_maxima = getVal('pf-entrada-max');
        asset.entrada_minima = getVal('pf-entrada-min');
        this._updateComputedRow(row, asset);
    }

    _updateComputedRow(row, asset) {
        const riscoZero = this._calcRiscoZero(asset);
        const percentEl = row.querySelector('.pf-percent');
        const retornoEl = row.querySelector('.pf-retorno');
        const riscoEl = row.querySelector('.pf-risco');
        const rzPrecoEl = row.querySelector('.pf-rz-preco');
        const entradaMaxVal = this._toNumber(asset.entrada_maxima);
        const entradaMinVal = this._toNumber(asset.entrada_minima);
        if (percentEl) percentEl.value = this.formatPercent(this._calcPercentual(asset));
        if (retornoEl) retornoEl.value = this.formatPercent(this._calcRetorno(asset));
        if (riscoEl) riscoEl.value = this.formatPercent(this._calcRisco(asset));
        if (rzPrecoEl) rzPrecoEl.value = this.formatCurrency(riscoZero.preco);
        const maxEl = row.querySelector('.pf-max-pct');
        const minEl = row.querySelector('.pf-min-pct');
        const entradaMaxEl = row.querySelector('.pf-entrada-max');
        const entradaMinEl = row.querySelector('.pf-entrada-min');
        const maxPctVal = this._formatPctFromEntry(asset.entrada, entradaMaxVal);
        const minPctVal = this._formatPctFromEntry(asset.entrada, entradaMinVal);
        if (maxEl) maxEl.value = maxPctVal;
        if (minEl) minEl.value = minPctVal;
        if (entradaMaxEl && document.activeElement !== entradaMaxEl) {
            entradaMaxEl.value = Number.isFinite(entradaMaxVal) ? this.formatCurrency(entradaMaxVal) : '';
        }
        if (entradaMinEl && document.activeElement !== entradaMinEl) {
            entradaMinEl.value = Number.isFinite(entradaMinVal) ? this.formatCurrency(entradaMinVal) : '';
        }
    }

    formatCurrency(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return 'R$ 0,00';
        return num.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL', minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    formatPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return '0,00%';
        return `${num.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
    }

    _formatCurrencyMaybe(value) {
        const num = this._toNumber(value);
        return Number.isFinite(num) ? this.formatCurrency(num) : '-';
    }

    _formatPctFromEntry(entry, target) {
        const e = this._toNumber(entry);
        const t = this._toNumber(target);
        if (!Number.isFinite(e) || !Number.isFinite(t) || e === 0) return '-';
        const pct = ((t / e) - 1) * 100;
        return this.formatPercent(pct);
    }

    _calcPercentual(asset) {
        const e = parseFloat(asset.entrada) || 0;
        const u = parseFloat(asset.ultimo) || 0;
        if (e <= 0 || !Number.isFinite(u)) return 0;
        return ((u - e) / e) * 100;
    }

    async _handleSymbol(idx, symbol) {
        this.assets[idx].symbol = symbol.toUpperCase();
        this._render();
        if (!symbol) return;
        try {
            const res = await fetch(`/api/quote/${symbol.toUpperCase()}`);
            const data = await res.json();
            if (data && typeof data.price === 'number') {
                this.assets[idx].ultimo = Number(data.price);
                this._render();
            }
        } catch (err) {
            console.warn('Falha ao buscar último preço', err);
        }
    }

    _calcRetorno(asset) {
        const e = parseFloat(asset.entrada) || 0;
        const o = parseFloat(asset.objetivo) || 0;
        if (e <= 0) return 0;
        return ((o - e) / e) * 100;
    }

    _calcRisco(asset) {
        const e = parseFloat(asset.entrada) || 0;
        const s = parseFloat(asset.stop) || 0;
        if (e <= 0) return 0;
        return ((e - s) / e) * 100;
    }

    _calcRiscoZero(asset) {
        const e = parseFloat(asset.entrada) || 0;
        const s = parseFloat(asset.stop) || 0;
        if (e <= 0) return { preco: 0, pct: 0 };
        const preco = e + (e - s);
        const pct = ((preco - e) / e) * 100;
        return { preco, pct };
    }

    _unformatCurrencyInput(el) {
        const raw = (el.value || '').replace(/[^0-9,\.]/g, '');
        const normalized = raw.replace(',', '.');
        el.value = normalized;
    }

    _formatCurrencyInput(el, idx, cls) {
        const num = this._toNumber(el.value);
        if (!Number.isFinite(num)) {
            el.value = '';
            return;
        }
        el.value = this.formatCurrency(num);
        if (typeof idx === 'number') {
            if (cls === 'pf-entrada-max') this.assets[idx].entrada_maxima = num;
            if (cls === 'pf-entrada-min') this.assets[idx].entrada_minima = num;
        }
    }

    _unformatPercentInput(el) {
        const raw = (el.value || '').replace(/[^0-9,\.\-]/g, '');
        const normalized = raw.replace(',', '.');
        el.value = normalized;
    }

    // Percent inputs are readonly now; leave formatting helper for future use if needed.
    _formatPercentInput(el, idx) {
        const num = this._toNumber(el.value);
        if (!Number.isFinite(num)) {
            el.value = '';
            return;
        }
        el.value = this.formatPercent(num);
    }

    _toNumber(value) {
        if (value === null || value === undefined) return NaN;
        const clean = String(value).replace(/[^0-9,\.\-]/g, '').replace(',', '.');
        const n = parseFloat(clean);
        return Number.isFinite(n) ? n : NaN;
    }

    async buildPortfolio() {
        if (this.assets.length === 0) {
            this.showToast('Adicione ao menos 1 ativo', 'warning');
            return;
        }
        if (this.assets.length > this.maxAssets) {
            this.showToast(`Máximo de ${this.maxAssets} ativos`, 'warning');
            return;
        }
        const payload = {
            start_date: this.startInput?.value,
            end_date: this.endInput?.value,
            assets: this.assets.map(a => {
                const entradaNum = parseFloat(a.entrada) || 0;
                const entradaMaxPrice = this._toNumber(a.entrada_maxima);
                const entradaMinPrice = this._toNumber(a.entrada_minima);
                return {
                    symbol: a.symbol,
                    entrada: entradaNum,
                    objetivo: parseFloat(a.objetivo) || 0,
                    stop_loss: parseFloat(a.stop) || 0,
                    entrada_maxima: Number.isFinite(entradaMaxPrice) ? entradaMaxPrice : null,
                    entrada_minima: Number.isFinite(entradaMinPrice) ? entradaMinPrice : null,
                };
            }),
        };

        try {
            const res = await fetch('/api/portfolio/v2/build', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) {
                this.showToast(data.error || 'Falha ao construir carteira', 'danger');
                return;
            }
            this.showToast('Carteira construída com sucesso', 'success');
            this._renderPortfolioView(data.portfolio, this.lastContainer);
            this.loadHistory();
        } catch (err) {
            console.error('Erro ao construir carteira', err);
            this.showToast('Erro ao construir carteira', 'danger');
        }
    }

    _renderPortfolioView(portfolio, container) {
        if (!container || !portfolio) return;
        const assets = portfolio.assets || [];
        container.innerHTML = `
            <div class="d-flex justify-content-between align-items-center flex-wrap mb-2">
                <div><strong>Período:</strong> ${portfolio.start_date} a ${portfolio.end_date}</div>
                <span class="badge bg-light text-dark">${assets.length} ativos</span>
            </div>
            <div class="table-responsive">
                <table class="table table-sm align-middle mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Ativo</th><th>Entrada</th><th>Objetivo</th><th>Stop</th><th>Último</th><th>%</th><th>Retorno%</th><th>Risco%</th><th>Risco Zero</th><th>Ent. Máx</th><th>Ent. Máx %</th><th>Ent. Mín</th><th>Ent. Mín %</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${assets.map(a => `
                            <tr>
                                <td>${a.symbol}</td>
                                <td>${this._formatCurrencyMaybe(a.entry)}</td>
                                <td>${this._formatCurrencyMaybe(a.objective)}</td>
                                <td>${this._formatCurrencyMaybe(a.stop_loss)}</td>
                                <td>${this._formatCurrencyMaybe(a.ultimo_preco ?? '-')}</td>
                                <td>${this.formatPercent(a.percentual)}</td>
                                <td>${this.formatPercent(a.retorno_pct)}</td>
                                <td>${this.formatPercent(a.risco_pct)}</td>
                                <td>${this._formatCurrencyMaybe(a.risco_zero_preco)}</td>
                                <td>${this._formatCurrencyMaybe(a.entry_maxima)}</td>
                                <td>${this._formatPctFromEntry(a.entry, a.entry_maxima)}</td>
                                <td>${this._formatCurrencyMaybe(a.entry_minima)}</td>
                                <td>${this._formatPctFromEntry(a.entry, a.entry_minima)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    async loadHistory() {
        if (!this.historyTbody) return;
        try {
            const res = await fetch('/api/portfolio/v2/list');
            const data = await res.json();
            const items = Array.isArray(data) ? data : [];
            this.historyTbody.innerHTML = items.map(item => {
                const assets = item.assets || [];
                const names = assets.map(a => a.symbol).join(', ');
                return `<tr>
                    <td>${item.start_date} a ${item.end_date}</td>
                    <td>${names}</td>
                    <td>${item.created_at || '-'}</td>
                </tr>`;
            }).join('');
            if (items.length > 0 && this.lastContainer) {
                this._renderPortfolioView(items[0], this.lastContainer);
            }
        } catch (err) {
            console.error('Erro ao carregar histórico de carteiras', err);
            this.showToast('Erro ao carregar histórico de carteiras', 'danger');
        }
    }
}