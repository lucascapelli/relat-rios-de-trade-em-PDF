// Trading System Frontend Application
class TradingApp {
    constructor() {
        this.socket = null;
        this.currentSymbol = 'PETR4';
        this.currentInterval = '15m';
        this.charts = {};
        this.chartConfig = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d']
        };
        this.lastChartData = null;
        this.subscribedSymbols = new Set();
        this.operations = [];
        
        this.init();
    }
    
    init() {
        // Initialize Socket.IO
        this.initSocket();
        
        // Load initial data
        this.loadDashboardData();
        this.loadChart(this.currentSymbol, this.currentInterval);
        this.loadOperationsHistory();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Show welcome message
        setTimeout(() => {
            this.showToast('Sistema de trading carregado com sucesso!', 'success');
        }, 1000);
    }
    
    initSocket() {
        // Connect to WebSocket server
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Conectado ao servidor WebSocket');
            this.updateConnectionStatus(true);
            this.showToast('Conectado ao servidor em tempo real', 'success');
            
            // Subscribe to default symbols
            this.subscribeToSymbol(this.currentSymbol);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Desconectado do servidor WebSocket');
            this.updateConnectionStatus(false);
            this.showToast('Conexão perdida. Tentando reconectar...', 'warning');
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
            
        } catch (error) {
            console.error('Erro ao carregar dados do dashboard:', error);
            this.showToast('Erro ao carregar dados do dashboard', 'danger');
        }
    }
    
    createStockItem(stock) {
        const div = document.createElement('div');
        div.className = 'list-group-item list-group-item-action';
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
        return div;
    }
    
    async loadChart(symbol, interval) {
        try {
            const response = await fetch(`/api/chart/${symbol}/${interval}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.lastChartData = data.chart_data;
            this.renderChart(this.lastChartData, 'main-chart');
            this.renderChart(this.lastChartData, 'analysis-chart');
            this.updateIndicators(data.indicators);
            if (Array.isArray(data.candles) && data.candles.length > 0) {
                this.updateCurrentPrice(data.candles[data.candles.length - 1]);
            }
            
            // Update labels with normalized símbolo
            const activeSymbol = data.symbol || symbol;
            this.currentSymbol = activeSymbol;
            document.getElementById('current-symbol').textContent = activeSymbol;
            const chartSymbolInput = document.getElementById('chart-symbol-input');
            if (chartSymbolInput) {
                chartSymbolInput.value = activeSymbol;
            }
            document.getElementById('chart-title').textContent = 
                `${activeSymbol} - ${this.getIntervalName(interval)}`;

            if (this.socket) {
                this.socket.emit('request_chart', {
                    symbol: activeSymbol,
                    interval
                });
            }
                
        } catch (error) {
            console.error('Erro ao carregar gráfico:', error);
            this.showToast('Erro ao carregar gráfico', 'danger');
        }
    }
    
    renderChart(chartData, containerId) {
        if (!chartData) {
            return;
        }
        
        const container = document.getElementById(containerId);
        if (!container) {
            return;
        }

        if (!chartData.data || !chartData.layout) {
            console.warn('Formato do gráfico inválido recebido', chartData);
            return;
        }
        
        if (this.charts[containerId]) {
            Plotly.react(container, chartData.data, chartData.layout, this.chartConfig);
        } else {
            Plotly.newPlot(container, chartData.data, chartData.layout, this.chartConfig);
            this.charts[containerId] = true;
        }
    }

    resizeChart(containerId) {
        const container = document.getElementById(containerId);
        if (container && this.charts[containerId]) {
            Plotly.Plots.resize(container);
        }
    }
    
    updateIndicators(indicators) {
        const panel = document.getElementById('indicators-panel');
        panel.innerHTML = `
            <div class="col-md-3">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <small class="text-muted">SMA 9</small>
                        <div class="h5">${indicators.sma_9 ? indicators.sma_9.toFixed(2) : 'N/A'}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <small class="text-muted">SMA 21</small>
                        <div class="h5">${indicators.sma_21 ? indicators.sma_21.toFixed(2) : 'N/A'}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <small class="text-muted">RSI</small>
                        <div class="h5 ${indicators.rsi > 70 ? 'text-danger' : indicators.rsi < 30 ? 'text-success' : ''}">
                            ${indicators.rsi ? indicators.rsi.toFixed(2) : 'N/A'}
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light">
                    <div class="card-body text-center">
                        <small class="text-muted">Trend</small>
                        <div class="h5">
                            ${indicators.sma_9 && indicators.sma_21 ? 
                                (indicators.sma_9 > indicators.sma_21 ? '📈 Alta' : '📉 Baixa') : 'N/A'}
                        </div>
                    </div>
                </div>
            </div>
        `;
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
        
        this.operations.forEach(op => {
            const row = document.createElement('tr');
            row.className = `operation-card ${op.status.toLowerCase().replace(' ', '-')}`;
            row.innerHTML = `
                <td>${op.id}</td>
                <td>
                    <strong>${op.symbol}</strong>
                    ${op.pdf_path ? 
                        '<i class="fas fa-file-pdf ms-1 text-danger" title="Possui PDF"></i>' : ''}
                </td>
                <td>
                    <span class="badge ${op.tipo === 'COMPRA' ? 'bg-success' : 'bg-danger'}">
                        ${op.tipo}
                    </span>
                </td>
                <td>R$ ${op.entrada.toFixed(2)}</td>
                <td>R$ ${op.stop.toFixed(2)}</td>
                <td>R$ ${op.alvo.toFixed(2)}</td>
                <td>${op.quantidade}</td>
                <td>
                    <span class="badge ${this.getStatusBadgeClass(op.status)}">
                        ${op.status}
                    </span>
                </td>
                <td>${new Date(op.created_at).toLocaleDateString('pt-BR')}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-operation" 
                            data-id="${op.id}">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${op.pdf_path ? `
                    <a href="/reports/${op.pdf_path.split('/').pop()}" 
                       class="btn btn-sm btn-outline-danger" target="_blank">
                        <i class="fas fa-file-pdf"></i>
                    </a>
                    ` : ''}
                </td>
            `;
            tbody.appendChild(row);
        });
        
        // Add event listeners to view buttons
        document.querySelectorAll('.view-operation').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const opId = e.currentTarget.getAttribute('data-id');
                this.showOperationDetails(opId);
            });
        });
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
        const total = this.operations.length;
        const success = this.operations.filter(op => op.status.includes('ALVO')).length;
        const stops = this.operations.filter(op => op.status.includes('STOP')).length;
        const open = this.operations.filter(op => op.status === 'ABERTA').length;
        
        document.getElementById('stats-operations').textContent = total;
        document.getElementById('success-count').textContent = success;
        document.getElementById('stop-count').textContent = stops;
        document.getElementById('open-count').textContent = open;
        
        const winRate = total > 0 ? ((success / (success + stops)) * 100).toFixed(1) : 0;
        document.getElementById('win-rate').textContent = `${winRate}%`;
        document.getElementById('stats-winrate').textContent = `${winRate}%`;
        
        // Calculate profit/loss (simplified)
        let profit = 0;
        this.operations.forEach(op => {
            if (op.status.includes('ALVO')) {
                profit += (op.alvo - op.entrada) * op.quantidade;
            } else if (op.status.includes('STOP')) {
                profit += (op.stop - op.entrada) * op.quantidade;
            }
        });
        
        document.getElementById('stats-profit').textContent = 
            `R$ ${profit.toFixed(2)}`;
        document.getElementById('stats-assets').textContent = 
            new Set(this.operations.map(op => op.symbol)).size;
    }
    
    getStatusBadgeClass(status) {
        if (status.includes('ALVO')) return 'bg-success';
        if (status.includes('STOP')) return 'bg-danger';
        if (status === 'ABERTA') return 'bg-warning';
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
        if (!this.subscribedSymbols.has(symbol)) {
            this.socket.emit('subscribe', { symbol });
            this.subscribedSymbols.add(symbol);
        }
    }
    
    unsubscribeFromSymbol(symbol) {
        if (this.subscribedSymbols.has(symbol)) {
            this.socket.emit('unsubscribe', { symbol });
            this.subscribedSymbols.delete(symbol);
        }
    }
    
    handlePriceUpdate(data) {
        // Update price displays
        document.querySelectorAll(`.price-display[data-symbol="${data.symbol}"]`).forEach(el => {
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
        if (data.symbol === this.currentSymbol) {
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
        if (data.symbol === this.currentSymbol && data.interval === this.currentInterval) {
            const chartPayload = data.chart;
            if (!chartPayload) {
                console.warn('Atualização de chart sem payload recebida', data);
                return;
            }

            this.lastChartData = chartPayload;
            this.renderChart(this.lastChartData, 'analysis-chart');
            this.renderChart(this.lastChartData, 'main-chart');
            this.resizeChart('analysis-chart');
            this.resizeChart('main-chart');
        }
    }

    handleSectionChange(section) {
        if (section === 'chart') {
            if (this.lastChartData) {
                this.renderChart(this.lastChartData, 'analysis-chart');
                this.resizeChart('analysis-chart');
            } else {
                this.loadChart(this.currentSymbol, this.currentInterval);
            }
            
            if (this.socket) {
                this.socket.emit('request_chart', {
                    symbol: this.currentSymbol,
                    interval: this.currentInterval
                });
            }
        }
    }
    
    showOperationDetails(operationId) {
        const operation = this.operations.find(op => op.id == operationId);
        if (!operation) return;
        
        const modalBody = document.getElementById('operation-details');
        modalBody.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Informações da Operação</h6>
                    <table class="table table-sm">
                        <tr><th>Ativo:</th><td>${operation.symbol}</td></tr>
                        <tr><th>Tipo:</th><td>${operation.tipo}</td></tr>
                        <tr><th>Entrada:</th><td>R$ ${operation.entrada.toFixed(2)}</td></tr>
                        <tr><th>Stop:</th><td>R$ ${operation.stop.toFixed(2)}</td></tr>
                        <tr><th>Alvo:</th><td>R$ ${operation.alvo.toFixed(2)}</td></tr>
                        <tr><th>Quantidade:</th><td>${operation.quantidade}</td></tr>
                        <tr><th>Status:</th><td><span class="badge ${this.getStatusBadgeClass(operation.status)}">${operation.status}</span></td></tr>
                        <tr><th>Data:</th><td>${new Date(operation.created_at).toLocaleString('pt-BR')}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Análise</h6>
                    <table class="table table-sm">
                        <tr><th>Pontos Alvo:</th><td>${(operation.alvo - operation.entrada).toFixed(2)}</td></tr>
                        <tr><th>Pontos Stop:</th><td>${(operation.entrada - operation.stop).toFixed(2)}</td></tr>
                        <tr><th>Risco/Retorno:</th><td>${((operation.alvo - operation.entrada) / (operation.entrada - operation.stop)).toFixed(2)}:1</td></tr>
                        <tr><th>Valor Total:</th><td>R$ ${(operation.entrada * operation.quantidade).toFixed(2)}</td></tr>
                        <tr><th>Risco Total:</th><td>R$ ${((operation.entrada - operation.stop) * operation.quantidade).toFixed(2)}</td></tr>
                        <tr><th>Retorno Potencial:</th><td>R$ ${((operation.alvo - operation.entrada) * operation.quantidade).toFixed(2)}</td></tr>
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
            pdfBtn.href = `/reports/${operation.pdf_path.split('/').pop()}`;
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
            '5m': '5 Minutos',
            '15m': '15 Minutos',
            '1h': '1 Hora',
            '1d': 'Diário'
        };
        return intervals[interval] || interval;
    }
    
    setupEventListeners() {
        // Timeframe buttons
        document.querySelectorAll('.timeframe-btn, .timeframe-btn-chart').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const interval = e.currentTarget.getAttribute('data-interval');
                this.currentInterval = interval;
                
                // Update active button
                e.currentTarget.parentElement.querySelectorAll('.btn').forEach(b => {
                    b.classList.remove('active');
                });
                e.currentTarget.classList.add('active');
                
                // Reload chart
                this.loadChart(this.currentSymbol, interval);
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
            const operation = {
                symbol: document.getElementById('quick-symbol').value,
                tipo: document.getElementById('quick-type').value,
                entrada: parseFloat(document.getElementById('quick-entrada').value),
                stop: parseFloat(document.getElementById('quick-entrada').value) * 0.98,
                alvo: parseFloat(document.getElementById('quick-entrada').value) * 1.02,
                quantidade: parseInt(document.getElementById('quick-quantidade').value),
                timeframe: '15m'
            };
            
            try {
                await this.registerOperation(operation);
                document.getElementById('quick-entrada').value = '';
            } catch (error) {
                // Error already handled by registerOperation
            }
        });
        
        // Fetch price button
        document.getElementById('fetch-price-btn').addEventListener('click', async () => {
            const symbol = document.getElementById('operation-symbol').value.trim();
            if (symbol) {
                try {
                    const response = await fetch(`/api/quote/${symbol}`);
                    const data = await response.json();
                    
                    document.getElementById('operation-entrada').value = data.price.toFixed(2);
                    document.getElementById('asset-name').textContent = data.name;
                    document.getElementById('operation-stop').value = (data.price * 0.98).toFixed(2);
                    document.getElementById('operation-alvo').value = (data.price * 1.02).toFixed(2);
                    
                    this.calculateOperationSummary();
                    
                } catch (error) {
                    this.showToast('Erro ao buscar preço do ativo', 'danger');
                }
            }
        });
        
        // Operation form submit
        document.getElementById('operation-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const operation = {
                ativo: document.getElementById('operation-symbol').value,
                tipo: document.getElementById('operation-type').value,
                entrada: parseFloat(document.getElementById('operation-entrada').value),
                stop: parseFloat(document.getElementById('operation-stop').value),
                alvo: parseFloat(document.getElementById('operation-alvo').value),
                quantidade: parseInt(document.getElementById('operation-quantidade').value),
                timeframe: document.getElementById('operation-timeframe').value,
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
                
            } finally {
                // Re-enable submit button
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        });
        
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
        
        // Auto-calculate operation values
        ['operation-entrada', 'operation-stop', 'operation-alvo', 'operation-quantidade'].forEach(id => {
            document.getElementById(id).addEventListener('input', () => {
                this.calculateOperationSummary();
            });
        });
    }
    
    calculateOperationSummary() {
        const entrada = parseFloat(document.getElementById('operation-entrada').value) || 0;
        const stop = parseFloat(document.getElementById('operation-stop').value) || 0;
        const alvo = parseFloat(document.getElementById('operation-alvo').value) || 0;
        const quantidade = parseInt(document.getElementById('operation-quantidade').value) || 0;
        
        // Calculate values
        const pontosStop = entrada - stop;
        const pontosAlvo = alvo - entrada;
        const riscoReward = pontosStop > 0 ? (pontosAlvo / pontosStop).toFixed(2) : 0;
        
        const totalValue = entrada * quantidade;
        const riscoTotal = pontosStop * quantidade;
        
        // Update display
        document.getElementById('risk-reward').value = `${riscoReward}:1`;
        document.getElementById('total-value').textContent = `R$ ${totalValue.toFixed(2)}`;
        document.getElementById('pontos-stop').textContent = pontosStop.toFixed(2);
        document.getElementById('pontos-alvo').textContent = pontosAlvo.toFixed(2);
        document.getElementById('risco-total').textContent = `R$ ${riscoTotal.toFixed(2)}`;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingApp = new TradingApp();
});