// Nova Operação Manager - Unified System (Swing Trade, Day Trade, Carteiras)
class NovaOperacaoManager {
    constructor(app) {
        this.app = app;
        this.currentMode = 'swing';
        this.dayTradeEntries = { C: [], V: [] }; // Compras e Vendas
        this.portfolioAssets = [];
        this.init();
    }

    init() {
        console.log('NovaOperacaoManager init() called');
        this.setupModeSelector();
        this.setupSwingTrade();
        this.setupDayTrade();
        this.setupPortfolio();
        this.setDefaultDates();
        console.log('NovaOperacaoManager init() completed');
    }

    setDefaultDates() {
        const today = new Date().toISOString().split('T')[0];
        const elements = ['swing-trade-date', 'daytrade-date', 'portfolio-reference-date'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.value) el.value = today;
        });
    }

    setupModeSelector() {
        console.log('setupModeSelector() called');
        ['mode-swing', 'mode-daytrade', 'mode-portfolio'].forEach(id => {
            const element = document.getElementById(id);
            console.log(`Checking element ${id}:`, element);
            if (element) {
                element.addEventListener('change', (e) => {
                    console.log('Mode change event triggered for:', id);
                    if (e.target.checked) {
                        const mode = id.replace('mode-', '');
                        this.switchMode(mode);
                    }
                });
                console.log(`Event listener added to ${id}`);
            } else {
                console.error(`Element not found: ${id}`);
            }
        });
    }

    switchMode(mode) {
        console.log('Switching to mode:', mode);
        this.currentMode = mode;
        // Hide all containers
        document.querySelectorAll('.mode-container').forEach(el => el.style.display = 'none');
        
        // Show selected container
        const containers = {
            'swing': 'swing-trade-container',
            'daytrade': 'daytrade-container',
            'portfolio': 'portfolio-container'
        };
        const container = document.getElementById(containers[mode]);
        if (container) {
            container.style.display = 'block';
            console.log('Container shown:', containers[mode]);
        } else {
            console.error('Container not found:', containers[mode]);
        }
    }

    // ========== SWING TRADE ==========
    setupSwingTrade() {
        const form = document.getElementById('swing-trade-form');
        if (!form) return;

        // Calculate percentages on input
        ['swing-entry', 'swing-target', 'swing-stop'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', () => this.calculateSwingPercentages());
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.submitSwingTrade();
        });

        // AI Text generation
        document.getElementById('swing-generate-ai-text')?.addEventListener('click', () => {
            this.generateAIText('swing');
        });

        // Update PDF text
        document.getElementById('swing-update-pdf-text')?.addEventListener('click', () => {
            this.showToast('Funcionalidade de atualização de PDF será implementada', 'info');
        });
    }

    calculateSwingPercentages() {
        const entry = parseFloat(document.getElementById('swing-entry')?.value) || 0;
        const target = parseFloat(document.getElementById('swing-target')?.value) || 0;
        const stop = parseFloat(document.getElementById('swing-stop')?.value) || 0;

        if (entry > 0) {
            const targetPercent = ((target - entry) / entry * 100).toFixed(2);
            const stopPercent = ((entry - stop) / entry * 100).toFixed(2);
            
            document.getElementById('swing-target-percent').value = targetPercent;
            document.getElementById('swing-stop-percent').value = Math.abs(stopPercent);
        }
    }

    async submitSwingTrade() {
        const data = {
            symbol: document.getElementById('swing-symbol')?.value,
            direction: document.getElementById('swing-direction')?.value,
            entry: parseFloat(document.getElementById('swing-entry')?.value),
            entry_min: parseFloat(document.getElementById('swing-entry-min')?.value) || parseFloat(document.getElementById('swing-entry')?.value),
            entry_max: parseFloat(document.getElementById('swing-entry-max')?.value) || parseFloat(document.getElementById('swing-entry')?.value),
            target: parseFloat(document.getElementById('swing-target')?.value),
            stop: parseFloat(document.getElementById('swing-stop')?.value),
            quantity: parseInt(document.getElementById('swing-quantity')?.value),
            trade_date: document.getElementById('swing-trade-date')?.value,
            timeframe_major: document.getElementById('swing-tf-major')?.value,
            timeframe_minor: document.getElementById('swing-tf-minor')?.value,
            risk_amount: parseFloat(document.getElementById('swing-risk-amount')?.value) || null,
            risk_percent: parseFloat(document.getElementById('swing-risk-percent')?.value) || null,
            target_percent: parseFloat(document.getElementById('swing-target-percent')?.value) || null,
            stop_percent: parseFloat(document.getElementById('swing-stop-percent')?.value) || null,
            analytical_text: document.getElementById('swing-analytical-text')?.value,
            client_name: document.getElementById('swing-client-name')?.value || null
        };

        try {
            const response = await fetch('/api/swing-trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showToast('Swing trade registrado com sucesso!', 'success');
                
                // Generate and open PDF
                if (result.id) {
                    try {
                        const pdfResponse = await fetch(`/api/swing-trade/${result.id}/pdf`);
                        const pdfResult = await pdfResponse.json();
                        
                        if (pdfResponse.ok && pdfResult.url) {
                            // Open PDF in new tab
                            window.open(pdfResult.url, '_blank');
                            this.showToast('PDF gerado com sucesso!', 'success');
                        } else {
                            const detail = pdfResult && pdfResult.error ? `: ${pdfResult.error}` : '';
                            this.showToast(`Trade registrado, mas erro ao gerar PDF${detail}`, 'warning');
                        }
                    } catch (pdfError) {
                        console.error('Erro ao gerar PDF:', pdfError);
                        this.showToast('Trade registrado, mas erro ao gerar PDF', 'warning');
                    }
                }
                
                document.getElementById('swing-trade-form')?.reset();
                this.setDefaultDates();
            } else {
                this.showToast('Erro: ' + (result.error || 'Erro desconhecido'), 'danger');
            }
        } catch (error) {
            this.showToast('Erro ao enviar swing trade: ' + error.message, 'danger');
        }
    }

    // ========== DAY TRADE ==========
    setupDayTrade() {
        const form = document.getElementById('daytrade-form');
        if (!form) return;

        document.getElementById('dt-add-entry')?.addEventListener('click', () => {
            this.addDayTradeEntry();
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.submitDayTrade();
        });
    }

    addDayTradeEntry() {
        const entry = {
            symbol: document.getElementById('dt-entry-symbol')?.value,
            direction: document.getElementById('dt-entry-direction')?.value,
            entry: parseFloat(document.getElementById('dt-entry-price')?.value),
            max_entry_variation: parseFloat(document.getElementById('dt-entry-variation')?.value) || 0,
            target: parseFloat(document.getElementById('dt-entry-target')?.value),
            stop: parseFloat(document.getElementById('dt-entry-stop')?.value)
        };

        if (!entry.symbol || !entry.entry || !entry.target || !entry.stop) {
            this.showToast('Preencha todos os campos obrigatórios do trade', 'warning');
            return;
        }

        // Calculate percentages
        entry.risk_zero_price = entry.entry; // Simplified
        entry.risk_zero_percent = 0;
        entry.target_percent = ((entry.target - entry.entry) / entry.entry * 100).toFixed(2);
        entry.stop_percent = Math.abs((entry.stop - entry.entry) / entry.entry * 100).toFixed(2);

        // Add to appropriate list
        const dir = entry.direction === 'C' ? 'C' : 'V';
        this.dayTradeEntries[dir].push(entry);

        // Update UI
        this.renderDayTradeEntries();

        // Clear form
        ['dt-entry-symbol', 'dt-entry-price', 'dt-entry-variation', 'dt-entry-target', 'dt-entry-stop'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });

        this.showToast(`Trade ${entry.symbol} adicionado`, 'success');
    }

    renderDayTradeEntries() {
        // Render Compras
        const buysTable = document.getElementById('dt-buys-tbody');
        if (buysTable) {
            buysTable.innerHTML = this.dayTradeEntries.C.map((entry, idx) => `
                <tr>
                    <td><strong>${entry.symbol}</strong></td>
                    <td>R$ ${entry.entry.toFixed(2)}</td>
                    <td>R$ ${entry.risk_zero_price?.toFixed(2) || '-'}</td>
                    <td>${entry.risk_zero_percent?.toFixed(2) || '0.00'}%</td>
                    <td>R$ ${entry.target.toFixed(2)}</td>
                    <td class="text-success">${entry.target_percent}%</td>
                    <td>R$ ${entry.stop.toFixed(2)}</td>
                    <td class="text-danger">${entry.stop_percent}%</td>
                    <td>
                        <button class="btn btn-sm btn-danger" onclick="novaOperacaoManager.removeDayTradeEntry('C', ${idx})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('');
        }

        // Render Vendas
        const sellsTable = document.getElementById('dt-sells-tbody');
        if (sellsTable) {
            sellsTable.innerHTML = this.dayTradeEntries.V.map((entry, idx) => `
                <tr>
                    <td><strong>${entry.symbol}</strong></td>
                    <td>R$ ${entry.entry.toFixed(2)}</td>
                    <td>R$ ${entry.risk_zero_price?.toFixed(2) || '-'}</td>
                    <td>${entry.risk_zero_percent?.toFixed(2) || '0.00'}%</td>
                    <td>R$ ${entry.target.toFixed(2)}</td>
                    <td class="text-success">${entry.target_percent}%</td>
                    <td>R$ ${entry.stop.toFixed(2)}</td>
                    <td class="text-danger">${entry.stop_percent}%</td>
                    <td>
                        <button class="btn btn-sm btn-danger" onclick="novaOperacaoManager.removeDayTradeEntry('V', ${idx})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('');
        }
    }

    removeDayTradeEntry(direction, index) {
        this.dayTradeEntries[direction].splice(index, 1);
        this.renderDayTradeEntries();
        this.showToast('Trade removido', 'info');
    }

    async submitDayTrade() {
        const allEntries = [...this.dayTradeEntries.C, ...this.dayTradeEntries.V];
        
        if (allEntries.length === 0) {
            this.showToast('Adicione pelo menos um trade', 'warning');
            return;
        }

        const data = {
            trade_date: document.getElementById('daytrade-date')?.value,
            timeframe_major: '1h',
            timeframe_minor: '15m',
            risk_amount: parseFloat(document.getElementById('daytrade-risk-amount')?.value) || null,
            risk_percent: parseFloat(document.getElementById('daytrade-risk-percent')?.value) || null,
            entries: allEntries
        };

        try {
            const response = await fetch('/api/day-trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showToast('Day trade registrado com sucesso!', 'success');
                
                // Generate and open PDF
                if (result.id) {
                    try {
                        const pdfResponse = await fetch(`/api/day-trade/${result.id}/pdf`);
                        const pdfResult = await pdfResponse.json();
                        
                        if (pdfResponse.ok && pdfResult.url) {
                            window.open(pdfResult.url, '_blank');
                            this.showToast('PDF gerado com sucesso!', 'success');
                        } else {
                            const detail = pdfResult && pdfResult.error ? `: ${pdfResult.error}` : '';
                            this.showToast(`Sessão registrada, mas erro ao gerar PDF${detail}`, 'warning');
                        }
                    } catch (pdfError) {
                        console.error('Erro ao gerar PDF:', pdfError);
                        this.showToast('Sessão registrada, mas erro ao gerar PDF', 'warning');
                    }
                }
                
                this.dayTradeEntries = { C: [], V: [] };
                this.renderDayTradeEntries();
                document.getElementById('daytrade-form')?.reset();
                this.setDefaultDates();
            } else {
                this.showToast('Erro: ' + (result.error || 'Erro desconhecido'), 'danger');
            }
        } catch (error) {
            this.showToast('Erro ao enviar day trade: ' + error.message, 'danger');
        }
    }

    // ========== PORTFOLIO ==========
    setupPortfolio() {
        const form = document.getElementById('portfolio-form');
        if (!form) return;

        document.getElementById('pf-load-assets')?.addEventListener('click', async () => {
            await this.loadManipulatedAssets();
        });

        document.getElementById('pf-generate-pdf')?.addEventListener('click', async () => {
            await this.generateManipulatedPortfolioPdf();
        });

        document.getElementById('pf-clear-assets')?.addEventListener('click', () => {
            this.clearManipulatedAssets();
        });
    }

    getPortfolioFilters() {
        const period = document.getElementById('portfolio-period')?.value || 'weekly';
        const reference_date = document.getElementById('portfolio-reference-date')?.value || null;
        const include_daytrade = document.getElementById('portfolio-include-daytrade')?.checked !== false;
        const include_swing = document.getElementById('portfolio-include-swing')?.checked !== false;
        return { period, reference_date, include_daytrade, include_swing };
    }

    async loadManipulatedAssets() {
        const { period, reference_date, include_daytrade, include_swing } = this.getPortfolioFilters();

        try {
            const qs = new URLSearchParams({
                period,
                ...(reference_date ? { reference_date } : {}),
                include_daytrade: include_daytrade ? '1' : '0',
                include_swing: include_swing ? '1' : '0'
            });

            const response = await fetch(`/api/portfolio/manipulated?${qs.toString()}`);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result?.error || 'Erro ao carregar ativos');
            }

            this.portfolioAssets = Array.isArray(result.assets) ? result.assets : [];
            this.renderPortfolioAssets();

            const rangeLabel = document.getElementById('pf-range-label');
            if (rangeLabel) {
                const start = result.start_date || '-';
                const end = result.end_date || '-';
                const count = this.portfolioAssets.length;
                rangeLabel.textContent = `Período: ${start} até ${end} • Itens: ${count}`;
            }

            this.showToast('Ativos carregados', 'success');
        } catch (error) {
            console.error('Erro ao carregar ativos manipulados:', error);
            this.showToast(`Erro ao carregar ativos: ${error.message}`, 'danger');
        }
    }

    clearManipulatedAssets() {
        this.portfolioAssets = [];
        this.renderPortfolioAssets();
        const rangeLabel = document.getElementById('pf-range-label');
        if (rangeLabel) rangeLabel.textContent = '';
        this.showToast('Lista limpa', 'info');
    }

    renderPortfolioAssets() {
        const tbody = document.getElementById('pf-assets-tbody');
        if (!tbody) return;

        const fmt = (value) => {
            const num = Number(value);
            return Number.isFinite(num) ? `R$ ${num.toFixed(2)}` : '-';
        };

        const originBadge = (source) => {
            const s = String(source || '').toLowerCase();
            if (s === 'daytrade') return '<span class="badge bg-success">Day Trade</span>';
            if (s === 'swing') return '<span class="badge bg-primary">Swing</span>';
            return '<span class="badge bg-secondary">-</span>';
        };

        tbody.innerHTML = this.portfolioAssets.map((asset) => `
            <tr>
                <td><strong>${asset.symbol}</strong></td>
                <td>${fmt(asset.entry)}</td>
                <td>${fmt(asset.entry_max)}</td>
                <td>${fmt(asset.risk_zero)}</td>
                <td>${fmt(asset.target)}</td>
                <td>${fmt(asset.stop)}</td>
                <td>${originBadge(asset.source)}</td>
                <td class="text-muted small">${asset.trade_date || '-'}</td>
            </tr>
        `).join('');
    }

    async generateManipulatedPortfolioPdf() {
        const { period, reference_date, include_daytrade, include_swing } = this.getPortfolioFilters();
        const analytical_text = document.getElementById('portfolio-analytical-text')?.value || '';

        try {
            const response = await fetch('/api/portfolio/manipulated/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    period,
                    reference_date,
                    include_daytrade,
                    include_swing,
                    analytical_text
                })
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result?.error || 'Erro ao gerar PDF');
            }

            if (result.url) {
                window.open(result.url, '_blank');
            }
            this.showToast(`PDF gerado (${result.count || 0} itens)`, 'success');
        } catch (error) {
            console.error('Erro ao gerar PDF da carteira derivada:', error);
            this.showToast(`Erro ao gerar PDF: ${error.message}`, 'danger');
        }
    }

    // ========== UTILITIES ==========
    showToast(message, type = 'info') {
        if (this.app && typeof this.app.showToast === 'function') {
            this.app.showToast(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    generateAIText(mode) {
        // Placeholder for AI text generation
        const texts = {
            swing: 'Análise técnica gerada por IA: Ativo em tendência de alta com suporte em níveis importantes. Recomendação de entrada na faixa especificada com stop loss protetor.',
            daytrade: 'Setup intraday identificado com alta probabilidade de sucesso baseado em padrões de candlestick e volume.',
            portfolio: 'Carteira diversificada com foco em ações de alta qualidade e potencial de valorização de médio/longo prazo.'
        };
        
        const textareaId = mode === 'swing' ? 'swing-analytical-text' : 'portfolio-analytical-text';
        const textarea = document.getElementById(textareaId);
        if (textarea) {
            textarea.value = texts[mode] || '';
            this.showToast('Texto gerado! Você pode editá-lo livremente', 'success');
        }
    }
}

// Initialize when app is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for tradingApp to be available
    const initNovaOperacao = () => {
        if (window.tradingApp) {
            window.novaOperacaoManager = new NovaOperacaoManager(window.tradingApp);
            console.log('NovaOperacaoManager initialized');
        } else {
            // Retry after a short delay
            setTimeout(initNovaOperacao, 100);
        }
    };
    initNovaOperacao();
});
