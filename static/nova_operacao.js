// Nova Operação Manager - Unified System (Swing Trade, Day Trade, Carteiras)
class NovaOperacaoManager {
    constructor(app) {
        this.app = app;
        this.currentMode = 'swing';
        this.dayTradeEntries = { C: [], V: [] }; // Compras e Vendas
        this.portfolioAssets = [];
        this.savedPortfolios = [];
        this.selectedPortfolioId = null;
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

        // Oculta Filtros Globais na aba Carteiras
        const filtersRow = document.getElementById('global-filters-row');
        if (filtersRow) {
            filtersRow.style.display = (mode === 'portfolio') ? 'none' : '';
        }

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

        if (mode === 'portfolio') {
            this.fetchSavedPortfolios();
        }
    }

    // ========== SWING TRADE ==========


    setupSwingTrade() {
        const form = document.getElementById('swing-trade-form');
        if (!form) return;

        // Cálculo automático de percentuais
        ['swing-entry', 'swing-target', 'swing-stop'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', () => this.calculateSwingPercentages());
        });

        // Atualizar pré-visualização ao alterar campos relevantes
        const previewFields = [
            'swing-symbol', 'swing-entry', 'swing-entry-max',
            'swing-target', 'swing-stop', 'swing-partial-exit', 'swing-partial'
        ];
        previewFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => this.updateSwingPreview());
            }
        });
        this.updateSwingPreview();

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

        // Atalhos de ativos por classe
        setTimeout(() => {
            const assetClassSelect = document.getElementById('filter-asset-class');
            const assetShortcutsDiv = document.getElementById('asset-shortcuts');
            if (assetClassSelect && assetShortcutsDiv) {
                const shortcuts = {
                    'ACOES': ['PETR4', 'VALE3', 'ITUB4', 'BBDC4'],
                    'DOLAR': ['DOL1!', 'WDOFUT', 'DOLFUT', 'DOLQ24'],
                    'INDICE': ['IBOV', 'WINFUT', 'IND1!', 'IBX100'],
                    'BITCOIN': ['BTC-USD', 'BTCBRL', 'BTCUSDT', 'XBTUSD'],
                    'SP500': ['SPX', 'SPY', 'IVVB11', 'ES1!'],
                    'BOI': ['BGIQ24', 'BGIFUT', 'BGI1!', 'BGIV24'],
                    'MILHO': ['CCMFUT', 'CCM1!', 'CCMQ24', 'CCMV24'],
                    'CUSTOM': []
                };
                function renderShortcuts(classe) {
                    const ativos = shortcuts[classe] || [];
                    if (ativos.length === 0) {
                        assetShortcutsDiv.innerHTML = '';
                        return;
                    }
                    assetShortcutsDiv.innerHTML = '<div class="d-flex flex-wrap gap-2">' + ativos.map(a => `<button type="button" class="btn btn-outline-secondary btn-sm asset-shortcut-btn">${a}</button>`).join('') + '</div>';
                    assetShortcutsDiv.querySelectorAll('.asset-shortcut-btn').forEach(btn => {
                        btn.addEventListener('click', () => {
                            document.getElementById('swing-symbol').value = btn.textContent;
                            this.updateSwingPreview();
                        });
                    });
                }
                assetClassSelect.addEventListener('change', (e) => {
                    renderShortcuts(e.target.value);
                });
                // Render inicial se não for ALL
                if (assetClassSelect.value && assetClassSelect.value !== 'ALL') {
                    renderShortcuts(assetClassSelect.value);
                } else {
                    assetShortcutsDiv.innerHTML = '';
                }
            }
        }, 0);
    }


    updateSwingPreview() {
        // Pega valores dos campos
        const ticker = document.getElementById('swing-symbol')?.value || '';
        const entry = document.getElementById('swing-entry')?.value || '';
        const entryMaxPercent = document.getElementById('swing-entry-max')?.value || '';
        const entryMax = (() => {
            const e = parseFloat(entry);
            const p = parseFloat(entryMaxPercent);
            if (e && p) return (e * (1 + p / 100)).toFixed(2);
            return '';
        })();
        const partialExit = document.getElementById('swing-partial-exit')?.value || '';
        const partial = document.getElementById('swing-partial')?.value || '';
        const target = document.getElementById('swing-target')?.value || '';
        const stop = document.getElementById('swing-stop')?.value || '';

        document.getElementById('preview-ticker').textContent = ticker;
        document.getElementById('preview-entry').textContent = entry;
        document.getElementById('preview-entry-min').textContent = entry; // Campo removido, usar entrada
        document.getElementById('preview-entry-max').textContent = entryMax;
        document.getElementById('preview-partial-exit').textContent = partialExit;
        document.getElementById('preview-partial').textContent = partial;
        document.getElementById('preview-target').textContent = target;
        document.getElementById('preview-stop').textContent = stop;
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
        const entry = parseFloat(document.getElementById('swing-entry')?.value);
        const entryMin = parseFloat(document.getElementById('swing-entry-min')?.value) || entry;
        const entryMaxPercent = parseFloat(document.getElementById('swing-entry-max')?.value);
        const entryMax = (entry && entryMaxPercent) ? (entry * (1 + entryMaxPercent / 100)) : entry;
        const data = {
            symbol: document.getElementById('swing-symbol')?.value,
            direction: document.getElementById('swing-direction')?.value,
            entry: entry,
            entry_min: entryMin,
            entry_max: entryMax,
            target: parseFloat(document.getElementById('swing-target')?.value),
            stop: parseFloat(document.getElementById('swing-stop')?.value),
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

        form.addEventListener('reset', () => {
            this.dayTradeEntries = { C: [], V: [] };
            this.renderDayTradeEntries();
            this.setDefaultDates();
        });

        this.updateDayTradeSummary();
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

        this.updateDayTradeSummary();
    }

    updateDayTradeSummary() {
        const buys = this.dayTradeEntries.C.length;
        const sells = this.dayTradeEntries.V.length;
        const total = buys + sells;

        const totalBadge = document.getElementById('dt-total-count');
        if (totalBadge) {
            totalBadge.textContent = `${total} trade${total === 1 ? '' : 's'} na lista`;
        }

        const summary = document.getElementById('dt-list-summary');
        if (summary) {
            const buysLabel = buys === 1 ? 'compra' : 'compras';
            const sellsLabel = sells === 1 ? 'venda' : 'vendas';
            summary.textContent = `${buys} ${buysLabel} • ${sells} ${sellsLabel}`;
        }

        const emptyState = document.getElementById('dt-empty-state');
        const tableWrapper = document.getElementById('dt-table-wrapper');
        if (emptyState) emptyState.style.display = total === 0 ? 'block' : 'none';
        if (tableWrapper) tableWrapper.style.display = total === 0 ? 'none' : 'block';

        const pdfBtn = document.getElementById('daytrade-generate-pdf');
        if (pdfBtn) {
            pdfBtn.disabled = total === 0;
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
                // Atualiza estatísticas rápidas do dashboard
                if (window.tradingApp && typeof window.tradingApp.loadOperationsHistory === 'function') {
                    window.tradingApp.loadOperationsHistory();
                }
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

        document.getElementById('pf-refresh-saved')?.addEventListener('click', async () => {
            await this.fetchSavedPortfolios(true);
        });

        document.getElementById('pf-load-saved')?.addEventListener('click', () => {
            this.loadSavedPortfolioFromSelect();
        });

        document.getElementById('pf-pdf-saved')?.addEventListener('click', async () => {
            await this.generateSavedPortfolioPdf();
        });

        this.fetchSavedPortfolios();
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

    async fetchSavedPortfolios(showToast = false) {
        try {
            const response = await fetch('/api/portfolio/list');
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result?.error || 'Erro ao carregar carteiras salvas');
            }

            this.savedPortfolios = Array.isArray(result) ? result : [];
            this.renderSavedPortfolioOptions();

            if (showToast) {
                this.showToast('Carteiras atualizadas', 'success');
            }
        } catch (error) {
            console.error('Erro ao listar carteiras:', error);
            this.showToast(`Erro ao listar carteiras: ${error.message}`, 'danger');
        }
    }

    renderSavedPortfolioOptions() {
        const select = document.getElementById('pf-saved-select');
        if (!select) return;

        const options = ['<option value="">Selecione uma carteira salva</option>'];
        this.savedPortfolios.forEach((p) => {
            const labelParts = [
                `#${p.id}`,
                p.portfolio_type || 'GERAL',
                p.start_date && p.end_date ? `${p.start_date} -> ${p.end_date}` : (p.created_at || ''),
                p.version ? `v${p.version}` : null,
            ].filter(Boolean);
            options.push(`<option value="${p.id}">${labelParts.join(' • ')}</option>`);
        });

        select.innerHTML = options.join('');
        if (!this.savedPortfolios.find(p => p.id === this.selectedPortfolioId)) {
            this.selectedPortfolioId = null;
        }
        if (this.selectedPortfolioId) {
            select.value = String(this.selectedPortfolioId);
        }
    }

    loadSavedPortfolioFromSelect() {
        const select = document.getElementById('pf-saved-select');
        if (!select) return;
        const id = parseInt(select.value, 10);
        if (!id) {
            this.showToast('Escolha uma carteira salva para carregar', 'warning');
            return;
        }

        const portfolio = this.savedPortfolios.find((p) => p.id === id);
        if (!portfolio) {
            this.showToast('Carteira não encontrada na lista', 'danger');
            return;
        }

        this.applySavedPortfolio(portfolio);
    }

    applySavedPortfolio(portfolio) {
        this.selectedPortfolioId = portfolio.id;
        this.portfolioAssets = Array.isArray(portfolio.assets) ? portfolio.assets : [];
        this.renderPortfolioAssets();

        const rangeLabel = document.getElementById('pf-range-label');
        if (rangeLabel) {
            const start = portfolio.start_date || '-';
            const end = portfolio.end_date || '-';
            const count = this.portfolioAssets.length;
            const version = portfolio.version ? ` • v${portfolio.version}` : '';
            rangeLabel.textContent = `Carteira salva: ${start} até ${end} • Itens: ${count}${version}`;
        }

        const textarea = document.getElementById('portfolio-analytical-text');
        if (textarea) {
            textarea.value = portfolio.analytical_text || '';
        }

        const meta = document.getElementById('pf-saved-meta');
        if (meta) {
            meta.textContent = `${portfolio.portfolio_type || 'GERAL'} • Estado: ${portfolio.state || '-'} • Criada em ${portfolio.created_at || '-'}`;
        }

        this.showToast('Carteira salva carregada', 'success');
    }

    async generateSavedPortfolioPdf() {
        if (!this.selectedPortfolioId) {
            this.showToast('Selecione e carregue uma carteira salva antes de gerar PDF', 'warning');
            return;
        }

        try {
            const response = await fetch(`/api/portfolio/${this.selectedPortfolioId}/pdf`);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result?.error || 'Erro ao gerar PDF da carteira salva');
            }

            if (result.url) {
                window.open(result.url, '_blank');
            }

            this.showToast('PDF da carteira salva gerado', 'success');
        } catch (error) {
            console.error('Erro ao gerar PDF salvo:', error);
            this.showToast(`Erro ao gerar PDF: ${error.message}`, 'danger');
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
        // Placeholder aguardando texto padrão do Gustavo
        const textoPadrao = 'Aguardando texto padrão do Gustavo';
        const textareaId = mode === 'swing' ? 'swing-analytical-text' : 'portfolio-analytical-text';
        const textarea = document.getElementById(textareaId);
        if (textarea) {
            textarea.value = textoPadrao;
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
