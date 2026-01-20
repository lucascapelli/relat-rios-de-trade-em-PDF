// Configuração do SocketIO para tempo real
const socket = io();
let currentRealtimeChart = null;

// Elementos DOM - Inicializar após carregamento da página
let connectionStatus, realtimeAsset, realtimeTimeframe, realtimeRefreshBtn, realtimeChart, priceStats;

function initializeDOM() {
    connectionStatus = document.getElementById('connectionStatus');
    realtimeAsset = document.getElementById('realtimeAsset');
    realtimeTimeframe = document.getElementById('realtimeTimeframe');
    realtimeRefreshBtn = document.getElementById('realtimeRefreshBtn');
    realtimeChart = document.getElementById('realtimeChart');
    priceStats = document.getElementById('priceStats');
    
    // Adicionar listeners
    if (realtimeRefreshBtn) {
        realtimeRefreshBtn.addEventListener('click', requestRealtimeChart);
    }
    if (realtimeAsset) {
        realtimeAsset.addEventListener('change', requestRealtimeChart);
    }
    if (realtimeTimeframe) {
        realtimeTimeframe.addEventListener('change', requestRealtimeChart);
    }
    
    // Inicializar gráfico ao carregar
    requestRealtimeChart();
}

// Eventos SocketIO
socket.on('connect', function() {
    console.log('Conectado ao servidor WebSocket');
    connectionStatus.textContent = 'Conectado';
    connectionStatus.className = 'status-indicator status-connected';
});

socket.on('disconnect', function() {
    console.log('Desconectado do servidor');
    connectionStatus.textContent = 'Desconectado';
    connectionStatus.className = 'status-indicator status-disconnected';
});

socket.on('chart_data', function(data) {
    console.log('Dados do gráfico recebidos:', data);
    if (data.chart) {
        renderRealtimeChart(data.chart);
    }
    if (data.symbol) {
        updatePriceStatsFromChart(data);
    }
});

socket.on('price_update', function(data) {
    console.log('Atualização de preço:', data);
    updatePriceStats(data);
});

socket.on('chart_error', function(data) {
    console.error('Erro no gráfico:', data.error);
    alert('Erro ao carregar gráfico: ' + data.error);
});

// Funções
function requestRealtimeChart() {
    const asset = realtimeAsset ? realtimeAsset.value : 'PETR4.SA';
    const timeframe = realtimeTimeframe ? realtimeTimeframe.value : '15m';
    
    console.log(`Solicitando gráfico para ${asset} - ${timeframe}`);
    
    // Usar HTTP REST ao invés de SocketIO para melhor confiabilidade
    fetch(`/api/chart/${asset}/${timeframe}`)
        .then(response => response.json())
        .then(data => {
            console.log('Dados do gráfico recebidos:', data);
            if (data.chart_data) {
                renderRealtimeChart(data.chart_data);
            }
            // Buscar e atualizar dados de preço
            fetch(`/api/quote/${asset}`)
                .then(r => r.json())
                .then(priceData => {
                    console.log('Dados de preço:', priceData);
                    updatePriceStats(priceData);
                })
                .catch(err => console.error('Erro ao buscar preço:', err));
        })
        .catch(error => {
            console.error('Erro ao buscar gráfico:', error);
            alert('Erro ao carregar gráfico: ' + error.message);
        });
}

function renderRealtimeChart(chartData) {
    if (!chartData || !chartData.data) {
        console.error('Dados inválidos para o gráfico');
        return;
    }
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud']
    };
    
    if (currentRealtimeChart) {
        Plotly.react(realtimeChart, chartData.data, chartData.layout, config);
    } else {
        Plotly.newPlot(realtimeChart, chartData.data, chartData.layout, config);
        currentRealtimeChart = true;
    }
}

function updatePriceStats(data) {
    const asset = data.symbol || data.ativo || data.asset || 'N/A';
    const timestamp = new Date().toLocaleTimeString('pt-BR');
    
    // Mapear campos da API para variáveis locais
    const preco = data.price || data.preco || 'N/A';
    const variacao = data.change_percent || data.variacao || 0;
    const volume = data.volume || 0;
    const alta = data.high || data.alta || 'N/A';
    const baixa = data.low || data.baixa || 'N/A';
    const abertura = data.open || data.abertura || 'N/A';
    
    const statsHtml = `
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Ativo</div>
            <div style="font-size: 18px; font-weight: bold;">${asset}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Preço Atual</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${preco !== 'N/A' ? parseFloat(preco).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Variação</div>
            <div style="font-size: 18px; font-weight: bold; color: ${variacao >= 0 ? '#27ae60' : '#e74c3c'};">
                ${variacao !== 'N/A' ? parseFloat(variacao).toFixed(2) + '%' : 'N/A'}
            </div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Volume</div>
            <div style="font-size: 18px; font-weight: bold;">${volume ? (volume / 1000).toFixed(0) + 'K' : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Máxima</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${alta !== 'N/A' ? parseFloat(alta).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Mínima</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${baixa !== 'N/A' ? parseFloat(baixa).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Abertura</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${abertura !== 'N/A' ? parseFloat(abertura).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Atualizado</div>
            <div style="font-size: 14px; font-weight: bold;">${timestamp}</div>
        </div>
    `;
    
    if (priceStats) {
        priceStats.innerHTML = statsHtml;
    }
}

function updatePriceStatsFromChart(data) {
    // Função alternativa para quando os dados vêm do gráfico
    updatePriceStats(data);
}

// Event Listeners
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDOM);
} else {
    initializeDOM();
}

// Adicionar função ao objeto window para que onclick="" funcione
window.showTab = window.showTab || function() {};

// Inicialização quando a aba é aberta
function initRealtimeOnTabChange() {
    const graficosTab = document.getElementById('graficos');
    if (graficosTab && graficosTab.classList.contains('active')) {
        requestRealtimeChart();
        
        // Auto-atualizar a cada 30 segundos
        if (window.realtimeInterval) {
            clearInterval(window.realtimeInterval);
        }
        window.realtimeInterval = setInterval(requestRealtimeChart, 30000);
    } else {
        if (window.realtimeInterval) {
            clearInterval(window.realtimeInterval);
        }
    }
}

// Override da função showTab para inicializar realtime
const originalShowTab = window.showTab;
window.showTab = function(tabName, event) {
    originalShowTab(tabName, event);
    if (tabName === 'graficos') {
        setTimeout(initRealtimeOnTabChange, 100);
    }
};

// Inicialização ao carregar a página
window.addEventListener('load', function() {
    // Carrega o gráfico inicial se estiver na aba de gráficos
    if (document.getElementById('graficos').classList.contains('active')) {
        requestRealtimeChart();
    }
});
