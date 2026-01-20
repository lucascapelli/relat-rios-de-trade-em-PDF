// Configuração do SocketIO para tempo real
const socket = io();
let currentRealtimeChart = null;

// Elementos DOM
const connectionStatus = document.getElementById('connectionStatus');
const realtimeAsset = document.getElementById('realtimeAsset');
const realtimeTimeframe = document.getElementById('realtimeTimeframe');
const realtimeRefreshBtn = document.getElementById('realtimeRefreshBtn');
const realtimeChart = document.getElementById('realtimeChart');
const priceStats = document.getElementById('priceStats');

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
    updatePriceDisplay(data);
});

socket.on('chart_error', function(data) {
    console.error('Erro no gráfico:', data.error);
    alert('Erro ao carregar gráfico: ' + data.error);
});

// Funções
function requestRealtimeChart() {
    const asset = realtimeAsset.value;
    const timeframe = realtimeTimeframe.value;
    
    console.log(`Solicitando gráfico para ${asset} - ${timeframe}`);
    socket.emit('request_chart', { 
        symbol: asset, 
        interval: timeframe
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
    
    const statsHtml = `
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Ativo</div>
            <div style="font-size: 18px; font-weight: bold;">${asset}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Preço Atual</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${data.preco ? parseFloat(data.preco).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Variação</div>
            <div style="font-size: 18px; font-weight: bold; color: ${data.variacao >= 0 ? '#27ae60' : '#e74c3c'};">
                ${data.variacao ? parseFloat(data.variacao).toFixed(2) + '%' : 'N/A'}
            </div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Volume</div>
            <div style="font-size: 18px; font-weight: bold;">${data.volume ? (data.volume / 1000).toFixed(0) + 'K' : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Máxima</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${data.alta ? parseFloat(data.alta).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Mínima</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${data.baixa ? parseFloat(data.baixa).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Abertura</div>
            <div style="font-size: 18px; font-weight: bold;">R$ ${data.abertura ? parseFloat(data.abertura).toFixed(2) : 'N/A'}</div>
        </div>
        <div class="stat-box" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Atualizado</div>
            <div style="font-size: 14px; font-weight: bold;">${timestamp}</div>
        </div>
    `;
    
    priceStats.innerHTML = statsHtml;
}

function updatePriceStatsFromChart(data) {
    // Função alternativa para quando os dados vêm do gráfico
    updatePriceStats(data);
}

// Event Listeners
realtimeRefreshBtn.addEventListener('click', requestRealtimeChart);
realtimeAsset.addEventListener('change', requestRealtimeChart);
realtimeTimeframe.addEventListener('change', requestRealtimeChart);

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
