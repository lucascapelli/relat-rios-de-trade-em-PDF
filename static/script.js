// Tab switching
function showTab(tabName, event) {
    const tabs = document.querySelectorAll('.tab-content');
    const navTabs = document.querySelectorAll('.nav-tab');
    
    tabs.forEach(tab => tab.classList.remove('active'));
    navTabs.forEach(tab => tab.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
    
    // Carregar histórico quando mudar pra aba
    if (tabName === 'historico') {
        carregarHistorico();
    }
}

// Enviar formulário para API Flask
document.getElementById('operacaoForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const dados = {
        ativo: document.getElementById('ativo').value,
        tipo: document.getElementById('tipo').value,
        entrada: parseFloat(document.getElementById('entrada').value),
        stop: parseFloat(document.getElementById('stop').value),
        alvo: parseFloat(document.getElementById('alvo').value),
        quantidade: parseInt(document.getElementById('quantidade').value),
        observacoes: document.getElementById('observacoes').value,
        preco_atual: parseFloat(document.getElementById('entrada').value) // Default = entrada
    };
    
    // Mostrar carregamento
    const submitBtn = document.querySelector('button[type="submit"]');
    const btnOriginal = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Gerando PDF...';
    
    fetch('/api/operacao', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dados)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Sucesso:', data);
        alert(`✓ Relatório gerado!\nID: ${data.id}\nPDF salvo em: ${data.pdf_path}`);
        document.getElementById('operacaoForm').reset();
        submitBtn.disabled = false;
        submitBtn.textContent = btnOriginal;
    })
    .catch(error => {
        console.error('Erro:', error);
        alert('❌ Erro ao gerar relatório. Verifique o console.');
        submitBtn.disabled = false;
        submitBtn.textContent = btnOriginal;
    });
});

// Carregar histórico de operações
function carregarHistorico() {
    fetch('/history')
    .then(response => response.json())
    .then(data => {
        const tbody = document.querySelector('#historico tbody');
        tbody.innerHTML = '';
        
        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4">Nenhuma operação registrada</td></tr>';
            return;
        }
        
        data.forEach(op => {
            const tr = document.createElement('tr');
            const dataBR = new Date(op.created_at).toLocaleDateString('pt-BR');
            tr.innerHTML = `
                <td>${op.id}</td>
                <td>${op.ativo}</td>
                <td>${dataBR}</td>
                <td><a href="${op.pdf_path}" download>📥 Download</a></td>
            `;
            tbody.appendChild(tr);
        });
    })
    .catch(error => console.error('Erro ao carregar histórico:', error));
}
