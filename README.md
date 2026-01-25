# relatoriostrade

Sistema interno para geração de relatórios de trade em PDF.

## Tecnologias
- Python (Flask)
- SQLite
- Plotly
- ReportLab (PDF)
- mplfinance/matplotlib (gráficos em PNG para o PDF)

## Funcionalidades
- Formulário de operação
- Cálculos automáticos
- Gráficos de candle (4 timeframes)
- Geração de relatórios em PDF
- Histórico simples
- Configurações institucionais centralizadas

## Setup
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o sistema:
   ```bash
   python app.py
   ```

## Estrutura inicial
- app.py: ponto de entrada Flask
- static/: arquivos estáticos (CSS, logo)
- templates/: HTMLs
- config.py: configurações institucionais
- database.db: SQLite

## Observações
- O sistema roda localmente.
- PDF gerado conforme template fornecido.
- Estrutura modular para futuras expansões.
