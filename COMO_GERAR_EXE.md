# Como gerar o executável (.exe) do sistema

1. Certifique-se de que todas as dependências estejam instaladas:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Gere o executável com o comando:
   ```bash
   pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" app.py
   ```
   - O executável será criado na pasta `dist`.
   - Para rodar, basta dar duplo clique no arquivo `app.exe` (ou executar pelo terminal).

3. Observações:
   - O sistema abrirá normalmente no navegador padrão, acessando http://localhost:5000
   - Se adicionar novos arquivos em `static` ou `templates`, gere o .exe novamente.
   - O arquivo de configuração `config.py` deve estar na mesma pasta do executável.

4. Dica:
   - Para personalizar o ícone do .exe, adicione a opção `--icon=icone.ico` ao comando.

---

Dúvidas ou problemas? Consulte a documentação do PyInstaller: https://pyinstaller.org/
