from app import create_app, get_services
from app.background import start_background_tasks
import logging

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wsgi")

app = create_app()
services = get_services(app)

# Iniciar tarefas em background
# Nota: Usar gunicorn com -w 1 para evitar duplicação de tarefas de background
start_background_tasks(services)

if __name__ == "__main__":
    app.run()
