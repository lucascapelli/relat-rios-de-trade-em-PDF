"""Run script for the trading reports system."""
from __future__ import annotations

import os
import threading
import time
import webbrowser

from app import create_app, get_services, socketio
from app.background import start_background_tasks
from app.config import PROJECT_ROOT
from app.utils import logger


LOCK_FILE = PROJECT_ROOT / ".browser_lock"


def should_open_browser() -> bool:
    try:
        if LOCK_FILE.exists():
            file_age = time.time() - LOCK_FILE.stat().st_mtime
            if file_age < 10:
                return False
            LOCK_FILE.unlink(missing_ok=True)

        LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except Exception:
        return False


def is_primary_process(app) -> bool:
    return (not app.debug) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def main(app=None) -> None:
    app = app or create_app()
    services = get_services(app)

    if is_primary_process(app):
        logger.info("🔁 Processo principal do Flask")
        start_background_tasks(services)

        if should_open_browser():
            def _open_browser_once() -> None:
                time.sleep(2)
                webbrowser.open("http://127.0.0.1:5000")
                logger.info("🌐 Navegador aberto automaticamente")

            threading.Thread(target=_open_browser_once, daemon=True).start()
        else:
            logger.info("🌐 Navegador já foi aberto (pulando)")

        logger.info("✅ Servidor disponível em http://127.0.0.1:5000")
    else:
        logger.info("🔄 Processo do reloader (monitoramento)")

    try:
        socketio.run(
            app,
            debug=True,
            port=5000,
            allow_unsafe_werkzeug=True,
            log_output=False,
        )
    finally:
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
	main()
