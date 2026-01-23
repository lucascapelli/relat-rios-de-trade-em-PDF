"""Background task orchestration."""
from __future__ import annotations

import threading
import time

from .services import Services
from .utils import logger


def start_background_tasks(services: Services) -> None:
    logger.info("🚀 Iniciando sistema de dados financeiros...")

    update_thread = threading.Thread(target=services.realtime_manager.start_updates, daemon=True)
    update_thread.start()

    def cache_cleaner() -> None:
        while True:
            time.sleep(300)
            services.price_cache.clear_expired()
            services.chart_cache.clear_expired()
            logger.debug("🧹 Cache limpo")

    cleaner_thread = threading.Thread(target=cache_cleaner, daemon=True)
    cleaner_thread.start()
