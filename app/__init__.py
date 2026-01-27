"""Application factory for the trading reports system."""
from __future__ import annotations

from flask import Flask
from flask_socketio import SocketIO

from .background import start_background_tasks
from .cache import CacheManager
from .charts import ChartGenerator
from .config import (
    CHART_CACHE_EXPIRY,
    DB_PATH,
    DEFAULT_SECRET_KEY,
    PRICE_CACHE_EXPIRY,
    PROJECT_ROOT,
    REPORTS_DIR,
    SECRET_KEY,
)
from .database import Database
from .market_data import FinanceData
from .realtime import RealTimeManager
from .reports import ReportGenerator
from .routes.api import register_api_routes
from .routes.socket_handlers import register_socket_handlers
from .services import Services
from .utils import logger
from .auth import bp_auth

socketio = SocketIO(cors_allowed_origins="*")


def create_services(socketio_instance: SocketIO) -> Services:
    finance_data = FinanceData()
    chart_generator = ChartGenerator()
    database = Database(str(DB_PATH))
    report_generator = ReportGenerator(str(REPORTS_DIR))
    realtime_manager = RealTimeManager(socketio_instance, finance_data)
    price_cache = CacheManager(PRICE_CACHE_EXPIRY)
    chart_cache = CacheManager(CHART_CACHE_EXPIRY)

    return Services(
        finance_data=finance_data,
        chart_generator=chart_generator,
        database=database,
        report_generator=report_generator,
        realtime_manager=realtime_manager,
        price_cache=price_cache,
        chart_cache=chart_cache,
    )


def create_app(start_background: bool = False) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(PROJECT_ROOT / "static"),
        template_folder=str(PROJECT_ROOT / "templates"),
    )
    app.config["SECRET_KEY"] = SECRET_KEY

    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    services = create_services(socketio)
    app.extensions["services"] = services

    app.register_blueprint(bp_auth)

    register_api_routes(app, services)
    register_socket_handlers(socketio, services)

    if start_background:
        start_background_tasks(services)

    logger.info("Aplicação Flask inicializada")
    return app


def get_services(app: Flask) -> Services:
    return app.extensions["services"]


__all__ = ["create_app", "socketio", "get_services"]
