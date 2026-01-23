"""Route registration helpers."""
from .api import register_api_routes
from .socket_handlers import register_socket_handlers

__all__ = ["register_api_routes", "register_socket_handlers"]
