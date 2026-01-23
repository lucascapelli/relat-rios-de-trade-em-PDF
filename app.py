"""Compatibility entrypoint forwarding to the new modular application."""
from __future__ import annotations

from app import create_app
from run import main

app = create_app()

if __name__ == "__main__":
    main(app)
