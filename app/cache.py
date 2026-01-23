"""Simple in-memory cache with expiry control."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple


class CacheManager:
    def __init__(self, expiry_seconds: int = 60) -> None:
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.expiry = expiry_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.expiry:
                return data
        return None

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, datetime.now())

    def clear_expired(self) -> None:
        now = datetime.now()
        expired_keys = [
            key
            for key, (_, timestamp) in self.cache.items()
            if (now - timestamp).seconds >= self.expiry
        ]
        for key in expired_keys:
            del self.cache[key]
