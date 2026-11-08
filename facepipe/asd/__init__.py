"""Active speaker detection backends behind a common interface."""

from facepipe.asd.base import ASDBackend, ASDResult, get_backend

__all__ = ["ASDBackend", "ASDResult", "get_backend"]
