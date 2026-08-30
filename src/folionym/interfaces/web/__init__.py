"""Loopback-only browser interface for Folionym."""

from .app import create_app
from .cli import main

__all__ = ["create_app", "main"]
