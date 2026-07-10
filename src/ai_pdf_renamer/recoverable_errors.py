"""Shared recoverable exception groups for operator-facing boundaries."""

from __future__ import annotations

import json

import requests

COMMON_RECOVERABLE_EXCEPTIONS = (
    json.JSONDecodeError,
    KeyError,
    OSError,
    requests.RequestException,
    RuntimeError,
    TypeError,
    ValueError,
)
