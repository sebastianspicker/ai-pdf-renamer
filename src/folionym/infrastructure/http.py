"""Structural validation for configured HTTP endpoints."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

_URL_UNSAFE_CHARACTER_RE = re.compile(r"[\x00-\x20\x7f\\\\]")
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)


@dataclass(frozen=True)
class ValidatedHttpEndpoint:
    """A normalized HTTP(S) URL and the host facts used by transport policy."""

    url: str
    scheme: str
    hostname: str
    ip_host: ipaddress.IPv4Address | ipaddress.IPv6Address | None

    @property
    def is_literal_loopback(self) -> bool:
        """Return whether the URL uses a loopback IP literal."""
        return self.ip_host is not None and self.ip_host.is_loopback

    @property
    def is_loopback(self) -> bool:
        """Return whether the URL uses a loopback IP literal or localhost."""
        return self.is_literal_loopback or self.hostname.lower() == "localhost"


def _parse_http_url(value: str) -> tuple[str, SplitResult]:
    """Normalize and parse a URL after rejecting controls, spaces, and backslashes."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    if _URL_UNSAFE_CHARACTER_RE.search(normalized):
        raise ValueError("contains unsafe characters")
    try:
        return normalized, urlsplit(normalized)
    except ValueError as exc:
        raise ValueError("is malformed") from exc


def _validated_authority(parsed: SplitResult) -> tuple[str, int | None]:
    """Validate the authority, reject credentials, and return its host and port."""
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("is malformed") from exc
    if not parsed.netloc or hostname is None:
        raise ValueError("must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("must not include credentials")
    if port is None and parsed.netloc.endswith(":"):
        raise ValueError("has an empty port")
    return hostname, port


def _validated_dns_host(hostname: str, address_error: ValueError) -> None:
    """Validate a DNS hostname after it was rejected as an IP literal."""
    try:
        ascii_host = hostname.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("has an invalid host") from exc
    labels = ascii_host.split(".")
    if len(ascii_host) > 253 or not labels or not all(_HOST_LABEL_RE.fullmatch(label) for label in labels):
        raise ValueError("has an invalid host") from address_error


def _validated_ip_host(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Validate a hostname and return its IP literal, or None for a DNS name."""
    try:
        return ipaddress.ip_address(hostname)
    except ValueError as address_error:
        _validated_dns_host(hostname, address_error)
        return None


def validate_http_endpoint(value: str) -> ValidatedHttpEndpoint:
    """Return validated HTTP endpoint facts without applying a caller's transport policy."""
    normalized, parsed = _parse_http_url(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("must use HTTP or HTTPS")
    if parsed.fragment:
        raise ValueError("must not include a fragment")
    hostname, _port = _validated_authority(parsed)
    return ValidatedHttpEndpoint(
        url=normalized,
        scheme=scheme,
        hostname=hostname,
        ip_host=_validated_ip_host(hostname),
    )
