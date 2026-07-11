"""Post-rename side effects for the batch renamer."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

import requests

from .config import RenamerConfig
from .renamer_output import _append_export_row

logger = logging.getLogger("ai_pdf_renamer.renamer")


@dataclass
class PostRenameAction:
    file_path: Path
    target: Path
    current_base: str
    meta: dict[str, object]
    export_rows: list[dict[str, object]]


def _write_pdf_title_metadata(pdf_path: Path, title: str) -> None:
    """Write /Title metadata to PDF (PyMuPDF). No-op if fitz unavailable or on error.

    Writes to a temporary file first, then atomically replaces the original to
    prevent corruption under concurrent access.
    """
    try:
        import tempfile

        import fitz

        doc = fitz.open(pdf_path)
        try:
            doc.set_metadata({"title": title or pdf_path.stem})
            # Write to a temporary file in the same directory, then replace atomically.
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=pdf_path.parent)
            os.close(fd)
            tmp = Path(tmp_path)
            encryption_keep = getattr(fitz, "PDF_ENCRYPT_KEEP", None)
            try:
                if encryption_keep is None:
                    doc.save(str(tmp), incremental=False)
                else:
                    doc.save(str(tmp), incremental=False, encryption=encryption_keep)
            except TypeError:
                # Older PyMuPDF builds may expose incompatible save signatures.
                doc.save(str(tmp), incremental=False)
            except (AttributeError, OSError, RuntimeError, ValueError):
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
        finally:
            doc.close()
        # Do not replace a valid PDF with an empty temp file after a partial write.
        tmp_size = tmp.stat().st_size
        if tmp_size == 0:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise OSError(f"Temporary PDF file is empty (possible disk full): {tmp}")
        # Atomic replace (os.replace is atomic on POSIX, best-effort on Windows).
        os.replace(tmp, pdf_path)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Could not write PDF metadata for %s: %s", pdf_path, exc)


# C0 control chars (incl. NUL) must be stripped from env-var values passed to hook commands.
_HOOK_ENV_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _hook_env(old_path: Path, new_path: Path, meta: dict[str, object]) -> dict[str, str]:
    # Sanitize path strings: strip C0 control characters (\x00-\x1f) and DEL (\x7f)
    # from values passed via env vars to hook commands. NUL (\x00) is especially
    # important as it truncates C strings, but other control chars can also cause issues.
    old_value = _HOOK_ENV_CONTROL_RE.sub("", str(old_path))
    new_value = _HOOK_ENV_CONTROL_RE.sub("", str(new_path))
    env = {**os.environ, "AI_PDF_RENAMER_OLD_PATH": old_value, "AI_PDF_RENAMER_NEW_PATH": new_value}
    try:
        meta_json = json.dumps(meta, default=str)
    except (TypeError, ValueError):
        meta_json = "{}"
    env["AI_PDF_RENAMER_META"] = meta_json
    return env


_URL_UNSAFE_CHARACTER_RE = re.compile(r"[\x00-\x20\x7f\\]")
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)


def _parsed_hook_url(value: str) -> SplitResult:
    if _URL_UNSAFE_CHARACTER_RE.search(value):
        raise ValueError("contains unsafe characters")
    try:
        return urlsplit(value)
    except ValueError as exc:
        raise ValueError("is malformed") from exc


def _validated_hook_authority(parsed: SplitResult) -> str:
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
    return hostname


def _validated_hook_scheme(parsed: SplitResult) -> str:
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("must use HTTP or HTTPS")
    return scheme


def _validated_ip_host(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname)
    except ValueError as address_error:
        try:
            ascii_host = hostname.rstrip(".").encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("has an invalid host") from exc
        labels = ascii_host.split(".")
        if len(ascii_host) > 253 or not labels or not all(_HOST_LABEL_RE.fullmatch(label) for label in labels):
            raise ValueError("has an invalid host") from address_error
        return None


def _validated_hook_url(value: str) -> str:
    """Return a safe HTTP(S) hook URL or raise a value-only validation error."""
    parsed = _parsed_hook_url(value)
    scheme = _validated_hook_scheme(parsed)
    hostname = _validated_hook_authority(parsed)
    ip_host = _validated_ip_host(hostname)
    if scheme == "http" and (ip_host is None or not ip_host.is_loopback):
        raise ValueError("unencrypted plain HTTP requires a literal loopback IP address")
    return value


def _post_hook_http(cmd: str, old_path: Path, new_path: Path, meta: dict[str, object]) -> None:
    payload = {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "meta": meta,
    }
    with requests.Session() as session:
        session.trust_env = False
        resp = session.post(cmd, json=payload, timeout=10, allow_redirects=False)
        resp.raise_for_status()


def _warn_local_hook_disabled() -> None:
    logger.warning(
        "Local post-rename hook commands are disabled and were not run. "
        "Configure an HTTPS post-rename hook endpoint instead."
    )


def run_post_rename_hook(hook_cmd: str, old_path: Path, new_path: Path, meta: dict[str, object]) -> None:
    """Run post-rename HTTP endpoint. Local command hooks are rejected."""
    _hook_env(old_path, new_path, meta)
    cmd = (hook_cmd or "").strip()
    if not cmd:
        return
    try:
        if "://" not in cmd:
            _warn_local_hook_disabled()
            return
        validated_url = _validated_hook_url(cmd)
        _post_hook_http(validated_url, old_path, new_path, meta)
    except ValueError as exc:
        logger.warning("Post-rename hook URL rejected: %s.", exc)
    except requests.RequestException:
        logger.warning("Post-rename hook HTTP call failed.")
    except (AttributeError, OSError, TypeError):
        logger.warning("Post-rename hook failed.")


_run_post_rename_hook = run_post_rename_hook


def _has_log_delimiter(value: str) -> bool:
    return "\t" in value or "\n" in value or "\r" in value


def _write_rename_log_entry(rename_log_path: str | Path, file_path: Path, target: Path) -> None:
    file_path_str = str(file_path)
    target_str = str(target)
    if _has_log_delimiter(file_path_str):
        logger.warning(
            "Cannot write rename log entry for %s: original path contains tab or newline "
            "characters (unsupported by the tab-delimited log format). Undo will not be "
            "available for this file.",
            file_path,
        )
        return
    if _has_log_delimiter(target_str):
        logger.warning(
            "Cannot write rename log entry for %s: target path contains tab or newline "
            "characters (unsupported by the tab-delimited log format). Undo will not be "
            "available for this file.",
            target,
        )
        return
    Path(rename_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(rename_log_path, "a", encoding="utf-8") as f:
        f.write(f"{file_path_str}\t{target}\n")


def _configured_hook_command(config: RenamerConfig) -> str:
    return (config.post_rename_hook or "").strip() or (os.environ.get("AI_PDF_RENAMER_POST_RENAME_HOOK") or "").strip()


def _apply_post_rename_actions(
    config: RenamerConfig,
    action: PostRenameAction,
) -> None:
    """Write rename log, PDF metadata, and export row after a successful rename."""
    if config.export_metadata_path:
        _append_export_row(action.export_rows, file_path=action.file_path, target=action.target, meta=action.meta)
    if config.rename_log_path:
        _write_rename_log_entry(config.rename_log_path, action.file_path, action.target)
    if config.write_pdf_metadata:
        _write_pdf_title_metadata(action.target, action.current_base)
    hook_cmd = _configured_hook_command(config)
    if hook_cmd:
        _run_post_rename_hook(hook_cmd, action.file_path, action.target, action.meta)


def _make_post_rename_success_callback(
    config: RenamerConfig,
    meta: dict[str, object] | None,
    export_rows: list[dict[str, object]],
) -> Callable[[Path, Path, str], None]:
    current_meta = meta or {}

    def _on_rename_success(_fp: Path, _target: Path, _current_base: str) -> None:
        _apply_post_rename_actions(
            config,
            PostRenameAction(_fp, _target, _current_base, current_meta, export_rows),
        )

    return _on_rename_success
