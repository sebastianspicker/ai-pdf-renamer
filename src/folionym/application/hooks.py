"""Post-rename side effects for the batch renamer."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ..infrastructure.http import validate_http_endpoint
from ..settings import RenamerConfig
from .artifacts import _append_export_row, _write_rename_log_entry

logger = logging.getLogger("folionym.renamer")


@dataclass
class PostRenameAction:
    """Inputs for post-rename logging, metadata, export, and hook side effects."""

    file_path: Path
    target: Path
    current_base: str
    meta: dict[str, object]
    export_rows: list[dict[str, object]]


def _cleanup_temp_pdf(temp_pdf: Path | None) -> None:
    """Remove a temporary PDF when a metadata write cannot be completed."""
    if temp_pdf is not None:
        with contextlib.suppress(OSError):
            temp_pdf.unlink()


def _create_pdf_metadata_temp(pdf_path: Path) -> Path:
    """Create a same-directory temporary PDF and close its file descriptor."""
    import tempfile

    fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=pdf_path.parent)
    temp_pdf = Path(temp_path)
    try:
        os.close(fd)
    except OSError:
        _cleanup_temp_pdf(temp_pdf)
        raise
    return temp_pdf


def _save_pdf_to_temp(doc: Any, temp_pdf: Path, encryption_keep: object | None) -> None:
    """Save metadata to a temporary PDF, retrying old PyMuPDF signatures once."""
    save_options: dict[str, object] = {"incremental": False}
    if encryption_keep is not None:
        save_options["encryption"] = encryption_keep

    last_type_error: TypeError | None = None
    for options in (save_options, {"incremental": False}):
        try:
            doc.save(str(temp_pdf), **options)
            return
        except TypeError as exc:
            # Older PyMuPDF builds may expose incompatible save signatures.
            last_type_error = exc
        except AttributeError, OSError, RuntimeError, ValueError:
            _cleanup_temp_pdf(temp_pdf)
            raise

    _cleanup_temp_pdf(temp_pdf)
    if last_type_error is not None:
        raise last_type_error


def _replace_nonempty_temp_pdf(temp_pdf: Path, pdf_path: Path) -> None:
    """Atomically replace a PDF only after its temporary replacement has data."""
    if temp_pdf.stat().st_size == 0:
        raise OSError(f"Temporary PDF file is empty (possible disk full): {temp_pdf}")
    # Atomic replace (os.replace is atomic on POSIX, best-effort on Windows).
    os.replace(temp_pdf, pdf_path)


def _write_pdf_title_metadata(pdf_path: Path, title: str) -> None:
    """Write /Title metadata to PDF (PyMuPDF). No-op if fitz unavailable or on error.

    Writes to a temporary file first, then atomically replaces the original to
    prevent corruption under concurrent access.
    """
    temp_pdf: Path | None = None
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            doc.set_metadata({"title": title or pdf_path.stem})
            temp_pdf = _create_pdf_metadata_temp(pdf_path)
            encryption_keep = getattr(fitz, "PDF_ENCRYPT_KEEP", None)
            _save_pdf_to_temp(doc, temp_pdf, encryption_keep)
        finally:
            doc.close()
        _replace_nonempty_temp_pdf(temp_pdf, pdf_path)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _cleanup_temp_pdf(temp_pdf)
        logger.warning("Could not write PDF metadata for %s: %s", pdf_path, exc)


# C0 control chars (incl. NUL) must be stripped from env-var values passed to hook commands.
_HOOK_ENV_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _hook_env(old_path: Path, new_path: Path, meta: dict[str, object]) -> dict[str, str]:
    """Build sanitized hook environment values, using empty metadata when JSON encoding fails."""
    # Sanitize path strings: strip C0 control characters (\x00-\x1f) and DEL (\x7f)
    # from values passed via env vars to hook commands. NUL (\x00) is especially
    # important as it truncates C strings, but other control chars can also cause issues.
    old_value = _HOOK_ENV_CONTROL_RE.sub("", str(old_path))
    new_value = _HOOK_ENV_CONTROL_RE.sub("", str(new_path))
    env = {**os.environ, "FOLIONYM_OLD_PATH": old_value, "FOLIONYM_NEW_PATH": new_value}
    # fmt: off
    try:
        meta_json = json.dumps(meta, default=str)
    except (TypeError, ValueError):
        meta_json = "{}"
    # fmt: on
    env["FOLIONYM_META"] = meta_json
    return env


def _validated_hook_url(value: str) -> str:
    """Return a safe HTTP(S) hook URL or raise a value-only validation error."""
    endpoint = validate_http_endpoint(value)
    if endpoint.scheme == "http" and not endpoint.is_literal_loopback:
        raise ValueError("unencrypted plain HTTP requires a literal loopback IP address")
    return endpoint.url


def _post_hook_http(cmd: str, old_path: Path, new_path: Path, meta: dict[str, object]) -> None:
    """POST the rename payload without proxy environment or redirects and raise on HTTP errors."""
    payload = {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "meta": meta,
    }
    with requests.Session() as session:
        session.trust_env = False
        resp = session.post(cmd, json=payload, timeout=10, allow_redirects=False)
        if 300 <= resp.status_code < 400:
            raise requests.TooManyRedirects("post-rename hook redirects are disabled")
        resp.raise_for_status()


def _warn_local_hook_disabled() -> None:
    """Warn that local command hooks are disabled."""
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
    # fmt: off
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
    # fmt: on


_run_post_rename_hook = run_post_rename_hook


def _configured_hook_command(config: RenamerConfig) -> str:
    """Return the configured hook URL, preferring explicit configuration over the environment."""
    configured = (config.output.hooks.post_rename_hook or "").strip()
    return configured or (os.environ.get("FOLIONYM_POST_RENAME_HOOK") or "").strip()


def _apply_post_rename_actions(
    config: RenamerConfig,
    action: PostRenameAction,
) -> None:
    """Write rename log, PDF metadata, and export row after a successful rename."""
    paths = config.output.paths
    if paths.export_metadata_path:
        _append_export_row(action.export_rows, file_path=action.file_path, target=action.target, meta=action.meta)
    if paths.rename_log_path:
        _write_rename_log_entry(paths.rename_log_path, action.file_path, action.target)
    if config.output.mode.write_pdf_metadata:
        _write_pdf_title_metadata(action.target, action.current_base)
    hook_cmd = _configured_hook_command(config)
    if hook_cmd:
        _run_post_rename_hook(hook_cmd, action.file_path, action.target, action.meta)


def _make_post_rename_success_callback(
    config: RenamerConfig,
    meta: dict[str, object] | None,
    export_rows: list[dict[str, object]],
) -> Callable[[Path, Path, str], None]:
    """Capture metadata and export state in a post-rename side-effect callback."""
    current_meta = meta or {}

    def _on_rename_success(_fp: Path, _target: Path, _current_base: str) -> None:
        """Apply configured post-rename actions after a completed rename."""
        _apply_post_rename_actions(
            config,
            PostRenameAction(_fp, _target, _current_base, current_meta, export_rows),
        )

    return _on_rename_success
