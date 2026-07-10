from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_MANAGED_HANDLER_ATTR = "_ai_pdf_renamer_managed"
_LOG_FORMATTER_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


class StructuredLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line (for CI/monitoring)."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except _LOG_FORMATTER_EXCEPTIONS as exc:
            message = f"<message unavailable: {exc!s}>"
        try:
            payload = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": message,
            }
            if record.name != "root":
                payload["logger"] = record.name
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)
        except _LOG_FORMATTER_EXCEPTIONS as exc:
            return json.dumps(
                {
                    "level": "WARNING",
                    "message": f"Log formatter error: {exc!s}",
                },
                ensure_ascii=False,
            )


def setup_logging(*, log_file: str | Path = "error.log", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    formatter = _log_formatter()
    _setup_console_logging(root, formatter, level)
    _setup_file_logging(root, formatter, log_file, level)


def _log_formatter() -> logging.Formatter:
    use_structured = os.environ.get("AI_PDF_RENAMER_STRUCTURED_LOGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if use_structured:
        return StructuredLogFormatter()
    return logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def _setup_console_logging(root: logging.Logger, formatter: logging.Formatter, level: int) -> None:
    console_handlers = [
        h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    for console_handler in console_handlers:
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
    if not console_handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_ATTR, True)
        root.addHandler(console_handler)


def _managed_file_handlers(root: logging.Logger) -> list[logging.FileHandler]:
    return [h for h in root.handlers if isinstance(h, logging.FileHandler) and getattr(h, _MANAGED_HANDLER_ATTR, False)]


def _reuse_or_remove_file_handlers(
    root: logging.Logger,
    formatter: logging.Formatter,
    log_path: Path,
    level: int,
) -> bool:
    for file_handler in _managed_file_handlers(root):
        if Path(file_handler.baseFilename).resolve() != log_path:
            root.removeHandler(file_handler)
            file_handler.close()
            continue
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        return True
    return False


def _add_file_handler(root: logging.Logger, formatter: logging.Formatter, log_path: Path, level: int) -> None:
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_ATTR, True)
    root.addHandler(file_handler)


def _setup_file_logging(
    root: logging.Logger,
    formatter: logging.Formatter,
    log_file: str | Path,
    level: int,
) -> None:
    try:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if _reuse_or_remove_file_handlers(root, formatter, log_path, level):
            return
        _add_file_handler(root, formatter, log_path, level)
    except OSError as exc:
        # Logging setup may run before the normal console is configured; stderr
        # keeps the failure visible without aborting the CLI.
        import sys

        print(f"Warning: Could not create file handler for {log_file}: {exc}", file=sys.stderr)
