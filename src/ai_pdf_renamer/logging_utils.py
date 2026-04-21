from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_MANAGED_HANDLER_ATTR = "_ai_pdf_renamer_managed"


class StructuredLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line (for CI/monitoring)."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception as exc:
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
        except Exception as exc:
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

    use_structured = os.environ.get("AI_PDF_RENAMER_STRUCTURED_LOGS", "").strip() in (
        "1",
        "true",
        "yes",
    )
    if use_structured:
        formatter: logging.Formatter = StructuredLogFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

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

    try:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        managed_file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler) and getattr(h, _MANAGED_HANDLER_ATTR, False)
        ]
        for file_handler in managed_file_handlers:
            if Path(file_handler.baseFilename).resolve() != log_path:
                root.removeHandler(file_handler)
                file_handler.close()
                continue
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            return

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MANAGED_HANDLER_ATTR, True)
        root.addHandler(file_handler)
    except OSError as exc:
        # P2: Log to stderr as fallback instead of swallowing silently
        import sys

        print(f"Warning: Could not create file handler for {log_file}: {exc}", file=sys.stderr)
