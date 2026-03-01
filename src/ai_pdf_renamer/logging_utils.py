from __future__ import annotations

import json
import logging
import os
from pathlib import Path


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

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            root.debug("Could not create file handler for %s", log_file)
