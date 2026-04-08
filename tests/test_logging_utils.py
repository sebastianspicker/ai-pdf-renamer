from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from ai_pdf_renamer.logging_utils import setup_logging


def test_setup_logging_adds_console_when_only_file_handler_exists(tmp_path) -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()

        pre_file = logging.FileHandler(str(tmp_path / "pre.log"), encoding="utf-8")
        root.addHandler(pre_file)

        setup_logging(log_file=tmp_path / "new.log", level=logging.INFO)

        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers
        )
        has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
        assert has_console is True
        assert has_file is True
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)


def test_setup_logging_reconfigures_managed_handlers(tmp_path) -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()

        first_log = tmp_path / "first.log"
        second_log = tmp_path / "second.log"

        setup_logging(log_file=first_log, level=logging.WARNING)
        setup_logging(log_file=second_log, level=logging.DEBUG)

        console_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        managed_file_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "_ai_pdf_renamer_managed", False)
        ]

        assert len(console_handlers) == 1
        assert console_handlers[0].level == logging.DEBUG
        assert len(managed_file_handlers) == 1
        assert managed_file_handlers[0].level == logging.DEBUG
        assert Path(managed_file_handlers[0].baseFilename) == second_log
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)
