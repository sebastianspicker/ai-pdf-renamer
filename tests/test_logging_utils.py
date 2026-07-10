from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from ai_pdf_renamer.logging_utils import setup_logging


@contextlib.contextmanager
def _isolated_root_logger() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        _clear_root_handlers(root)
        yield root
    finally:
        _clear_root_handlers(root)
        for handler in old_handlers:
            root.addHandler(handler)
        root.setLevel(old_level)


def _clear_root_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


def _managed_file_handlers(root: logging.Logger) -> list[logging.FileHandler]:
    return [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.FileHandler) and getattr(handler, "_ai_pdf_renamer_managed", False)
    ]


def _console_handlers(root: logging.Logger) -> list[logging.StreamHandler]:
    return [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]


def test_setup_logging_adds_console_when_only_file_handler_exists(tmp_path) -> None:
    with _isolated_root_logger() as root:
        pre_file = logging.FileHandler(str(tmp_path / "pre.log"), encoding="utf-8")
        root.addHandler(pre_file)

        setup_logging(log_file=tmp_path / "new.log", level=logging.INFO)

        has_console = bool(_console_handlers(root))
        has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
        assert has_console is True
        assert has_file is True


def test_setup_logging_reconfigures_managed_handlers(tmp_path) -> None:
    with _isolated_root_logger() as root:
        first_log = tmp_path / "first.log"
        second_log = tmp_path / "second.log"

        setup_logging(log_file=first_log, level=logging.WARNING)
        setup_logging(log_file=second_log, level=logging.DEBUG)

        console_handlers = _console_handlers(root)
        managed_file_handlers = _managed_file_handlers(root)

        assert len(console_handlers) == 1
        assert console_handlers[0].level == logging.DEBUG
        assert len(managed_file_handlers) == 1
        assert managed_file_handlers[0].level == logging.DEBUG
        assert Path(managed_file_handlers[0].baseFilename) == second_log


def test_setup_logging_reuses_relative_managed_file_handler(tmp_path) -> None:
    old_cwd = Path.cwd()
    with _isolated_root_logger() as root:
        relative_log = Path("logs") / "relative.log"
        expected_log = (tmp_path / relative_log).resolve()
        expected_log.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.chdir(tmp_path)
            setup_logging(log_file=relative_log, level=logging.INFO)
            first_handler = _managed_file_handlers(root)[0]

            setup_logging(log_file=relative_log, level=logging.DEBUG)
            managed_file_handlers = _managed_file_handlers(root)
        finally:
            os.chdir(old_cwd)

        assert len(managed_file_handlers) == 1
        assert managed_file_handlers[0] is first_handler
        assert managed_file_handlers[0].level == logging.DEBUG
        assert Path(managed_file_handlers[0].baseFilename) == expected_log
