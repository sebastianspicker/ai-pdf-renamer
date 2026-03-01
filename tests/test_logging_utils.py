from __future__ import annotations

import logging

from ai_pdf_renamer.logging_utils import setup_logging


def test_setup_logging_adds_console_when_only_file_handler_exists(tmp_path) -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

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
            try:
                h.close()
            except Exception:
                pass
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)
