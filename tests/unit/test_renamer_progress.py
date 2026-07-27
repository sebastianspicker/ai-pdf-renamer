"""Tests for renamer_progress helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from folionym.config import OutputConfig, OutputProgressConfig, RenamerConfig
from folionym.renamer_progress import (
    _create_progress_reporter,
    _NullProgressReporter,
)


def _make_config(**kw: object) -> RenamerConfig:
    return RenamerConfig(
        output=OutputConfig(
            progress_options=OutputProgressConfig(
                progress=bool(kw.get("progress", False)),
                quiet_progress=bool(kw.get("quiet_progress", False)),
            )
        )
    )


class TestNullProgressReporter:
    def test_context_manager_returns_self(self) -> None:
        r = _NullProgressReporter()
        with r as ctx:
            assert ctx is r

    def test_update_is_noop(self) -> None:
        r = _NullProgressReporter()
        r.update(1, 10, Path("file.pdf"))  # must not raise

    def test_exit_returns_none(self) -> None:
        r = _NullProgressReporter()
        assert r.__exit__(None, None, None) is None


class TestCreateProgressReporter:
    def test_returns_null_when_progress_disabled(self) -> None:
        cfg = _make_config(progress=False, quiet_progress=False)
        reporter = _create_progress_reporter(5, cfg)
        assert isinstance(reporter, _NullProgressReporter)

    def test_returns_rich_when_progress_enabled(self) -> None:
        from folionym.renamer_progress import _RichProgressReporter

        cfg = _make_config(progress=True, quiet_progress=False)
        reporter = _create_progress_reporter(5, cfg)
        assert isinstance(reporter, _RichProgressReporter)

    def test_returns_rich_when_quiet_progress_enabled(self) -> None:
        from folionym.renamer_progress import _RichProgressReporter

        cfg = _make_config(progress=False, quiet_progress=True)
        reporter = _create_progress_reporter(5, cfg)
        assert isinstance(reporter, _RichProgressReporter)

    def test_falls_back_to_null_on_import_error(self) -> None:
        cfg = _make_config(progress=True)
        with patch("folionym.renamer_progress._RichProgressReporter", side_effect=ImportError):
            reporter = _create_progress_reporter(5, cfg)
        assert isinstance(reporter, _NullProgressReporter)

    def test_rich_reporter_context_manager(self) -> None:
        from folionym.renamer_progress import _RichProgressReporter

        cfg = _make_config(progress=True)
        reporter = _create_progress_reporter(5, cfg)
        assert isinstance(reporter, _RichProgressReporter)
        with reporter as ctx:
            ctx.update(1, 5, Path("a.pdf"))
            ctx.update(5, 5, Path("z.pdf"))
