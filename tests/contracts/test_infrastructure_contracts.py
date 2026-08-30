"""Contracts for the low-level dependency boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from folionym.infrastructure.http import validate_http_endpoint
from folionym.infrastructure.private_io import atomic_write_private_text, open_private_append_text
from folionym.infrastructure.resources import data_path, load_json_data


def test_infrastructure_resources_and_http_validation_preserve_runtime_semantics() -> None:
    resource = data_path("meta_stopwords.json")
    assert resource.is_file()
    assert isinstance(load_json_data(resource), dict)

    endpoint = validate_http_endpoint(" https://127.0.0.1:8080/v1 ")
    assert (endpoint.url, endpoint.scheme, endpoint.hostname, endpoint.is_literal_loopback) == (
        "https://127.0.0.1:8080/v1",
        "https",
        "127.0.0.1",
        True,
    )
    with pytest.raises(ValueError, match="must not include credentials"):
        validate_http_endpoint("https://user@example.test/v1")


@pytest.mark.skipif(os.name == "nt", reason="POSIX owner-only permissions contract")
def test_private_io_keeps_replacement_atomic_and_append_streams_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "private.txt"
    target.write_text("original", encoding="utf-8")

    def fail_before_write(_temporary_path: Path) -> None:
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        atomic_write_private_text(target, "replacement", before_write=fail_before_write)
    assert target.read_text(encoding="utf-8") == "original"

    atomic_write_private_text(target, "replacement")
    with open_private_append_text(target) as handle:
        handle.write("+append")
    assert target.read_text(encoding="utf-8") == "replacement+append"
    assert target.stat().st_mode & 0o777 == 0o600


def test_low_level_consumers_do_not_depend_on_application_implementations() -> None:
    package_root = Path(__file__).parents[2] / "src" / "folionym"
    legacy_modules = (
        "data_paths.py",
        "http_url.py",
        "logging_utils.py",
        "recoverable_errors.py",
        "application/_private_io.py",
    )
    assert all(not (package_root / module).exists() for module in legacy_modules)

    application_imports = "application."
    for module in ("infrastructure/private_io.py", "infrastructure/logging.py", "infrastructure/files.py"):
        assert application_imports not in (package_root / module).read_text(encoding="utf-8")
    assert "application.discovery" not in (package_root / "extraction/pipeline.py").read_text(encoding="utf-8")
