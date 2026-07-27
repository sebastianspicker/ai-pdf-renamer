"""Verify filename generation dependencies, caching, and rename integration."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from folionym.config import (
    LLMConfig,
    LLMContentConfig,
    LLMRuntimeConfig,
    OutputConfig,
    OutputNamingConfig,
    RenamerConfig,
)
from folionym.filename import FilenameGenerationRequest, generate_filename
from folionym.filename_models import FilenameGenerationContext, FilenameGenerationDependencies
from folionym.heuristics import HeuristicRule, HeuristicScorer
from folionym.text_utils import Stopwords

REFERENCE_TODAY = date(2026, 4, 8)


def _config(  # noqa: PLR0913
    *,
    use_llm: bool = True,
    use_single_llm_call: bool = True,
    simple_naming_mode: bool = False,
    cache_dir: str | Path | None = None,
    language: str = "de",
    desired_case: str = "kebabCase",
) -> RenamerConfig:
    return RenamerConfig(
        llm=LLMConfig(
            runtime=LLMRuntimeConfig(
                use_llm=use_llm,
                use_single_llm_call=use_single_llm_call,
                simple_naming_mode=simple_naming_mode,
            ),
            content=LLMContentConfig(cache_dir=cache_dir),
        ),
        output=OutputConfig(naming=OutputNamingConfig(language=language, desired_case=desired_case)),
    )


def test_generate_filename_loads_fallible_dependencies_before_owned_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import folionym.filename as filename_module

    factory_called = False

    def create_client(_config: RenamerConfig) -> object:
        nonlocal factory_called
        factory_called = True
        return MagicMock()

    def fail_scorer(_language: str) -> HeuristicScorer:
        raise ValueError("heuristic data unavailable")

    monkeypatch.setattr(filename_module, "create_llm_client_from_config", create_client)
    monkeypatch.setattr(filename_module, "default_heuristic_scorer", fail_scorer)

    with pytest.raises(ValueError, match="heuristic data unavailable"):
        filename_module.generate_filename(
            "Invoice",
            FilenameGenerationRequest(config=_config(use_llm=True)),
        )

    assert factory_called is False


def test_generate_filename_closes_owned_client_when_generation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import folionym.filename as filename_module

    client = MagicMock()
    client.model = "test-model"
    client.base_url = "http://localhost"

    def fail_date(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("date resolution failed")

    monkeypatch.setattr(filename_module, "create_llm_client_from_config", lambda _config: client)
    monkeypatch.setattr(filename_module, "_get_date_str", fail_date)

    with pytest.raises(RuntimeError, match="date resolution failed"):
        filename_module.generate_filename(
            "Invoice",
            FilenameGenerationRequest(
                config=_config(use_llm=True),
                dependencies=FilenameGenerationDependencies(
                    heuristic_scorer=HeuristicScorer(rules=[]), stopwords=Stopwords(words=set())
                ),
            ),
        )

    client.close.assert_called_once_with()


def test_generate_filename_stopwords_and_dedup(monkeypatch) -> None:
    import folionym.filename_llm_metadata as filename_llm_metadata
    import folionym.filename_metadata as filename_metadata

    monkeypatch.setattr(filename_llm_metadata, "get_document_summary", lambda *a, **k: "Some summary")
    monkeypatch.setattr(
        filename_llm_metadata,
        "get_document_keywords",
        lambda *a, **k: ["invoice", "summary", "tax"],
    )
    monkeypatch.setattr(filename_metadata, "get_document_category", lambda *a, **k: "invoice")
    monkeypatch.setattr(
        filename_llm_metadata,
        "get_final_summary_tokens",
        lambda *a, **k: ["invoice", "payment", "json"],
    )

    scorer = HeuristicScorer(
        rules=[
            HeuristicRule(
                pattern=re.compile("invoice", re.IGNORECASE),
                category="invoice",
                score=10,
            )
        ]
    )
    stopwords = Stopwords(words={"summary", "json"})

    name, _ = generate_filename(
        "Invoice dated 2024-01-09",
        FilenameGenerationRequest(
            config=_config(language="de", desired_case="kebabCase", use_single_llm_call=False),
            dependencies=FilenameGenerationDependencies(
                llm_client=object(), heuristic_scorer=scorer, stopwords=stopwords
            ),
            context=FilenameGenerationContext(today=REFERENCE_TODAY),
        ),
    )

    assert name == "20240109-invoice-tax-payment"


def test_generate_filename_camel_case(monkeypatch) -> None:
    import folionym.filename_llm_metadata as filename_llm_metadata
    import folionym.filename_metadata as filename_metadata

    monkeypatch.setattr(filename_llm_metadata, "get_document_summary", lambda *a, **k: "x")
    monkeypatch.setattr(filename_llm_metadata, "get_document_keywords", lambda *a, **k: ["Foo Bar"])
    monkeypatch.setattr(filename_metadata, "get_document_category", lambda *a, **k: "My Category")
    monkeypatch.setattr(filename_llm_metadata, "get_final_summary_tokens", lambda *a, **k: ["Baz"])

    name, _ = generate_filename(
        "2024-02-01",
        FilenameGenerationRequest(
            config=_config(language="de", desired_case="camelCase", use_single_llm_call=False),
            dependencies=FilenameGenerationDependencies(
                llm_client=object(), heuristic_scorer=HeuristicScorer(rules=[]), stopwords=Stopwords(words=set())
            ),
            context=FilenameGenerationContext(today=REFERENCE_TODAY),
        ),
    )
    assert name.startswith("20240201")
    assert "MyCategory" in name


class FakeTailSensitiveClient:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None, response_format=None
    ) -> str:
        self.calls += 1
        return f"tail_sensitive_name_{self.calls}"


def _write_same_size_tail(path, tail: bytes) -> None:
    path.write_bytes(b"A" * 65_536 + b"B" * 65_536 + tail * 1_024)


def _tail_sensitive_filename(source_path, config: RenamerConfig, client: FakeTailSensitiveClient) -> str:
    name, _ = generate_filename(
        "Invoice dated 2024-01-09",
        FilenameGenerationRequest(
            config=config,
            dependencies=FilenameGenerationDependencies(
                llm_client=client, heuristic_scorer=HeuristicScorer(rules=[]), stopwords=Stopwords(words=set())
            ),
            context=FilenameGenerationContext(today=REFERENCE_TODAY, source_path=source_path),
        ),
    )
    return name


def test_generate_filename_invalidates_cache_when_same_size_source_tail_changes(tmp_path) -> None:
    source_path = tmp_path / "doc.pdf"
    _write_same_size_tail(source_path, b"C")
    config = _config(
        language="en",
        simple_naming_mode=True,
        cache_dir=tmp_path / "cache",
    )
    client = FakeTailSensitiveClient()

    first_name = _tail_sensitive_filename(source_path, config, client)
    _write_same_size_tail(source_path, b"D")
    second_name = _tail_sensitive_filename(source_path, config, client)

    assert first_name == "20240109-tail_sensitive_name_1"
    assert second_name == "20240109-tail_sensitive_name_2"
    assert client.calls == 2


def test_rename_skips_empty_pdf(monkeypatch, tmp_path) -> None:
    import folionym.renamer as renamer_mod

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    called = {"count": 0}

    def _gen(*a, **k):
        called["count"] += 1
        return "should-not", {}

    monkeypatch.setattr(renamer_mod, "_generate_filename", _gen)
    monkeypatch.setattr(renamer_mod, "produce_rename_results", lambda *a, **k: [(pdf_path, None, None, None)])

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=RenamerConfig())

    assert pdf_path.exists()
    assert called["count"] == 0
    assert not (tmp_path / "should-not.pdf").exists()


def test_rename_invalid_directory_raises() -> None:
    from folionym.renamer import rename_pdfs_in_directory

    missing = "tests/this-directory-does-not-exist"
    with pytest.raises(FileNotFoundError) as excinfo:
        rename_pdfs_in_directory(missing, config=RenamerConfig())

    assert "Directory does not exist" in str(excinfo.value)


def test_rename_collision_suffixes(monkeypatch, tmp_path) -> None:
    import folionym.renamer as renamer_mod

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"content")

    (tmp_path / "20240101-report.pdf").write_bytes(b"existing")
    (tmp_path / "20240101-report_1.pdf").write_bytes(b"existing")

    monkeypatch.setattr(
        renamer_mod,
        "produce_rename_results",
        lambda *a, **k: [(pdf_path, "20240101-report", {}, None)],
    )

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=RenamerConfig())

    assert not pdf_path.exists()
    assert (tmp_path / "20240101-report_2.pdf").exists()
