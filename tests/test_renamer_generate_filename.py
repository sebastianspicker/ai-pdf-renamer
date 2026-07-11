from __future__ import annotations

import re
from datetime import date

import pytest

from ai_pdf_renamer.filename import FilenameGenerationRequest
from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
from ai_pdf_renamer.renamer import RenamerConfig, generate_filename
from ai_pdf_renamer.text_utils import Stopwords

REFERENCE_TODAY = date(2026, 4, 8)


def test_generate_filename_stopwords_and_dedup(monkeypatch) -> None:
    import ai_pdf_renamer.filename_llm_metadata as filename_llm_metadata
    import ai_pdf_renamer.filename_metadata as filename_metadata

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
            config=RenamerConfig(language="de", desired_case="kebabCase", use_single_llm_call=False),
            llm_client=object(),  # unused due to monkeypatching
            heuristic_scorer=scorer,
            stopwords=stopwords,
            today=REFERENCE_TODAY,
        ),
    )

    assert name == "20240109-invoice-tax-payment"


def test_generate_filename_camel_case(monkeypatch) -> None:
    import ai_pdf_renamer.filename_llm_metadata as filename_llm_metadata
    import ai_pdf_renamer.filename_metadata as filename_metadata

    monkeypatch.setattr(filename_llm_metadata, "get_document_summary", lambda *a, **k: "x")
    monkeypatch.setattr(filename_llm_metadata, "get_document_keywords", lambda *a, **k: ["Foo Bar"])
    monkeypatch.setattr(filename_metadata, "get_document_category", lambda *a, **k: "My Category")
    monkeypatch.setattr(filename_llm_metadata, "get_final_summary_tokens", lambda *a, **k: ["Baz"])

    name, _ = generate_filename(
        "2024-02-01",
        FilenameGenerationRequest(
            config=RenamerConfig(language="de", desired_case="camelCase", use_single_llm_call=False),
            llm_client=object(),
            heuristic_scorer=HeuristicScorer(rules=[]),
            stopwords=Stopwords(words=set()),
            today=REFERENCE_TODAY,
        ),
    )
    assert name.startswith("20240201")
    assert "MyCategory" in name


def test_generate_filename_accepts_legacy_keyword_request() -> None:
    name, _ = generate_filename(
        "Invoice dated 2024-01-09",
        config=RenamerConfig(language="en", use_llm=False, desired_case="kebabCase"),
        heuristic_scorer=HeuristicScorer(
            rules=[HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=10)]
        ),
        stopwords=Stopwords(words=set()),
        today=REFERENCE_TODAY,
    )

    assert name == "20240109-invoice"


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
            llm_client=client,
            heuristic_scorer=HeuristicScorer(rules=[]),
            stopwords=Stopwords(words=set()),
            today=REFERENCE_TODAY,
            source_path=source_path,
        ),
    )
    return name


def test_generate_filename_invalidates_cache_when_same_size_source_tail_changes(tmp_path) -> None:
    source_path = tmp_path / "doc.pdf"
    _write_same_size_tail(source_path, b"C")
    config = RenamerConfig(
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
    import ai_pdf_renamer.renamer as renamer_mod

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    called = {"count": 0}

    def _gen(*a, **k):
        called["count"] += 1
        return "should-not", {}

    monkeypatch.setattr(renamer_mod, "generate_filename", _gen)
    monkeypatch.setattr(renamer_mod, "produce_rename_results", lambda *a, **k: [(pdf_path, None, None, None)])

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=renamer_mod.RenamerConfig())

    assert pdf_path.exists()
    assert called["count"] == 0
    assert not (tmp_path / "should-not.pdf").exists()


def test_rename_invalid_directory_raises() -> None:
    from ai_pdf_renamer.renamer import rename_pdfs_in_directory

    missing = "tests/this-directory-does-not-exist"
    with pytest.raises(FileNotFoundError) as excinfo:
        rename_pdfs_in_directory(missing, config=RenamerConfig())

    assert "Directory does not exist" in str(excinfo.value)


def test_rename_collision_suffixes(monkeypatch, tmp_path) -> None:
    import ai_pdf_renamer.renamer as renamer_mod

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"content")

    (tmp_path / "20240101-report.pdf").write_bytes(b"existing")
    (tmp_path / "20240101-report_1.pdf").write_bytes(b"existing")

    monkeypatch.setattr(
        renamer_mod,
        "produce_rename_results",
        lambda *a, **k: [(pdf_path, "20240101-report", {}, None)],
    )

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=renamer_mod.RenamerConfig())

    assert not pdf_path.exists()
    assert (tmp_path / "20240101-report_2.pdf").exists()
