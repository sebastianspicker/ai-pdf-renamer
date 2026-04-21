from __future__ import annotations

import re
from datetime import date

import pytest

from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
from ai_pdf_renamer.renamer import RenamerConfig, generate_filename
from ai_pdf_renamer.text_utils import Stopwords

REFERENCE_TODAY = date(2026, 4, 8)


def test_generate_filename_stopwords_and_dedup(monkeypatch) -> None:
    import ai_pdf_renamer.filename as filename_mod

    monkeypatch.setattr(filename_mod, "get_document_summary", lambda *a, **k: "Some summary")
    monkeypatch.setattr(
        filename_mod,
        "get_document_keywords",
        lambda *a, **k: ["invoice", "summary", "tax"],
    )
    monkeypatch.setattr(filename_mod, "get_document_category", lambda *a, **k: "invoice")
    monkeypatch.setattr(
        filename_mod,
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
        config=RenamerConfig(language="de", desired_case="kebabCase", use_single_llm_call=False),
        llm_client=object(),  # unused due to monkeypatching
        heuristic_scorer=scorer,
        stopwords=stopwords,
        today=REFERENCE_TODAY,
    )

    assert name == "20240109-invoice-tax-payment"


def test_generate_filename_camel_case(monkeypatch) -> None:
    import ai_pdf_renamer.filename as filename_mod

    monkeypatch.setattr(filename_mod, "get_document_summary", lambda *a, **k: "x")
    monkeypatch.setattr(filename_mod, "get_document_keywords", lambda *a, **k: ["Foo Bar"])
    monkeypatch.setattr(filename_mod, "get_document_category", lambda *a, **k: "My Category")
    monkeypatch.setattr(filename_mod, "get_final_summary_tokens", lambda *a, **k: ["Baz"])

    name, _ = generate_filename(
        "2024-02-01",
        config=RenamerConfig(language="de", desired_case="camelCase", use_single_llm_call=False),
        llm_client=object(),
        heuristic_scorer=HeuristicScorer(rules=[]),
        stopwords=Stopwords(words=set()),
        today=REFERENCE_TODAY,
    )
    assert name.startswith("20240201")
    assert "MyCategory" in name


def test_generate_filename_invalidates_cache_when_same_size_source_tail_changes(tmp_path) -> None:
    class FakeClient:
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None, response_format=None
        ):
            self.calls += 1
            return f"tail_sensitive_name_{self.calls}"

    source_path = tmp_path / "doc.pdf"
    source_path.write_bytes(b"A" * 65_536 + b"B" * 65_536 + b"C" * 1_024)

    config = RenamerConfig(
        language="en",
        simple_naming_mode=True,
        cache_dir=tmp_path / "cache",
    )
    client = FakeClient()

    first_name, _ = generate_filename(
        "Invoice dated 2024-01-09",
        config=config,
        llm_client=client,
        heuristic_scorer=HeuristicScorer(rules=[]),
        stopwords=Stopwords(words=set()),
        today=REFERENCE_TODAY,
        source_path=source_path,
    )

    source_path.write_bytes(b"A" * 65_536 + b"B" * 65_536 + b"D" * 1_024)

    second_name, _ = generate_filename(
        "Invoice dated 2024-01-09",
        config=config,
        llm_client=client,
        heuristic_scorer=HeuristicScorer(rules=[]),
        stopwords=Stopwords(words=set()),
        today=REFERENCE_TODAY,
        source_path=source_path,
    )

    assert first_name == "20240109-tail_sensitive_name_1"
    assert second_name == "20240109-tail_sensitive_name_2"
    assert client.calls == 2


def test_rename_skips_empty_pdf(monkeypatch, tmp_path) -> None:
    import ai_pdf_renamer.renamer as renamer_mod

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    monkeypatch.setattr(renamer_mod, "pdf_to_text", lambda *a, **k: "")

    called = {"count": 0}

    def _gen(*a, **k):
        called["count"] += 1
        return "should-not", {}

    monkeypatch.setattr(renamer_mod, "generate_filename", _gen)

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=renamer_mod.RenamerConfig())

    assert pdf_path.exists()
    assert called["count"] == 0
    assert not (tmp_path / "should-not.pdf").exists()


def test_rename_invalid_directory_raises() -> None:
    from ai_pdf_renamer.renamer import RenamerConfig, rename_pdfs_in_directory

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

    monkeypatch.setattr(renamer_mod, "pdf_to_text", lambda *a, **k: "content")
    monkeypatch.setattr(renamer_mod, "generate_filename", lambda *a, **k: ("20240101-report", {}))

    renamer_mod.rename_pdfs_in_directory(tmp_path, config=renamer_mod.RenamerConfig())

    assert not pdf_path.exists()
    assert (tmp_path / "20240101-report_2.pdf").exists()
