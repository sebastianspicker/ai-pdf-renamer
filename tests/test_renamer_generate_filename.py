from __future__ import annotations

import re
from datetime import date

import pytest

from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
from ai_pdf_renamer.renamer import RenamerConfig, generate_filename
from ai_pdf_renamer.text_utils import Stopwords


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
        config=RenamerConfig(language="de", desired_case="kebabCase"),
        llm_client=object(),  # unused due to monkeypatching
        heuristic_scorer=scorer,
        stopwords=stopwords,
        today=date(2000, 1, 1),
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
        config=RenamerConfig(language="de", desired_case="camelCase"),
        llm_client=object(),
        heuristic_scorer=HeuristicScorer(rules=[]),
        stopwords=Stopwords(words=set()),
        today=date(2000, 1, 1),
    )
    assert name.startswith("20240201")
    assert "MyCategory" in name


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
