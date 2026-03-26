from __future__ import annotations

from ai_pdf_renamer import data_paths


def test_data_path_falls_back_to_package_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)

    path = data_paths.data_path("meta_stopwords.json")

    assert path.exists()
    assert path.name == "meta_stopwords.json"
    assert path.parent.name == "data"


def test_data_dir_uses_packaged_data_when_repo_root_not_found(monkeypatch) -> None:
    module_file = data_paths.Path(data_paths.__file__).resolve()
    expected = (module_file.parent / "data").resolve()

    monkeypatch.setattr(data_paths, "_discover_repo_root", lambda start=None: None)
    monkeypatch.setattr(data_paths.Path, "cwd", classmethod(lambda cls: data_paths.Path("/tmp/not-used-cwd")))

    resolved = data_paths.data_dir()
    assert resolved == expected


def test_data_path_ignores_non_file_candidate_and_falls_back_to_package(monkeypatch, tmp_path) -> None:
    bad_candidate = tmp_path / "meta_stopwords.json"
    bad_candidate.mkdir()

    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)
    path = data_paths.data_path("meta_stopwords.json")
    assert path.exists()
    assert path.is_file()
    assert path.parent.name == "data"


def test_category_aliases_path_ignores_directory_override(monkeypatch, tmp_path) -> None:
    bad_candidate = tmp_path / "category_aliases.json"
    bad_candidate.mkdir()

    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)
    path = data_paths.category_aliases_path()
    assert path.exists()
    assert path.is_file()
    assert path.name == "category_aliases.json"
