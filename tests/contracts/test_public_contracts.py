"""Public import and command contracts independent of installed console scripts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from folionym import rename_ops, renamer
from folionym.config import RenamerConfig
from folionym.filename import FilenameGenerationRequest, generate_filename
from folionym.heuristics import CategoryCombineParams
from folionym.interfaces.cli import main as cli_main
from folionym.interfaces.cli.parser import build_parser
from folionym.interfaces.cli.undo import main as undo_main
from folionym.interfaces.tui import main as tui_main
from folionym.interfaces.web.cli import main as web_main
from folionym.llm.models import VisionCompletionOptions
from folionym.naming.models import FilenameGenerationDependencies


def test_documented_python_imports_remain_owned_by_their_public_modules() -> None:
    """Keep the supported import surface stable without reviving compatibility aliases."""
    assert RenamerConfig.__name__ == "RenamerConfig"
    assert FilenameGenerationRequest.__name__ == "FilenameGenerationRequest"
    assert CategoryCombineParams.__name__ == "CategoryCombineParams"
    assert callable(generate_filename)
    assert all(
        callable(getattr(renamer, name))
        for name in (
            "process_one_file",
            "suggest_rename_for_file",
            "produce_rename_results",
            "rename_pdfs_in_directory",
            "run_watch_loop",
        )
    )
    assert not hasattr(renamer, "RenamerConfig")
    assert not hasattr(renamer, "generate_filename")


def test_filename_facade_owns_a_lazy_llm_client_for_direct_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RenamerConfig()
    factory_calls: list[RenamerConfig] = []
    generated_with: list[object] = []
    closed: list[bool] = []

    class OwnedClient:
        def close(self) -> None:
            closed.append(True)

    owned_client = OwnedClient()

    def create_client(received_config: RenamerConfig) -> OwnedClient:
        factory_calls.append(received_config)
        return owned_client

    def generate(_content: str, request: FilenameGenerationRequest) -> tuple[str, dict[str, object]]:
        generated_with.append(request.llm_client)
        return "20260827-document.pdf", {"source": "llm"}

    monkeypatch.setattr("folionym.llm.http.create_llm_client_from_config", create_client)
    monkeypatch.setattr("folionym.filename._generate_filename", generate)

    assert generate_filename("invoice", FilenameGenerationRequest(config=config)) == (
        "20260827-document.pdf",
        {"source": "llm"},
    )
    assert factory_calls == [config]
    assert generated_with == [owned_client]
    assert closed == [True]


def test_filename_facade_preserves_an_injected_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RenamerConfig()
    generated_with: list[object] = []
    closed: list[bool] = []

    class InjectedClient:
        model = "injected"
        base_url = ""

        def complete(
            self,
            _prompt: str,
            *,
            temperature: float = 0.0,
            max_tokens: int | None = None,
            response_format: dict[str, str] | None = None,
        ) -> str:
            return ""

        def complete_vision(
            self,
            _image_b64: str,
            _prompt: str,
            options: VisionCompletionOptions | None = None,
        ) -> str:
            return ""

        def close(self) -> None:
            closed.append(True)

    injected_client = InjectedClient()

    def should_not_construct(_config: RenamerConfig) -> object:
        raise AssertionError("injected clients must not trigger HTTP construction")

    def generate(_content: str, request: FilenameGenerationRequest) -> tuple[str, dict[str, object]]:
        generated_with.append(request.llm_client)
        return "20260827-document.pdf", {}

    monkeypatch.setattr("folionym.llm.http.create_llm_client_from_config", should_not_construct)
    monkeypatch.setattr("folionym.filename._generate_filename", generate)

    request = FilenameGenerationRequest(
        config=config,
        dependencies=FilenameGenerationDependencies(llm_client=injected_client),
    )
    assert generate_filename("invoice", request) == ("20260827-document.pdf", {})
    assert generated_with == [injected_client]
    assert closed == []


def test_rename_ops_facade_has_the_complete_supported_export_set() -> None:
    expected = {
        "FILENAME_RESERVED_WIN",
        "FILENAME_UNSAFE_RE",
        "MAX_LLM_FILENAME_LEN",
        "MAX_RENAME_RETRIES",
        "RenameApplyOptions",
        "RenameAttemptState",
        "RenameRetryContext",
        "apply_single_rename",
        "is_path_within",
        "sanitize_filename_base",
        "sanitize_filename_from_llm",
    }
    assert set(rename_ops.__all__) == expected
    assert all(hasattr(rename_ops, export) for export in expected)


def test_main_cli_help_and_parser_stay_usable_without_installed_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0
    assert "--validate-config" in capsys.readouterr().out

    parsed = build_parser().parse_args(["--dir", "incoming", "--dry-run", "--no-llm"])
    assert (parsed.dirs, parsed.dry_run, parsed.use_llm) == (["incoming"], True, False)


@pytest.mark.parametrize(
    ("entrypoint", "argv", "expected"),
    [
        (undo_main, ["--help"], "folionym-undo"),
        (web_main, ["--help"], "--no-open"),
    ],
)
def test_argument_parsing_help_for_secondary_console_entrypoints(
    entrypoint: Callable[[list[str] | None], None], argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        entrypoint(argv)
    assert exc_info.value.code == 0
    assert expected in capsys.readouterr().out


def test_tui_console_entrypoint_starts_its_app_without_using_script_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []

    class FakeTui:
        def run(self) -> None:
            started.append(True)

    monkeypatch.setattr("folionym.interfaces.tui.app.FolionymTUI", FakeTui)
    tui_main()
    assert started == [True]
