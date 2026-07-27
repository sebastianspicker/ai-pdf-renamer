"""Command-line entry point, diagnostics, and dispatch for folionym.

The CLI owns user input and operator-facing errors. Shared normalization lives in
config_resolver so CLI and TUI runs produce the same RenamerConfig shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import requests
from rich.console import Console
from rich.markup import escape

from .cli_config import ConfigLoadError, _load_config_file, _load_override_category_map, load_config_file
from .cli_parser import build_parser
from .cli_runtime import read_prompt_value, resolve_dirs
from .config import RenamerConfig
from .config_resolver import build_config
from .data_paths import data_path
from .heuristics import load_heuristic_rules
from .logging_utils import setup_logging
from .recoverable_errors import COMMON_RECOVERABLE_EXCEPTIONS
from .renamer import rename_pdfs_in_directory, run_watch_loop
from .text_utils import VALID_CASE_CHOICES

__all__ = [
    "ConfigLoadError",
    "_load_config_file",
    "_load_override_category_map",
    "load_config_file",
    "main",
    "run_doctor_checks",
]

logger = logging.getLogger(__name__)
_console = Console(stderr=True)
_RECOVERABLE_CLI_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _is_interactive() -> bool:
    """True if stdin is a TTY (interactive prompt is safe)."""
    return sys.stdin.isatty()


@dataclass(frozen=True)
class OptionPromptSpec:
    """Describe how an unset CLI option can be safely resolved."""

    choice_prompt: str | None = None
    choices: list[str] | None = None
    choice_normalize: Callable[[str], str] | None = None
    free_prompt: str | None = None


@dataclass(frozen=True)
class OptionResolutionRequest:
    """Bundle the sources and fallback policy for one CLI option."""

    attr: str
    file_defaults: dict[str, object]
    file_key: str
    default: str
    prompt: OptionPromptSpec = field(default_factory=OptionPromptSpec)


def _resolve_option(
    args: argparse.Namespace,
    request: OptionResolutionRequest,
) -> str:
    """Resolve option: getattr(args) -> file_defaults -> interactive prompt (if TTY) -> default."""
    value = getattr(args, request.attr, None)
    if value is None:
        value = request.file_defaults.get(request.file_key)
    # Treat empty string as unset when choices are defined (avoid invalid "" for language/case)
    if value == "" and request.prompt.choices is not None:
        value = None
    if value is None:
        value = _resolve_missing_option(
            request.default,
            choice_prompt=request.prompt.choice_prompt,
            choices=request.prompt.choices,
            choice_normalize=request.prompt.choice_normalize,
            free_prompt=request.prompt.free_prompt,
        )
    return str(value) if value is not None else request.default


def _read_free_prompt(free_prompt: str, default: str) -> str:
    """Read free prompt and normalize it for the next decision point."""
    return read_prompt_value(free_prompt) or default


def _resolve_missing_option(
    default: str,
    *,
    choice_prompt: str | None,
    choices: list[str] | None,
    choice_normalize: Callable[[str], str] | None,
    free_prompt: str | None,
) -> str:
    """Resolve missing option using the established precedence rules."""
    if not _is_interactive():
        return default
    if choice_prompt is not None and choices is not None:
        return _prompt_choice(
            choice_prompt,
            choices=choices,
            default=default,
            normalize=choice_normalize,
        )
    if free_prompt is not None:
        return _read_free_prompt(free_prompt, default)
    return default


def _choice_mapping(
    choices: list[str],
    default: str,
    normalize: Callable[[str], str] | None,
) -> tuple[dict[str, str], str]:
    """Map normalized prompt input to canonical choices so accepted values retain declared spelling."""
    # First occurrence wins when normalized keys collide, such as choices that
    # differ only by casing.
    mapping: dict[str, str] = {}
    for choice in choices:
        key = normalize(choice) if normalize else choice
        if key not in mapping:
            mapping[key] = choice
    default_key = normalize(default) if normalize else default
    if default_key not in mapping:
        mapping[default_key] = default
    return (mapping, default_key)


def _prompt_choice(
    prompt: str,
    *,
    choices: list[str],
    default: str,
    normalize: Callable[[str], str] | None = None,
) -> str:
    """Prompt until the user supplies a valid choice or accepts the configured default."""
    mapping, default_key = _choice_mapping(choices, default, normalize)

    while True:
        value = read_prompt_value(prompt)
        if not value:
            return mapping[default_key]
        key = normalize(value) if normalize else value
        if key in mapping:
            return mapping[key]
        print(f"Invalid choice: {value}. Valid choices: {', '.join(choices)}")


def _resolve_log_config(args: argparse.Namespace) -> tuple[str, int]:
    """Resolve log file path and log level from args and env. Returns (log_file_path, log_level)."""
    log_file = getattr(args, "log_file", None) or os.environ.get("FOLIONYM_LOG_FILE")
    if not log_file:
        _default_log_dir = Path.home() / ".local" / "share" / "folionym"
        try:
            _default_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(_default_log_dir / "error.log")
        except OSError:
            log_file = str(Path.cwd() / "error.log")
    if getattr(args, "verbose", False):
        log_level = logging.DEBUG
    elif getattr(args, "quiet", False):
        log_level = logging.WARNING
    elif getattr(args, "log_level", None):
        log_level = getattr(logging, args.log_level)
    else:
        env_level = os.environ.get("FOLIONYM_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, env_level, logging.INFO)
    return (log_file, log_level)


def _probe_llm_endpoint(
    url: str, model: str, *, label: str, fail_is_warn: bool = False, use_chat_api: bool = False
) -> bool:
    """Probe an LLM completions endpoint. Returns True if reachable."""
    con = Console()
    # Probe the same API family the run will use so --doctor does not fail
    # against servers that expose only chat completions or only text completions.
    if use_chat_api:
        from .llm_backend import _chat_url_from_completions_url

        probe_url = _chat_url_from_completions_url(url)
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    else:
        probe_url = url
        payload = {"model": model, "prompt": "ping", "max_tokens": 1, "temperature": 0.0}
    try:
        with requests.Session() as session:
            session.trust_env = False
            resp = session.post(probe_url, json=payload, timeout=3.0, allow_redirects=False)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise ValueError("Response is not OpenAI-compatible completions JSON.")
        con.print(f"  [green]OK[/green]   {label}: {probe_url}")
        return True
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        if fail_is_warn:
            con.print(f"  [yellow]WARN[/yellow] {label}: {probe_url} [dim]({exc})[/dim]")
        else:
            con.print(f"  [red]FAIL[/red] {label}: {probe_url} [dim]({exc})[/dim]")
        return False


def _run_doctor_data_checks(con: Console) -> bool:
    """Validate packaged JSON data and report useful rule/category counts."""
    ok = True
    con.print("[bold]Data Files[/bold]")
    for filename in ("heuristic_scores.json", "meta_stopwords.json"):
        try:
            path = data_path(filename)
            raw = path.read_text(encoding="utf-8")
            json.loads(raw)
            if filename == "heuristic_scores.json":
                try:
                    rules = load_heuristic_rules(path)
                except (TypeError, ValueError) as exc:
                    ok = False
                    logger.warning("Doctor check failed for %s: %s", filename, exc)
                    con.print(f"  [red]FAIL[/red] {filename}: {exc}")
                    continue
                pattern_count = len(rules)
                category_count = len({rule.category for rule in rules})
                con.print(
                    f"  [green]OK[/green]   {filename} "
                    f"[dim](patterns={pattern_count}, categories={category_count})[/dim]"
                )
            else:
                con.print(f"  [green]OK[/green]   {filename}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            ok = False
            con.print(f"  [red]FAIL[/red] {filename}: {exc}")
    return ok


def _run_doctor_dependency_checks(con: Console) -> None:
    """Report which optional runtime features are available in this environment."""
    pdf_hint = escape("-- pip install -e '.[pdf]'")
    tui_hint = escape("-- needed for folionym-tui; install with pip install -e '.[tui]'")

    con.print()
    con.print("[bold]Dependencies[/bold]")
    if importlib.util.find_spec("fitz") is not None:
        con.print("  [green]OK[/green]   PyMuPDF (fitz)")
    else:
        con.print(f"  [yellow]WARN[/yellow] PyMuPDF (fitz) [dim]{pdf_hint}[/dim]")

    if importlib.util.find_spec("ocrmypdf") is not None:
        con.print("  [green]OK[/green]   ocrmypdf")
    else:
        con.print("  [dim]INFO[/dim] ocrmypdf [dim]-- only needed for --ocr[/dim]")

    if importlib.util.find_spec("tiktoken") is not None:
        con.print("  [green]OK[/green]   tiktoken")
    else:
        con.print("  [dim]INFO[/dim] tiktoken [dim]-- only needed for token truncation[/dim]")

    if importlib.util.find_spec("textual") is not None:
        con.print("  [green]OK[/green]   textual")
    else:
        con.print(f"  [yellow]WARN[/yellow] textual [dim]{tui_hint}[/dim]")


def _build_doctor_probe_config(args: argparse.Namespace) -> RenamerConfig:
    """Build doctor probe config so callers share one validated construction path."""
    probe_raw = vars(args).copy()
    probe_raw["use_llm"] = True
    try:
        return build_config(probe_raw)
    except (
        ValueError,
        TypeError,
    ):
        return build_config(
            {
                "llm_base_url": getattr(args, "llm_base_url", None) or None,
                "llm_model": getattr(args, "llm_model", None) or None,
                "llm_timeout_s": getattr(args, "llm_timeout_s", None),
                "use_llm": True,
            }
        )


def _run_doctor_llm_checks(args: argparse.Namespace, con: Console) -> bool:
    """Probe the configured LLM endpoint and report connectivity without sending document data."""
    con.print()
    con.print("[bold]LLM Connectivity[/bold]")
    use_llm = getattr(args, "use_llm", True)
    if not use_llm:
        con.print("  [dim]Skipped (--no-llm)[/dim]")
        return True

    from .llm_backend import create_llm_client_from_config

    probe_cfg = _build_doctor_probe_config(args)
    client = create_llm_client_from_config(probe_cfg)
    con.print(f"  [dim]Backend:[/dim] http  [dim]Model:[/dim] {client.model}")

    use_chat = probe_cfg.llm.runtime.llm_use_chat_api
    primary_url = client.base_url
    ok = _probe_llm_endpoint(
        primary_url,
        client.model,
        label="Configured/default endpoint",
        use_chat_api=use_chat,
    )

    if primary_url == "http://127.0.0.1:8080/v1/completions":
        _probe_llm_endpoint(
            "http://127.0.0.1:11434/v1/completions",
            client.model,
            label="Ollama (alternate local endpoint)",
            fail_is_warn=True,
            use_chat_api=use_chat,
        )
    return ok


def run_doctor_checks(args: argparse.Namespace) -> int:
    """Run preflight diagnostics for local environment and dependencies."""
    con = Console()

    con.print()
    con.print("[bold]Folionym Doctor[/bold]")
    con.print("[dim]Local-first document naming: checking data, dependencies, TUI, and LLM connectivity...[/dim]")
    con.print()

    ok = _run_doctor_data_checks(con)
    _run_doctor_dependency_checks(con)
    ok = _run_doctor_llm_checks(args, con) and ok

    con.print()
    if ok:
        con.print("[bold green]All checks passed.[/bold green]")
    else:
        con.print("[bold red]Some checks failed.[/bold red] See above for details.")
    con.print()
    return 0 if ok else 1


def _resolve_dirs(args: argparse.Namespace) -> tuple[list[str], str | None]:
    """Resolve directory list and optional single-file path from args. Raises SystemExit on error."""
    return resolve_dirs(args, is_interactive=_is_interactive, console=_console, logger=logger)


def _resolved_config_prompts(args: argparse.Namespace, file_defaults: dict[str, object]) -> dict[str, str]:
    """Resolve interactive/default naming options after config-file defaults are loaded."""
    language = _resolve_option(
        args,
        request=OptionResolutionRequest(
            attr="language",
            file_defaults=file_defaults,
            file_key="language",
            default="de",
            prompt=OptionPromptSpec(
                choice_prompt="Language (de/en, default: de): ",
                choices=["de", "en"],
                choice_normalize=str.lower,
            ),
        ),
    )
    desired_case = _resolve_option(
        args,
        request=OptionResolutionRequest(
            attr="desired_case",
            file_defaults=file_defaults,
            file_key="desired_case",
            default="kebabCase",
            prompt=OptionPromptSpec(
                choice_prompt="Desired case format (camelCase, kebabCase, snakeCase, default: kebabCase): ",
                choices=list(VALID_CASE_CHOICES),
                choice_normalize=str.lower,
            ),
        ),
    )
    project = _resolve_option(
        args,
        request=OptionResolutionRequest(
            attr="project",
            file_defaults=file_defaults,
            file_key="project",
            default="",
            prompt=OptionPromptSpec(free_prompt="Project name (optional): "),
        ),
    )
    version = _resolve_option(
        args,
        request=OptionResolutionRequest(
            attr="version",
            file_defaults=file_defaults,
            file_key="version",
            default="",
            prompt=OptionPromptSpec(free_prompt="Version (optional): "),
        ),
    )
    return {"language": language, "desired_case": desired_case, "project": project, "version": version}


def _raw_args_with_file_defaults(
    args: argparse.Namespace,
    file_defaults: dict[str, object],
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Merge config-file defaults into omitted CLI arguments without overriding explicit flags."""
    raw = vars(args).copy()
    for dest, file_value in file_defaults.items():
        if dest not in raw:
            continue
        if raw[dest] == parser.get_default(dest):
            raw[dest] = file_value
    return raw


def _apply_runtime_overrides(raw: dict[str, object], args: argparse.Namespace) -> None:
    """Inject CLI-only runtime values before the shared resolver validates the merged configuration."""
    raw["manual_mode"] = bool(getattr(args, "manual_file", None))
    if raw["manual_mode"]:
        raw["interactive"] = True
    if getattr(args, "override_category_file", None):
        raw["override_category_map"] = _load_override_category_map(args.override_category_file)


def _build_config_or_exit(raw: dict[str, object], file_defaults: dict[str, object]) -> RenamerConfig:
    """Build config or exit so callers share one validated construction path."""
    try:
        return build_config(raw, file_defaults=file_defaults)
    except ValueError as exc:
        _console.print(f"[red]Configuration error:[/red] {exc}")
        _console.print("[dim]Run --validate-config to check settings without processing files.[/dim]")
        raise SystemExit(1) from exc


def _build_config_from_args(
    args: argparse.Namespace,
    file_defaults: dict[str, object],
    *,
    parser: argparse.ArgumentParser,
) -> RenamerConfig:
    """Build RenamerConfig from args and file defaults (with interactive prompts where applicable)."""
    # Treat parser defaults as "unset" so file defaults win when the CLI omitted an option.
    raw = _raw_args_with_file_defaults(args, file_defaults, parser)
    raw.update(_resolved_config_prompts(args, file_defaults))
    _apply_runtime_overrides(raw, args)
    return _build_config_or_exit(raw, file_defaults)


def _files_override_for(directory: str, single_file: str | None) -> list[Path] | None:
    """Return the explicit file selection for a directory when single-file mode is active."""
    if not single_file:
        return None
    # Resolve the parent only. Keeping the final entry lexical lets discovery
    # reject a file swapped to a symlink after the initial CLI validation.
    requested_path = Path(os.path.abspath(Path(single_file).expanduser()))
    single_path = requested_path.parent.resolve() / requested_path.name
    if Path(directory).resolve() != single_path.parent:
        return None
    return [single_path]


def _run_requested_operation(
    dirs: list[str],
    config: RenamerConfig,
    args: argparse.Namespace,
    *,
    single_file: str | None,
) -> None:
    """Dispatch watch mode or one batch rename operation per resolved directory."""
    if getattr(args, "watch", False):
        if len(dirs) > 1:
            raise SystemExit("Error: --watch supports only one directory. Use a single --dir.")
        run_watch_loop(
            dirs[0],
            config=config,
            interval_seconds=float(getattr(args, "watch_interval", 60) or 60),
        )
        return

    for directory in dirs:
        if not directory:
            continue
        rename_pdfs_in_directory(
            directory,
            config=config,
            files_override=_files_override_for(directory, single_file),
        )


def _raise_cli_error(exc: Exception) -> NoReturn:
    """Render a CLI error consistently, then terminate with the supplied exit status."""
    if isinstance(exc, FileNotFoundError | NotADirectoryError):
        _console.print(f"[red]Path error:[/red] {exc}")
        _console.print("[dim]Check that the directory exists and is accessible.[/dim]")
        raise SystemExit(1) from exc
    if isinstance(exc, json.JSONDecodeError):
        _console.print(f"[red]Invalid JSON in data file[/red] (line {exc.lineno}): {exc.msg}")
        _console.print("[dim]Check heuristic_scores.json / meta_stopwords.json in the data directory.[/dim]")
        raise SystemExit(1) from exc
    if isinstance(exc, requests.Timeout):
        _console.print(f"[red]LLM timeout:[/red] {exc}")
        _console.print("[dim]Try increasing --llm-timeout or check server status. Use --doctor to verify.[/dim]")
        raise SystemExit(1) from exc
    if isinstance(exc, requests.RequestException):
        _console.print(f"[red]LLM/network error:[/red] {exc}")
        _console.print(
            "[dim]Check server status, endpoint URL, or try increasing --llm-timeout. Use --doctor to verify.[/dim]"
        )
        raise SystemExit(1) from exc
    if isinstance(exc, OSError):
        _console.print(f"[red]I/O error:[/red] {exc}")
        raise SystemExit(1) from exc
    if isinstance(exc, ValueError):
        _console.print(f"[red]Configuration error:[/red] {exc}")
        _console.print("[dim]Run --validate-config to check config precedence and merged values.[/dim]")
        raise SystemExit(1) from exc

    logger.debug("Unhandled exception", exc_info=True)
    _console.print(f"[red]Unexpected error:[/red] {exc}")
    _console.print("[dim]Run with --verbose for full traceback.[/dim]")
    raise SystemExit(1) from exc


def _run_renamer_or_watch(
    dirs: list[str],
    config: RenamerConfig,
    args: argparse.Namespace,
    *,
    single_file: str | None = None,
) -> None:
    """Run watch loop or rename each directory. Raises SystemExit on path/data errors."""
    try:
        _run_requested_operation(dirs, config, args, single_file=single_file)
    except _RECOVERABLE_CLI_EXCEPTIONS as exc:
        _raise_cli_error(exc)


def main(argv: list[str] | None = None) -> None:
    """Parse command-line input and hand off to the command lifecycle."""
    parser = build_parser()
    args = parser.parse_args(argv)

    log_file, log_level = _resolve_log_config(args)
    setup_logging(log_file=log_file, level=log_level)

    if getattr(args, "doctor", False):
        raise SystemExit(run_doctor_checks(args))

    try:
        file_defaults = _load_config_file(args.config, raise_on_error=True) if getattr(args, "config", None) else {}
    except ConfigLoadError as exc:
        raise SystemExit(1) from exc
    config = _build_config_from_args(args, file_defaults, parser=parser)
    if getattr(args, "validate_config", False):
        _console.print("[green]Configuration valid.[/green]")
        raise SystemExit(0)
    dirs, single_file = _resolve_dirs(args)
    _run_renamer_or_watch(dirs, config, args, single_file=single_file)


if __name__ == "__main__":
    main()
