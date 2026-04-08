from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import re
import shlex
import signal
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from .config import RenamerConfig as RenamerConfig
from .config import build_config_from_flat_dict as build_config_from_flat_dict
from .filename import generate_filename
from .loaders import _heuristic_scorer_cached, _stopwords_cached  # noqa: F401 re-export for tests
from .pdf_extract import get_pdf_metadata, pdf_first_page_to_image_base64, pdf_to_text, pdf_to_text_with_ocr
from .rename_ops import (
    MAX_RENAME_RETRIES,
    apply_single_rename,
    sanitize_filename_base,
)
from .renamer_extract import extract_pdf_content_with as _extract_pdf_content_with
from .renamer_files import collect_pdf_files as _collect_pdf_files
from .rules import (
    ProcessingRules,
    force_category_for_basename,
    load_processing_rules,
)

logger = logging.getLogger(__name__)


class _NullProgressReporter:
    def __enter__(self) -> _NullProgressReporter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def update(self, current: int, total: int, file_path: Path) -> None:
        return None


class _RichProgressReporter:
    def __init__(self, total: int, *, quiet: bool) -> None:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        columns: list[object] = []
        if quiet:
            columns.extend(
                [
                    TextColumn("{task.percentage:>3.0f}%"),
                    TextColumn("{task.completed}/{task.total}"),
                ]
            )
        else:
            columns.extend(
                [
                    TextColumn("{task.completed}/{task.total}"),
                    BarColumn(bar_width=None),
                    TextColumn("{task.percentage:>3.0f}%"),
                ]
            )
        columns.extend([TextColumn("{task.fields[filename]}", overflow="ellipsis"), TimeElapsedColumn()])
        self._progress = Progress(*columns, console=Console(stderr=True), transient=True)
        self._task_id = self._progress.add_task("Processing PDFs", total=total, filename="")

    def __enter__(self) -> _RichProgressReporter:
        self._progress.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._progress.stop()

    def update(self, current: int, total: int, file_path: Path) -> None:
        self._progress.update(self._task_id, total=total, completed=current, filename=file_path.name)


def _create_progress_reporter(total: int, config: RenamerConfig) -> _NullProgressReporter | _RichProgressReporter:
    """Create an opt-in progress reporter without affecting default CLI output."""
    if not (config.progress or config.quiet_progress):
        return _NullProgressReporter()
    try:
        return _RichProgressReporter(total, quiet=bool(config.quiet_progress))
    except ImportError:
        logger.warning("Rich progress unavailable; continuing without progress UI.")
        return _NullProgressReporter()


def _stop_requested(config: RenamerConfig) -> bool:
    stop_event = config.stop_event
    return bool(stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set())


def _extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    """Compatibility shim for the extraction pipeline.

    The strategy logic lives in renamer_extract.py; this wrapper keeps the
    renamer-level patch points stable for tests and local overrides.
    """
    return _extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=pdf_first_page_to_image_base64,
        pdf_to_text_fn=pdf_to_text,
        pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr,
    )


def _write_pdf_title_metadata(pdf_path: Path, title: str) -> None:
    """Write /Title metadata to PDF (PyMuPDF). No-op if fitz unavailable or on error.

    Writes to a temporary file first, then atomically replaces the original to
    prevent corruption under concurrent access.
    """
    try:
        import tempfile

        import fitz

        doc = fitz.open(pdf_path)
        try:
            doc.set_metadata({"title": title or pdf_path.stem})
            # Write to a temporary file in the same directory, then replace atomically.
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=pdf_path.parent)
            os.close(fd)
            tmp = Path(tmp_path)
            try:
                doc.save(str(tmp), incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
            except TypeError:
                # P2: PDF_ENCRYPT_KEEP may not exist in old PyMuPDF
                doc.save(str(tmp), incremental=False)
            except Exception:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
        finally:
            doc.close()
        # P1: Verify tmp file is valid (non-zero size) before replacing original
        tmp_size = tmp.stat().st_size
        if tmp_size == 0:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise OSError(f"Temporary PDF file is empty (possible disk full): {tmp}")
        # Atomic replace (os.replace is atomic on POSIX, best-effort on Windows).
        os.replace(tmp, pdf_path)
    except Exception as exc:
        logger.warning("Could not write PDF metadata for %s: %s", pdf_path, exc)


_CSV_FORMULA_TRIGGERS = frozenset("=+-@|")
_CSV_CONTROL_CHARS_RE = re.compile(r"[\t\r\n]")
# C0 control chars (incl. NUL) must be stripped from env-var values passed to hook commands.
_HOOK_ENV_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize_csv_cell(value: object) -> object:
    """Prevent CSV formula injection in spreadsheet apps (OWASP).

    Handles: leading formula triggers after whitespace stripping,
    inline control chars (tab/CR/LF) that enable cell injection.
    """
    if not isinstance(value, str) or not value:
        return value
    # Replace control chars that enable cell injection within the value.
    cleaned = _CSV_CONTROL_CHARS_RE.sub(" ", value)
    # Check first non-whitespace character for formula triggers.
    stripped = cleaned.lstrip()
    if stripped and stripped[0] in _CSV_FORMULA_TRIGGERS:
        return "'" + cleaned
    return cleaned


def _write_json_or_csv(
    path: Path,
    rows: list[Any],
    csv_fieldnames: list[str] | None,
) -> None:
    """Write rows to path as CSV (if csv_fieldnames and .csv suffix) or JSON. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv" and csv_fieldnames:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows({k: _sanitize_csv_cell(v) for k, v in row.items()} for row in rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_summary_json(
    summary_path: str | Path | None,
    *,
    directory: Path,
    processed: int,
    renamed: int,
    skipped: int,
    failed: int,
    dry_run: bool,
    failures: list[dict[str, str]],
) -> None:
    """Write run summary JSON as a single object. Best-effort: logs warning on write failure."""
    if not summary_path:
        return
    summary = {
        "directory": str(directory),
        "processed": processed,
        "renamed": renamed,
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
        "failures": failures,
    }
    target = Path(summary_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write summary JSON %s: %s", target, exc)


def _run_post_rename_hook(hook_cmd: str, old_path: Path, new_path: Path, meta: dict[str, object]) -> None:
    """Run post-rename hook command or HTTP endpoint. On failure log and continue."""
    # Sanitize path strings: strip C0 control characters (\x00-\x1f) and DEL (\x7f)
    # from values passed via env vars to hook commands. NUL (\x00) is especially
    # important as it truncates C strings, but other control chars can also cause issues.
    _old = _HOOK_ENV_CONTROL_RE.sub("", str(old_path))
    _new = _HOOK_ENV_CONTROL_RE.sub("", str(new_path))
    env = {**os.environ, "AI_PDF_RENAMER_OLD_PATH": _old, "AI_PDF_RENAMER_NEW_PATH": _new}
    try:
        meta_json = json.dumps(meta, default=str)
    except (TypeError, ValueError):
        meta_json = "{}"
    env["AI_PDF_RENAMER_META"] = meta_json
    cmd = (hook_cmd or "").strip()
    if not cmd:
        return
    try:
        cmd_lower = cmd.lower()
        if cmd_lower.startswith("http://") or cmd_lower.startswith("https://"):
            if cmd_lower.startswith("http://"):
                _host = cmd_lower[len("http://") :].split("/")[0].split(":")[0]
                if _host not in {"127.0.0.1", "::1", "localhost"}:
                    logger.warning(
                        "Post-rename hook uses plain HTTP with a non-loopback host (%s). "
                        "Document metadata will be transmitted unencrypted. Use HTTPS for remote endpoints.",
                        _host,
                    )
            payload = {
                "old_path": str(old_path),
                "new_path": str(new_path),
                "meta": meta,
            }
            with requests.Session() as session:
                session.trust_env = False
                resp = session.post(cmd, json=payload, timeout=10)
                resp.raise_for_status()
            return

        shell_meta = {"|", "&", ";", "<", ">", "$", "`", "(", ")", "\n"}
        needs_shell = any(ch in cmd for ch in shell_meta)

        if needs_shell:
            if os.name == "nt":
                shell_exe = os.environ.get("COMSPEC", "cmd.exe")
                args = [shell_exe, "/c", cmd]
            else:
                shell_exe = os.environ.get("SHELL", "/bin/sh")
                # P2: Use -c instead of -lc to avoid login shell behavior
                args = [shell_exe, "-c", cmd]
        else:
            args = shlex.split(cmd, posix=(os.name != "nt"))
            if not args:
                return

        # Command comes from explicit local operator configuration (--post-rename-hook / env var).
        subprocess.run(args, shell=False, env=env, timeout=120, check=False)  # nosec B603
    except subprocess.TimeoutExpired:
        logger.warning("Post-rename hook timed out (120s): %s", cmd[:80])
    except requests.RequestException as exc:
        logger.warning("Post-rename hook HTTP call failed: %s", exc)
    except Exception as e:
        logger.warning("Post-rename hook failed: %s", e)
        logger.debug("Post-rename hook traceback", exc_info=True)


def _apply_post_rename_actions(
    config: RenamerConfig,
    file_path: Path,
    target: Path,
    current_base: str,
    meta: dict[str, object],
    export_rows: list[dict[str, object]],
) -> None:
    """Write rename log, PDF metadata, and export row after a successful rename. Mutates export_rows."""
    if config.rename_log_path:
        file_path_str = str(file_path)
        target_str = str(target)
        if "\t" in file_path_str or "\n" in file_path_str or "\r" in file_path_str:
            logger.warning(
                "Cannot write rename log entry for %s: original path contains tab or newline "
                "characters (unsupported by the tab-delimited log format). Undo will not be "
                "available for this file.",
                file_path,
            )
        elif "\t" in target_str or "\n" in target_str or "\r" in target_str:
            logger.warning(
                "Cannot write rename log entry for %s: target path contains tab or newline "
                "characters (unsupported by the tab-delimited log format). Undo will not be "
                "available for this file.",
                target,
            )
        else:
            Path(config.rename_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config.rename_log_path, "a", encoding="utf-8") as f:
                f.write(f"{file_path_str}\t{target}\n")
    if config.write_pdf_metadata:
        _write_pdf_title_metadata(target, current_base)
    if config.export_metadata_path:
        export_rows.append(
            {
                "path": str(file_path),
                "new_name": target.name,
                "category": meta.get("category", ""),
                "summary": meta.get("summary", ""),
                "keywords": meta.get("keywords", ""),
                "category_source": meta.get("category_source", ""),
                "llm_failed": meta.get("llm_failed", False),
                "used_vision_fallback": meta.get("used_vision_fallback", False),
                "invoice_id": meta.get("invoice_id", ""),
                "amount": meta.get("amount", ""),
                "company": meta.get("company", ""),
            }
        )
    hook_cmd = (config.post_rename_hook or "").strip() or (
        os.environ.get("AI_PDF_RENAMER_POST_RENAME_HOOK") or ""
    ).strip()
    if hook_cmd:
        _run_post_rename_hook(hook_cmd, file_path, target, meta)


def _process_content_to_result(
    file_path: Path,
    content: str,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    used_vision: bool = False,
) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
    """
    Generate filename from already-extracted content. Returns (path, new_base, meta, error).
    Caller must ensure content is non-empty if expecting a non-skip result.
    """
    try:
        override_cat = (config.override_category_map or {}).get(file_path.name) or force_category_for_basename(
            rules, file_path.name
        )
        pdf_meta = get_pdf_metadata(file_path) if config.use_pdf_metadata_for_date else None
        filename_str, meta = generate_filename(
            content,
            config=config,
            override_category=override_cat,
            pdf_metadata=pdf_meta,
            rules=rules,
            source_path=file_path,
        )
        new_base = sanitize_filename_base(filename_str)
        meta = meta or {}
        meta["used_vision_fallback"] = used_vision
        return (file_path, new_base, meta, None)
    except Exception as exc:
        return (file_path, None, None, exc)


def _interactive_rename_prompt(
    file_path: Path,
    target: Path,
    default_base: str,
    edit_default_base: str | None = None,
) -> tuple[str, str, Path]:
    """Prompt for y/n/e=edit. Returns (reply, base, target); reply in {'y','n'}; base/target may change on edit.
    When edit_default_base is set, the edit prompt shows it as default; Enter accepts it."""
    reply: str = "y"
    base = default_base
    current_target = target
    while True:
        try:
            reply = (
                input(f"Rename '{file_path.name}' to '{current_target.name}'? (y/n/e=edit, default y): ")
                .strip()
                .lower()
                or "y"
            )
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply == "n":
            return ("n", base, current_target)
        if reply == "e":
            try:
                prompt = "New filename (without path)"
                if edit_default_base:
                    prompt += f" [default: {edit_default_base}]"
                prompt += ": "
                custom = input(prompt).strip()
                if not custom and edit_default_base:
                    custom = edit_default_base
                if custom:
                    custom_base = sanitize_filename_base(
                        custom.removesuffix(file_path.suffix)
                        if custom.lower().endswith(file_path.suffix.lower())
                        else custom
                    )
                    base = custom_base
                    current_target = file_path.with_name(custom_base + file_path.suffix)
                    return ("y", base, current_target)
            except (EOFError, KeyboardInterrupt):
                pass
            continue
        return ("y", base, current_target)


def _process_one_file(
    file_path: Path,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
    """
    Extract text and generate filename for one file. Returns (path, new_base, meta, error).
    If error is not None or new_base is None, the file should be counted as failed/skipped.
    Empty or unextractable PDF: returns (path, None, None, None); caller logs and skips the file.
    """
    if _stop_requested(config):
        return (file_path, None, None, None)
    try:
        content, used_vision = _extract_pdf_content(file_path, config)
    except Exception as exc:
        return (file_path, None, None, exc)
    if not content.strip():
        return (file_path, None, None, None)  # skipped empty
    try:
        return _process_content_to_result(file_path, content, config, rules=rules, used_vision=used_vision)
    except Exception as exc:
        return (file_path, None, None, exc)


def suggest_rename_for_file(
    file_path: Path,
    config: RenamerConfig,
) -> tuple[str | None, dict[str, object] | None, BaseException | None]:
    """
    Run the pipeline for one file and return the suggested new basename and metadata.
    Does not rename or prompt. Returns (new_base, meta, error).
    new_base is None if content is empty or an error occurred.
    """
    rules = load_processing_rules(config.rules_file)
    try:
        content, used_vision = _extract_pdf_content(file_path, config)
    except Exception as exc:
        return (None, None, exc)
    if not content.strip():
        return (None, None, None)
    try:
        _path, new_base, meta, process_err = _process_content_to_result(
            file_path, content, config, rules=rules, used_vision=used_vision
        )
        if process_err is not None:
            return (None, None, process_err)
        return (new_base, meta, None)
    except Exception as exc:
        return (None, None, exc)


def _produce_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce (file_path, new_base, meta, exc) per file; parallel or single-worker with prefetch."""
    workers = max(1, config.workers or 1)
    if config.interactive:
        workers = 1
    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures: list[tuple[Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]], Path]] = []
        results: list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]] = []
        stop_early = False
        try:
            for p in files:
                if _stop_requested(config):
                    logger.info("Stop requested. Ending processing early.")
                    stop_early = True
                    break
                futures.append((executor.submit(_process_one_file, p, config, rules), p))

            for future, p in futures:
                if _stop_requested(config):
                    logger.info("Stop requested. Ending processing early.")
                    stop_early = True
                    break
                try:
                    result = future.result()
                except BaseException as exc:
                    # P2: Catch BaseException to handle KeyboardInterrupt properly
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        stop_early = True
                        break
                    result = (p, None, None, exc)
                results.append(result)
                if progress_callback is not None:
                    progress_callback(len(results), len(files), p)
        finally:
            if stop_early:
                for pending, _ in futures:
                    pending.cancel()
                # Wait for already-running futures so they release file handles before returning.
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=False)
        return results
    results = []
    prefetched: Future[tuple[str, bool]] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, file_path in enumerate(files):
            if _stop_requested(config):
                logger.info("Stop requested. Ending processing early.")
                break
            try:
                content, used_vision = (
                    prefetched.result() if prefetched is not None else _extract_pdf_content(file_path, config)
                )
            except Exception as exc:
                results.append((file_path, None, None, exc))
                if progress_callback is not None:
                    progress_callback(len(results), len(files), file_path)
                prefetched = None
                continue
            prefetched = executor.submit(_extract_pdf_content, files[i + 1], config) if i + 1 < len(files) else None
            if not content.strip():
                results.append((file_path, None, None, None))
                if progress_callback is not None:
                    progress_callback(len(results), len(files), file_path)
                continue
            try:
                result = _process_content_to_result(file_path, content, config, rules=rules, used_vision=used_vision)
            except Exception as exc:
                result = (file_path, None, None, exc)
            results.append(result)
            if progress_callback is not None:
                progress_callback(len(results), len(files), file_path)
    return results


def _write_rename_outputs(
    config: RenamerConfig,
    directory: Path,
    *,
    export_rows: list[dict[str, object]],
    plan_entries: list[dict[str, str]],
    processed_count: int,
    renamed_count: int,
    skipped_count: int,
    failed_count: int,
    failure_details: list[dict[str, str]],
) -> None:
    """Write export metadata, plan file, and summary JSON after rename loop completes."""
    if config.export_metadata_path and export_rows:
        _write_json_or_csv(
            Path(config.export_metadata_path),
            export_rows,
            [
                "path",
                "new_name",
                "category",
                "summary",
                "keywords",
                "category_source",
                "llm_failed",
                "used_vision_fallback",
                "invoice_id",
                "amount",
                "company",
            ],
        )

    if config.plan_file_path and plan_entries:
        plan_path = Path(config.plan_file_path)
        _write_json_or_csv(plan_path, plan_entries, ["old", "new"])
        logger.info("Wrote rename plan (%s entries) to %s", len(plan_entries), plan_path)

    _write_summary_json(
        config.summary_json_path,
        directory=directory,
        processed=processed_count,
        renamed=renamed_count,
        skipped=skipped_count,
        failed=failed_count,
        dry_run=bool(config.dry_run),
        failures=failure_details,
    )

    logger.info(
        "Summary: %s file(s) processed, %s renamed, %s skipped, %s failed",
        processed_count,
        renamed_count,
        skipped_count,
        failed_count,
    )
    try:
        from rich.console import Console

        _con = Console(stderr=True)
        _con.print()
        parts = [f"[bold]{processed_count}[/bold] processed"]
        if renamed_count:
            parts.append(f"[green]{renamed_count} renamed[/green]")
        else:
            parts.append(f"{renamed_count} renamed")
        if skipped_count:
            parts.append(f"[yellow]{skipped_count} skipped[/yellow]")
        else:
            parts.append(f"{skipped_count} skipped")
        if failed_count:
            parts.append(f"[red]{failed_count} failed[/red]")
        else:
            parts.append(f"{failed_count} failed")
        _con.print("[bold]Summary:[/bold] " + ", ".join(parts))
    except ImportError:
        print(
            f"Summary: {processed_count} processed, {renamed_count} renamed, "
            f"{skipped_count} skipped, {failed_count} failed.",
            file=sys.stderr,
        )


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: RenamerConfig,
    files_override: list[Path] | None = None,
) -> set[Path]:
    """Rename all PDFs in a directory using the configured pipeline (extract, LLM/heuristic, rename).

    Write export metadata, plan file, and summary JSON after processing. Use files_override to
    process specific files instead of scanning the directory.
    """
    dir_str = str(directory).strip()
    if not dir_str:
        raise ValueError("Directory path must be non-empty. Use --dir or provide when prompted.")
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if files_override is None and not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    path = path.resolve()

    rules = load_processing_rules(config.rules_file)
    files = _collect_pdf_files(
        path,
        recursive=config.recursive,
        max_depth=config.max_depth,
        include_patterns=config.include_patterns,
        exclude_patterns=config.exclude_patterns,
        skip_if_already_named=config.skip_if_already_named,
        files_override=files_override,
        rules=rules,
    )
    if not files:
        logger.info("No matching PDF files found in %s", path)
        _write_summary_json(
            config.summary_json_path,
            directory=path,
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=bool(config.dry_run),
            failures=[],
        )
        return set()

    def _mtime_key(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=_mtime_key, reverse=True)

    if not config.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")

    renamed_count = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    failure_details: list[dict[str, str]] = []
    export_rows: list[dict[str, object]] = []
    plan_entries: list[dict[str, str]] = []
    renamed_targets: set[Path] = set()

    with _create_progress_reporter(len(files), config) as progress_reporter:
        if config.progress or config.quiet_progress:
            results = _produce_rename_results(files, config, rules=rules, progress_callback=progress_reporter.update)
        else:
            results = _produce_rename_results(files, config, rules=rules)

    for i, (file_path, new_base, meta, exc) in enumerate(results):
        if _stop_requested(config):
            logger.info("Stop requested. Ending rename/apply phase.")
            break
        processed_count += 1
        logger.info("Processing %s/%s: %s", i + 1, len(files), file_path)
        if exc is not None:
            # Data-file/config errors (e.g. invalid JSON) should propagate so CLI can exit with clear message.
            # P2: Use exception type checking where possible, with string check as fallback
            if isinstance(exc, (json.JSONDecodeError,)):
                raise exc
            if isinstance(exc, ValueError) and "Invalid JSON in data file" in str(exc):
                raise exc
            if isinstance(exc, ValueError) and "No text extracted from" in str(exc):
                logger.warning("Skipping %s: %s", file_path.name, exc)
                skipped_count += 1
                continue
            logger.exception("Failed to process %s: %s", file_path, exc)
            failed_count += 1
            failure_details.append({"file": str(file_path), "error": str(exc)})
            continue
        if new_base is None:
            # This case usually means truly empty content without causing an error.
            logger.info("PDF content is empty. Skipping %s.", file_path.name)
            skipped_count += 1
            continue
        meta = meta or {}
        base = new_base
        target = file_path.with_name(new_base + file_path.suffix)
        if config.interactive:
            if config.manual_mode:
                print(f"Suggested: {new_base}{file_path.suffix}")
                for k, v in (meta or {}).items():
                    if k in ("category", "summary", "keywords", "category_source") and v:
                        print(f"  {k}: {v}")
            reply, base, target = _interactive_rename_prompt(
                file_path,
                target,
                new_base,
                edit_default_base=new_base if config.manual_mode else None,
            )
            if reply == "n":
                skipped_count += 1
                continue
        try:
            # Default-parameter binding captures loop variables by value (not reference).
            # This is the standard Python pattern for closures inside loops (B023).
            def _on_rename_success(
                _fp: Path,
                _target: Path,
                _current_base: str,
                _meta: dict[str, object] = meta or {},
                _rows: list[dict[str, object]] = export_rows,
            ) -> None:
                _apply_post_rename_actions(config, _fp, _target, _current_base, _meta, _rows)

            success, target = apply_single_rename(
                file_path,
                base,
                plan_file_path=config.plan_file_path,
                plan_entries=plan_entries,
                dry_run=config.dry_run,
                backup_dir=config.backup_dir,
                on_success=_on_rename_success,
                max_filename_chars=config.max_filename_chars,
            )
            if not success:
                logger.error(
                    "Skipping %s: could not rename after %s attempts",
                    file_path.name,
                    MAX_RENAME_RETRIES,
                )
                failed_count += 1
                failure_details.append(
                    {
                        "file": str(file_path),
                        "error": f"Could not rename after {MAX_RENAME_RETRIES} attempts.",
                    }
                )
            else:
                if config.dry_run:
                    logger.info(
                        "Dry-run: would rename '%s' to '%s'",
                        file_path.name,
                        target.name,
                    )
                else:
                    logger.info("Renamed '%s' to '%s'", file_path.name, target.name)
                    if target != file_path:
                        renamed_targets.add(target.resolve())
                renamed_count += 1
        except Exception as e:
            logger.exception("Failed to process %s: %s", file_path, e)
            failed_count += 1
            failure_details.append({"file": str(file_path), "error": str(e)})

    _write_rename_outputs(
        config,
        path,
        export_rows=export_rows,
        plan_entries=plan_entries,
        processed_count=processed_count,
        renamed_count=renamed_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        failure_details=failure_details,
    )
    return renamed_targets


def run_watch_loop(
    directory: str | Path,
    *,
    config: RenamerConfig,
    interval_seconds: float = 60.0,
) -> None:
    """Run rename in a loop, scanning the directory every interval_seconds. Processes new/changed PDFs."""
    path = Path(directory).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    rules = load_processing_rules(config.rules_file)

    stop_requested = False

    def handle_stop(sig: int, frame: object) -> None:
        nonlocal stop_requested
        logger.info("Watch mode: received signal %s, stopping...", sig)
        stop_requested = True

    # P2: Only set signal handlers from the main thread (signal.signal raises ValueError otherwise)
    import threading as _threading

    _is_main_thread = _threading.current_thread() is _threading.main_thread()
    original_sigterm = None
    original_sigint = None
    if _is_main_thread:
        original_sigterm = signal.signal(signal.SIGTERM, handle_stop)
        original_sigint = signal.signal(signal.SIGINT, handle_stop)

    seen: dict[Path, float] = {}
    # P1: Track renamed target paths to prevent reprocessing
    renamed_targets: set[Path] = set()
    logger.info("Watch mode: scanning %s every %.1fs (Ctrl+C or SIGTERM to stop)", path, interval_seconds)
    try:
        while not stop_requested:
            try:
                # Cleanup 'seen' map: remove files that no longer exist
                to_remove = [p for p in seen if not p.exists()]
                for p in to_remove:
                    del seen[p]
                # Also clean up renamed_targets for files that no longer exist
                renamed_targets = {p for p in renamed_targets if p.exists()}

                files = _collect_pdf_files(
                    path,
                    recursive=config.recursive,
                    max_depth=config.max_depth,
                    include_patterns=config.include_patterns,
                    exclude_patterns=config.exclude_patterns,
                    skip_if_already_named=config.skip_if_already_named,
                    files_override=None,
                    rules=rules,
                )
                to_process: list[Path] = []
                for p in files:
                    # P1: Skip files that are known rename targets
                    if p in renamed_targets:
                        continue
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        continue
                    if p not in seen or seen[p] != mtime:
                        to_process.append(p)
                        seen[p] = mtime
                if to_process:
                    to_process.sort(key=lambda p: seen.get(p, 0), reverse=True)
                    for single in to_process:
                        if stop_requested:
                            break
                        actual_targets = rename_pdfs_in_directory(
                            path,
                            config=config,
                            files_override=[single],
                        )
                        if actual_targets:
                            renamed_targets.update(actual_targets)
                            for renamed_target in actual_targets:
                                with contextlib.suppress(OSError):
                                    seen[renamed_target] = renamed_target.stat().st_mtime
                if not stop_requested:
                    time.sleep(interval_seconds)
            except Exception as exc:
                if not stop_requested:
                    logger.exception("Watch iteration failed: %s", exc)
                    time.sleep(interval_seconds)
    finally:
        # Restore original handlers (P2: only if we set them)
        if _is_main_thread and original_sigterm is not None:
            signal.signal(signal.SIGTERM, original_sigterm)
        if _is_main_thread and original_sigint is not None:
            signal.signal(signal.SIGINT, original_sigint)
        logger.info("Watch stopped")
