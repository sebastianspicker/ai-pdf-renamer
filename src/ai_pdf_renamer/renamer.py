from __future__ import annotations

import csv
import fnmatch
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .config import RenamerConfig, build_config_from_flat_dict  # noqa: F401 re-export
from .filename import _llm_client_from_config, generate_filename
from .llm import complete_vision
from .loaders import _heuristic_scorer_cached, _stopwords_cached  # noqa: F401 re-export for tests
from .pdf_extract import (
    DEFAULT_MAX_CONTENT_TOKENS,
    get_pdf_metadata,
    pdf_first_page_to_image_base64,
    pdf_to_text,
    pdf_to_text_with_ocr,
)
from .rename_ops import (
    MAX_RENAME_RETRIES,
    apply_single_rename,
    sanitize_filename_base,
    sanitize_filename_from_llm,
)
from .llm_prompts import build_vision_filename_prompt
from .rules import (
    ProcessingRules,
    force_category_for_basename,
    load_processing_rules,
    should_skip_file_by_rules,
)


def _matches_patterns(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    """True if basename matches include (if set) and does not match any exclude."""
    if include is not None and include:
        if not any(fnmatch.fnmatch(name, p) for p in include):
            return False
    if exclude is not None and exclude:
        if any(fnmatch.fnmatch(name, p) for p in exclude):
            return False
    return True


def _collect_pdf_files(
    directory: Path,
    *,
    recursive: bool = False,
    max_depth: int = 0,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    skip_if_already_named: bool = False,
    files_override: list[Path] | None = None,
    rules: ProcessingRules | None = None,
) -> list[Path]:
    """Collect PDFs from directory (or files_override). Rules skip_files_by_pattern filters out matches."""
    if files_override is not None:
        candidates = [p for p in files_override if p.is_file() and p.suffix.lower() == ".pdf"]
    elif recursive:
        candidates = []
        for p in directory.rglob("*.pdf"):
            if not p.is_file() or p.name.startswith("."):
                continue
            if max_depth > 0:
                try:
                    rel = p.relative_to(directory)
                    if len(rel.parts) > max_depth:
                        continue
                except ValueError:
                    continue
            candidates.append(p)
    else:
        candidates = [
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
        ]
    out = [p for p in candidates if _matches_patterns(p.name, include_patterns, exclude_patterns)]
    if rules is not None:
        out = [p for p in out if not should_skip_file_by_rules(rules, p.name)]
    if skip_if_already_named:
        already_named = re.compile(r"^\d{8}-.+\.[pP][dD][fF]$")
        out = [p for p in out if not already_named.match(p.name)]
    return out


logger = logging.getLogger(__name__)


def _write_pdf_title_metadata(pdf_path: Path, title: str) -> None:
    """Write /Title metadata to PDF (PyMuPDF). No-op if fitz unavailable or on error."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            doc.set_metadata({"title": title or pdf_path.stem})
            try:
                doc.save_incremental()
            except Exception as inc_exc:
                logger.debug("Incremental save failed for %s (%s); falling back to full save.", pdf_path.name, inc_exc)
                doc.save(pdf_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("Could not write PDF metadata for %s: %s", pdf_path, exc)


def _write_json_or_csv(
    path: Path,
    rows: list[dict],
    csv_fieldnames: list[str] | None,
) -> None:
    """Write rows to path as CSV (if csv_fieldnames and .csv suffix) or JSON. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv" and csv_fieldnames:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def _run_post_rename_hook(hook_cmd: str, old_path: Path, new_path: Path, meta: dict) -> None:
    """Run post-rename hook in a subprocess. On failure log and continue; do not fail the run."""
    env = {**os.environ, "AI_PDF_RENAMER_OLD_PATH": str(old_path), "AI_PDF_RENAMER_NEW_PATH": str(new_path)}
    try:
        meta_json = json.dumps(meta, default=str)
    except (TypeError, ValueError):
        meta_json = "{}"
    env["AI_PDF_RENAMER_META"] = meta_json
    try:
        subprocess.run(
            hook_cmd,
            shell=True,
            env=env,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Post-rename hook timed out (120s): %s", hook_cmd[:80])
    except Exception as e:
        logger.warning("Post-rename hook failed: %s", e)


def _apply_post_rename_actions(
    config: RenamerConfig,
    file_path: Path,
    target: Path,
    current_base: str,
    meta: dict,
    export_rows: list[dict],
) -> None:
    """Write rename log, PDF metadata, and export row after a successful rename. Mutates export_rows."""
    if config.rename_log_path:
        Path(config.rename_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.rename_log_path, "a", encoding="utf-8") as f:
            f.write(f"{file_path}\t{target}\n")
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


def _effective_max_tokens(config: RenamerConfig) -> int:
    """Max tokens for PDF extraction from config or env (AI_PDF_RENAMER_MAX_TOKENS)."""
    if config.max_tokens_for_extraction is not None and config.max_tokens_for_extraction > 0:
        return config.max_tokens_for_extraction
    try:
        v = int(os.environ.get("AI_PDF_RENAMER_MAX_TOKENS", "") or 0)
        if v > 0:
            return v
    except ValueError:
        pass
    return DEFAULT_MAX_CONTENT_TOKENS


def _extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    """Extract text from PDF (OCR or plain) according to config. Used by _process_one_file and single-worker loop.
    When vision_first is True, tries vision on first page first; on success returns (vision_result, True).
    When use_vision_fallback is True and extracted text is shorter than vision_fallback_min_text_len,
    tries Ollama vision on the first page and uses the result as synthetic content.
    Returns (content, used_vision_fallback)."""
    if getattr(config, "vision_first", False):
        image_b64 = pdf_first_page_to_image_base64(path)
        if image_b64:
            client = _llm_client_from_config(config)
            model = getattr(config, "vision_model", None) or client.model
            prompt = build_vision_filename_prompt(config.language)
            timeout = getattr(config, "llm_timeout_s", None) or 60.0
            vision_text = complete_vision(
                client.base_url,
                model,
                image_b64,
                prompt,
                timeout_s=max(60.0, timeout * 2),
            )
            if vision_text:
                vision_text = sanitize_filename_from_llm(vision_text)
                return (vision_text, True)
        # vision_first but image or vision failed: fall through to text extraction
    if config.use_ocr:
        content = pdf_to_text_with_ocr(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=_effective_max_tokens(config),
            language=config.language,
        )
    else:
        content = pdf_to_text(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=_effective_max_tokens(config),
        )
    min_len = getattr(config, "vision_fallback_min_text_len", 50)
    if getattr(config, "use_vision_fallback", False) and len(content.strip()) < min_len:
        image_b64 = pdf_first_page_to_image_base64(path)
        if image_b64:
            client = _llm_client_from_config(config)
            model = getattr(config, "vision_model", None) or client.model
            prompt = build_vision_filename_prompt(config.language)
            timeout = getattr(config, "llm_timeout_s", None) or 60.0
            vision_text = complete_vision(
                client.base_url,
                model,
                image_b64,
                prompt,
                timeout_s=max(60.0, timeout * 2),
            )
            if vision_text:
                vision_text = sanitize_filename_from_llm(vision_text)
                logger.info(
                    "Used vision fallback for %s (text length %d < %d)",
                    path.name,
                    len(content.strip()),
                    min_len,
                )
                return (vision_text, True)
    return (content, False)


def _process_content_to_result(
    file_path: Path,
    content: str,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    used_vision: bool = False,
) -> tuple[Path, str | None, dict | None, BaseException | None]:
    """
    Generate filename from already-extracted content. Returns (path, new_base, meta, error).
    Caller must ensure content is non-empty if expecting a non-skip result.
    """
    try:
        override_cat = (config.override_category_map or {}).get(file_path.name) or force_category_for_basename(
            rules, file_path.name
        )
        pdf_meta = get_pdf_metadata(file_path) if getattr(config, "use_pdf_metadata_for_date", True) else None
        filename_str, meta = generate_filename(
            content,
            config=config,
            override_category=override_cat,
            pdf_metadata=pdf_meta,
            rules=rules,
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
) -> tuple[Path, str | None, dict | None, BaseException | None]:
    """
    Extract text and generate filename for one file. Returns (path, new_base, meta, error).
    If error is not None or new_base is None, the file should be counted as failed/skipped.
    Empty or unextractable PDF: returns (path, None, None, None); caller logs and skips the file.
    """
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
) -> tuple[str | None, dict | None, BaseException | None]:
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
        _path, new_base, meta, exc = _process_content_to_result(
            file_path, content, config, rules=rules, used_vision=used_vision
        )
        if exc is not None:
            return (None, None, exc)
        return (new_base, meta, None)
    except Exception as exc:
        return (None, None, exc)


def _produce_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
) -> list[tuple[Path, str | None, dict | None, BaseException | None]]:
    """Produce (file_path, new_base, meta, exc) per file; parallel or single-worker with prefetch."""
    workers = max(1, getattr(config, "workers", 1) or 1)
    if config.interactive:
        workers = 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda p: _process_one_file(p, config, rules),
                    files,
                )
            )
        order = {f: i for i, f in enumerate(files)}
        results.sort(key=lambda r: order.get(r[0], 0))
        return results
    results = []
    prefetched: Future[tuple[str, bool]] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, file_path in enumerate(files):
            try:
                content, used_vision = (
                    prefetched.result() if prefetched is not None else _extract_pdf_content(file_path, config)
                )
            except Exception as exc:
                results.append((file_path, None, None, exc))
                prefetched = None
                continue
            prefetched = executor.submit(_extract_pdf_content, files[i + 1], config) if i + 1 < len(files) else None
            if not content.strip():
                results.append((file_path, None, None, None))
                continue
            try:
                results.append(
                    _process_content_to_result(file_path, content, config, rules=rules, used_vision=used_vision)
                )
            except Exception as exc:
                results.append((file_path, None, None, exc))
    return results


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: RenamerConfig,
    files_override: list[Path] | None = None,
) -> None:
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
        return

    def _mtime_key(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=_mtime_key, reverse=True)

    if not config.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")

    renamed_count = 0
    skipped_count = 0
    failed_count = 0
    export_rows: list[dict] = []
    plan_entries: list[dict[str, str]] = []

    results = _produce_rename_results(files, config, rules=rules)

    for i, (file_path, new_base, meta, exc) in enumerate(results):
        logger.info("Processing %s/%s: %s", i + 1, len(files), file_path)
        if exc is not None:
            # Data-file/config errors (e.g. invalid JSON) should propagate so CLI can exit with clear message.
            if isinstance(exc, ValueError) and "Invalid JSON in data file" in str(exc):
                raise exc
            if isinstance(exc, ValueError) and "No text extracted from" in str(exc):
                logger.warning("Skipping %s: %s", file_path.name, exc)
                skipped_count += 1
                continue
            logger.exception("Failed to process %s: %s", file_path, exc)
            failed_count += 1
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
            if getattr(config, "manual_mode", False):
                print(f"Suggested: {new_base}{file_path.suffix}")
                for k, v in (meta or {}).items():
                    if k in ("category", "summary", "keywords", "category_source") and v:
                        print(f"  {k}: {v}")
            reply, base, target = _interactive_rename_prompt(
                file_path,
                target,
                new_base,
                edit_default_base=new_base if getattr(config, "manual_mode", False) else None,
            )
            if reply == "n":
                skipped_count += 1
                continue
        try:

            def _on_rename_success(
                _fp: Path, _target: Path, _current_base: str, _meta: dict = meta, _rows: list = export_rows
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
            else:
                if config.dry_run:
                    logger.info(
                        "Dry-run: would rename '%s' to '%s'",
                        file_path.name,
                        target.name,
                    )
                else:
                    logger.info("Renamed '%s' to '%s'", file_path.name, target.name)
                renamed_count += 1
        except Exception as e:
            logger.exception("Failed to process %s: %s", file_path, e)
            failed_count += 1

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

    if not files:
        logger.info("No PDFs found in %s", path)
        print(f"No PDFs found in {path}.", file=sys.stderr)
    else:
        logger.info(
            "Summary: %s file(s) processed, %s renamed, %s skipped, %s failed",
            len(files),
            renamed_count,
            skipped_count,
            failed_count,
        )
        print(
            f"Processed {len(files)}, renamed {renamed_count}, skipped {skipped_count}, failed {failed_count}.",
            file=sys.stderr,
        )


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

    stop_event = False

    def handle_stop(sig, frame):
        nonlocal stop_event
        logger.info("Watch mode: received signal %s, stopping...", sig)
        stop_event = True

    # Handle SIGTERM (Docker/systemd) and SIGINT (Ctrl+C)
    original_sigterm = signal.signal(signal.SIGTERM, handle_stop)
    original_sigint = signal.signal(signal.SIGINT, handle_stop)

    seen: dict[Path, float] = {}
    logger.info("Watch mode: scanning %s every %.1fs (Ctrl+C or SIGTERM to stop)", path, interval_seconds)
    try:
        while not stop_event:
            try:
                # Cleanup 'seen' map: remove files that no longer exist
                to_remove = [p for p in seen if not p.exists()]
                for p in to_remove:
                    del seen[p]

                files = _collect_pdf_files(
                    path,
                    recursive=config.recursive,
                    max_depth=config.max_depth,
                    include_patterns=config.include_patterns,
                    exclude_patterns=config.exclude_patterns,
                    skip_if_already_named=config.skip_if_already_named,
                    files_override=None,
                )
                to_process: list[Path] = []
                for p in files:
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
                        if stop_event:
                            break
                        rename_pdfs_in_directory(
                            path,
                            config=config,
                            files_override=[single],
                        )
                if not stop_event:
                    time.sleep(interval_seconds)
            except Exception as exc:
                if not stop_event:
                    logger.exception("Watch iteration failed: %s", exc)
                    time.sleep(interval_seconds)
    finally:
        # Restore original handlers
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
        logger.info("Watch stopped")
