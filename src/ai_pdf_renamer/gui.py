"""Tkinter GUI for AI-PDF-Renamer.

Provides a tabbed interface:
- Basic: common options and folder/file pickers
- Advanced: LLM, rules, export and performance controls
- Run: progress, logs, preview/apply/cancel actions
"""

from __future__ import annotations

import json
import logging
import queue
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import RenamerConfig
from .config_resolver import build_config
from .logging_utils import setup_logging
from .rename_ops import apply_single_rename, sanitize_filename_base
from .renamer import rename_pdfs_in_directory, suggest_rename_for_file

logger = logging.getLogger(__name__)

FONT_FAMILY = "Helvetica"
FONT_SIZE = 10
PAD = 8
PAD_SM = 4
SETTINGS_PATH = Path.home() / ".ai_pdf_renamer_gui.json"
PROCESS_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")


class _QueueHandler(logging.Handler):
    """Send log records to queue for GUI consumption."""

    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self._queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _load_settings() -> dict[str, object]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        raw = SETTINGS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_settings(data: dict[str, object]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.debug("Could not write GUI settings", exc_info=True)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI-PDF-Renamer")
        self.root.minsize(840, 620)
        self.root.geometry("980x740")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.result_queue: queue.Queue[tuple[bool, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False

        self.status_var = tk.StringVar(value="Idle")
        self.progress_cur = tk.IntVar(value=0)
        self.progress_max = tk.IntVar(value=100)

        self.vars = self._build_vars()
        self._load_vars_from_settings(_load_settings())

        self._build_ui()

    def _build_vars(self) -> dict[str, tk.Variable]:
        vars: dict[str, tk.Variable] = {
            "directory": tk.StringVar(value=""),
            "single_file": tk.StringVar(value=""),
            "language": tk.StringVar(value="de"),
            "case": tk.StringVar(value="kebabCase"),
            "date_format": tk.StringVar(value="dmy"),
            "project": tk.StringVar(value=""),
            "version": tk.StringVar(value=""),
            "preset": tk.StringVar(value=""),
            "template": tk.StringVar(value=""),
            "backup_dir": tk.StringVar(value=""),
            "rename_log": tk.StringVar(value=""),
            "export_metadata": tk.StringVar(value=""),
            "summary_json": tk.StringVar(value=""),
            "rules_file": tk.StringVar(value=""),
            "post_rename_hook": tk.StringVar(value=""),
            "llm_url": tk.StringVar(value=""),
            "llm_model": tk.StringVar(value=""),
            "llm_timeout": tk.StringVar(value=""),
            "max_tokens": tk.StringVar(value=""),
            "max_content_chars": tk.StringVar(value=""),
            "max_content_tokens": tk.StringVar(value=""),
            "workers": tk.StringVar(value="1"),
            "max_filename_chars": tk.StringVar(value=""),
            "dry_run": tk.BooleanVar(value=True),
            "use_llm": tk.BooleanVar(value=True),
            "use_ocr": tk.BooleanVar(value=False),
            "use_pdf_metadata_date": tk.BooleanVar(value=True),
            "use_structured_fields": tk.BooleanVar(value=True),
            "skip_already_named": tk.BooleanVar(value=False),
            "recursive": tk.BooleanVar(value=False),
            "write_pdf_metadata": tk.BooleanVar(value=False),
            "use_vision_fallback": tk.BooleanVar(value=False),
            "simple_naming_mode": tk.BooleanVar(value=False),
            "vision_first": tk.BooleanVar(value=False),
        }
        return vars

    def _load_vars_from_settings(self, data: dict[str, object]) -> None:
        for key, var in self.vars.items():
            if key not in data:
                continue
            value = data[key]
            try:
                if isinstance(var, tk.BooleanVar):
                    if isinstance(value, str):
                        var.set(value.strip().lower() in {"1", "true", "yes", "on"})
                    else:
                        var.set(bool(value))
                else:
                    var.set(str(value) if value is not None else "")
            except (tk.TclError, TypeError, ValueError):
                continue

    def _settings_snapshot(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, var in self.vars.items():
            out[key] = bool(var.get()) if isinstance(var, tk.BooleanVar) else str(var.get())
        return out

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure(".", font=(FONT_FAMILY, FONT_SIZE))

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=PAD)

        tab_basic = ttk.Frame(nb, padding=PAD)
        tab_adv = ttk.Frame(nb, padding=PAD)
        tab_run = ttk.Frame(nb, padding=PAD)
        nb.add(tab_basic, text="Basic")
        nb.add(tab_adv, text="Advanced")
        nb.add(tab_run, text="Run")

        self._build_tab_basic(tab_basic)
        self._build_tab_advanced(tab_adv)
        self._build_tab_run(tab_run)

    def _build_tab_basic(self, parent: ttk.Frame) -> None:
        folder_row = ttk.Frame(parent)
        folder_row.pack(fill=tk.X, pady=PAD_SM)
        ttk.Label(folder_row, text="Folder:").pack(side=tk.LEFT, padx=(0, PAD_SM))
        ttk.Entry(folder_row, textvariable=self.vars["directory"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="Browse…", command=self._choose_folder).pack(side=tk.LEFT, padx=(PAD_SM, 0))

        single_row = ttk.Frame(parent)
        single_row.pack(fill=tk.X, pady=PAD_SM)
        ttk.Label(single_row, text="Single file:").pack(side=tk.LEFT, padx=(0, PAD_SM))
        ttk.Entry(single_row, textvariable=self.vars["single_file"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(single_row, text="Browse…", command=self._choose_single_file).pack(side=tk.LEFT, padx=(PAD_SM, 0))

        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, pady=PAD_SM)
        labels = [
            ("Language", "language", ("de", "en")),
            ("Case", "case", ("kebabCase", "snakeCase", "camelCase")),
            ("Date format", "date_format", ("dmy", "mdy")),
            ("Preset", "preset", ("", "high-confidence-heuristic", "scanned")),
        ]
        for i, (label, key, values) in enumerate(labels):
            ttk.Label(grid, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, PAD_SM), pady=PAD_SM)
            ttk.Combobox(grid, textvariable=self.vars[key], values=values, state="readonly", width=28).grid(
                row=i, column=1, sticky=tk.W, pady=PAD_SM
            )

        pv = ttk.Frame(parent)
        pv.pack(fill=tk.X, pady=PAD_SM)
        ttk.Label(pv, text="Project:").pack(side=tk.LEFT)
        ttk.Entry(pv, textvariable=self.vars["project"], width=24).pack(side=tk.LEFT, padx=(PAD_SM, PAD))
        ttk.Label(pv, text="Version:").pack(side=tk.LEFT)
        ttk.Entry(pv, textvariable=self.vars["version"], width=24).pack(side=tk.LEFT, padx=(PAD_SM, 0))

        cb = ttk.Frame(parent)
        cb.pack(fill=tk.X, pady=PAD_SM)
        self._check(cb, "Dry run", "dry_run")
        self._check(cb, "Use LLM", "use_llm")
        self._check(cb, "OCR for scans", "use_ocr")
        self._check(cb, "Recursive", "recursive")
        self._check(cb, "Skip already named", "skip_already_named")

    def _build_tab_advanced(self, parent: ttk.Frame) -> None:
        def row(label: str, key: str, browse: str | None = None) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=PAD_SM)
            ttk.Label(frame, text=f"{label}:", width=20).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=self.vars[key]).pack(side=tk.LEFT, fill=tk.X, expand=True)
            if browse == "file":
                ttk.Button(frame, text="…", width=3, command=lambda k=key: self._choose_file_for_var(k)).pack(
                    side=tk.LEFT, padx=(PAD_SM, 0)
                )
            if browse == "dir":
                ttk.Button(frame, text="…", width=3, command=lambda k=key: self._choose_dir_for_var(k)).pack(
                    side=tk.LEFT, padx=(PAD_SM, 0)
                )

        row("Template", "template")
        row("Backup dir", "backup_dir", browse="dir")
        row("Rename log", "rename_log", browse="file")
        row("Export metadata", "export_metadata", browse="file")
        row("Summary JSON", "summary_json", browse="file")
        row("Rules file", "rules_file", browse="file")
        row("Post-rename hook", "post_rename_hook")

        ttk.Separator(parent).pack(fill=tk.X, pady=PAD)

        row("LLM URL", "llm_url")
        row("LLM model", "llm_model")
        row("LLM timeout (s)", "llm_timeout")
        row("Max extraction tokens", "max_tokens")
        row("Max content chars", "max_content_chars")
        row("Max content tokens", "max_content_tokens")
        row("Workers", "workers")
        row("Max filename chars", "max_filename_chars")

        flags = ttk.Frame(parent)
        flags.pack(fill=tk.X, pady=PAD_SM)
        self._check(flags, "Use PDF metadata date", "use_pdf_metadata_date")
        self._check(flags, "Use structured fields", "use_structured_fields")
        self._check(flags, "Write PDF title metadata", "write_pdf_metadata")
        self._check(flags, "Vision fallback", "use_vision_fallback")
        self._check(flags, "Simple naming", "simple_naming_mode")
        self._check(flags, "Vision first", "vision_first")

    def _build_tab_run(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(
            parent,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=self.progress_max.get(),
            variable=self.progress_cur,
        )
        self.progress.pack(fill=tk.X, pady=(PAD_SM, PAD_SM))

        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=22)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(PAD_SM, PAD))

        btn = ttk.Frame(parent)
        btn.pack(fill=tk.X)
        ttk.Button(btn, text="Preview (dry run)", command=lambda: self._run(dry=True)).pack(side=tk.LEFT, padx=(0, PAD))
        ttk.Button(btn, text="Apply renames", command=lambda: self._run(dry=False)).pack(side=tk.LEFT, padx=(0, PAD))
        ttk.Button(btn, text="Process one file", command=self._process_one_file).pack(side=tk.LEFT, padx=(0, PAD))
        ttk.Button(btn, text="Cancel", command=self._cancel_run).pack(side=tk.LEFT)

    def _check(self, parent: ttk.Frame, label: str, key: str) -> None:
        ttk.Checkbutton(parent, text=label, variable=self.vars[key]).pack(anchor=tk.W)

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder with PDFs")
        if path:
            self.vars["directory"].set(path)

    def _choose_single_file(self) -> None:
        path = filedialog.askopenfilename(title="Select a PDF", filetypes=[("PDF", "*.pdf"), ("All", "*.*")])
        if path:
            self.vars["single_file"].set(path)

    def _choose_file_for_var(self, key: str) -> None:
        path = filedialog.asksaveasfilename(title=f"Select {key}", filetypes=[("All", "*.*")])
        if path:
            self.vars[key].set(path)

    def _choose_dir_for_var(self, key: str) -> None:
        path = filedialog.askdirectory(title=f"Select {key}")
        if path:
            self.vars[key].set(path)

    def _build_config(self, *, dry_run: bool, manual_mode: bool = False) -> RenamerConfig:
        raw = {
            "language": self.vars["language"].get(),
            "desired_case": self.vars["case"].get(),
            "project": self.vars["project"].get(),
            "version": self.vars["version"].get(),
            "date_locale": self.vars["date_format"].get(),
            "dry_run": dry_run,
            "use_llm": self.vars["use_llm"].get(),
            "use_ocr": self.vars["use_ocr"].get(),
            "use_pdf_metadata_for_date": self.vars["use_pdf_metadata_date"].get(),
            "use_structured_fields": self.vars["use_structured_fields"].get(),
            "skip_if_already_named": self.vars["skip_already_named"].get(),
            "recursive": self.vars["recursive"].get(),
            "backup_dir": self.vars["backup_dir"].get(),
            "rename_log_path": self.vars["rename_log"].get(),
            "export_metadata_path": self.vars["export_metadata"].get(),
            "summary_json_path": self.vars["summary_json"].get(),
            "rules_file": self.vars["rules_file"].get(),
            "post_rename_hook": self.vars["post_rename_hook"].get(),
            "llm_base_url": self.vars["llm_url"].get(),
            "llm_model": self.vars["llm_model"].get(),
            "llm_timeout_s": self.vars["llm_timeout"].get(),
            "max_tokens_for_extraction": self.vars["max_tokens"].get(),
            "max_content_chars": self.vars["max_content_chars"].get(),
            "max_content_tokens": self.vars["max_content_tokens"].get(),
            "workers": self.vars["workers"].get(),
            "max_filename_chars": self.vars["max_filename_chars"].get(),
            "write_pdf_metadata": self.vars["write_pdf_metadata"].get(),
            "filename_template": self.vars["template"].get(),
            "use_vision_fallback": self.vars["use_vision_fallback"].get(),
            "simple_naming_mode": self.vars["simple_naming_mode"].get(),
            "vision_first": self.vars["vision_first"].get(),
            "preset": self.vars["preset"].get(),
            "manual_mode": manual_mode,
            "interactive": manual_mode,
            "stop_event": self.stop_event,
        }
        return build_config(raw)

    def _run_worker(self, directory: str, config: RenamerConfig) -> None:
        handler = _QueueHandler(self.log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            rename_pdfs_in_directory(directory, config=config)
            self.result_queue.put((True, "Completed"))
        except Exception as exc:
            self.result_queue.put((False, str(exc)))
        finally:
            root_logger.removeHandler(handler)
            self.log_queue.put(None)

    def _run(self, *, dry_run: bool) -> None:
        if self.running:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        directory = str(self.vars["directory"].get()).strip()
        if not directory:
            messagebox.showwarning("No directory", "Please select a directory first.")
            return

        try:
            config = self._build_config(dry_run=dry_run)
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        _save_settings(self._settings_snapshot())

        self.stop_event.clear()
        self.running = True
        self.status_var.set("Running…")
        self.progress_cur.set(0)
        self.progress_max.set(100)
        self.progress.configure(maximum=100)
        self.log_text.insert(tk.END, f"\n--- {'Preview' if dry_run else 'Apply'} ---\n")
        self.log_text.see(tk.END)

        worker = threading.Thread(target=self._run_worker, args=(directory, config), daemon=True)
        worker.start()
        self.root.after(100, self._poll_log)

    def _process_one_file(self) -> None:
        if self.running:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        file_path = str(self.vars["single_file"].get()).strip()
        if not file_path:
            self._choose_single_file()
            file_path = str(self.vars["single_file"].get()).strip()
        if not file_path:
            return

        fp = Path(file_path)
        if not fp.exists() or fp.suffix.lower() != ".pdf":
            messagebox.showerror("Invalid file", "Please select an existing PDF file.")
            return

        try:
            config = self._build_config(dry_run=False, manual_mode=True)
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        new_base, meta, err = suggest_rename_for_file(fp, config)
        if err is not None:
            messagebox.showerror("Error", str(err))
            return
        if new_base is None:
            messagebox.showinfo("Skipped", "File had no extractable content.")
            return

        suggested = new_base + fp.suffix
        answer = messagebox.askyesno("Apply rename", f"Rename:\n{fp.name}\n→\n{suggested}\n\nApply now?")
        if not answer:
            return

        success, target = apply_single_rename(
            fp,
            sanitize_filename_base(new_base),
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=config.backup_dir,
            on_success=None,
            max_filename_chars=config.max_filename_chars,
        )
        if success:
            self.log_text.insert(tk.END, f"Renamed: {fp.name} -> {target.name}\n")
            if meta:
                self.log_text.insert(tk.END, f"Meta: {meta}\n")
            self.log_text.see(tk.END)
            messagebox.showinfo("Done", f"Renamed to {target.name}")
        else:
            messagebox.showerror("Failed", "Could not rename file.")

    def _cancel_run(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.status_var.set("Cancel requested…")

    def _poll_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line is None:
                    self.running = False
                    ok, msg = self.result_queue.get_nowait() if not self.result_queue.empty() else (True, "Completed")
                    self.status_var.set("Idle")
                    if not ok:
                        messagebox.showerror("Run failed", msg)
                    return

                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)

                match = PROCESS_RE.search(line)
                if match:
                    current = int(match.group(1))
                    total = max(1, int(match.group(2)))
                    self.progress_max.set(total)
                    self.progress.configure(maximum=total)
                    self.progress_cur.set(current)
                    self.status_var.set(f"Processing {current}/{total}")
        except queue.Empty:
            pass

        self.root.after(100, self._poll_log)


def main() -> None:
    setup_logging(level=logging.INFO)
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    sys.exit(0)
