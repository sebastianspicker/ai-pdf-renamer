"""
Desktop GUI for AI-PDF-Renamer: folder picker, options, dry-run preview, apply.

Runs the renamer in a background thread and shows progress in a log area.
Requires: tkinter (stdlib), and [pdf] for PDF extraction.
Aligned with current CLI options: template, date format, PDF metadata, structured fields,
preset, skip-already-named, backup, rename log, recursive, write PDF metadata, project/version.
"""

from __future__ import annotations

import logging
import queue
import re
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .logging_utils import setup_logging
from .rename_ops import apply_single_rename, sanitize_filename_base
from .renamer import (
    RenamerConfig,
    build_config_from_flat_dict,
    rename_pdfs_in_directory,
    suggest_rename_for_file,
)

# --- Constants for theming ---
FONT_FAMILY = "Helvetica"
FONT_SIZE = 10
PAD = 8
PAD_SM = 4
BG_DARK = "#1e1e2e"
FG_DARK = "#cdd6f4"
BG_LOG_DARK = "#11111b"
ACCENT_DARK = "#89b4fa"
BG_LIGHT = "#f5f5f5"
FG_LIGHT = "#1e1e2e"
BG_LOG_LIGHT = "#ffffff"


def _run_renamer(
    directory: str,
    config: RenamerConfig,
    log_queue: queue.Queue[str],
) -> None:
    """Run rename in current thread; push log lines to log_queue."""
    handler = _QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        rename_pdfs_in_directory(directory, config=config)
    except Exception as exc:
        log_queue.put(f"Error: {exc!s}\n")
    finally:
        root_logger.removeHandler(handler)
        log_queue.put(None)  # sentinel: done


class _QueueHandler(logging.Handler):
    """Send log records as single lines into a queue for the GUI."""

    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self._queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._queue.put(msg + "\n")
        except Exception:
            self.handleError(record)


_PROCESSING_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")


def _poll_log(
    log_queue: queue.Queue[str],
    text: scrolledtext.ScrolledText,
    on_done: Callable[[], None] | None = None,
    status_var: tk.StringVar | None = None,
) -> None:
    """Poll log queue and append to text widget; update status; reschedule until sentinel."""
    try:
        while True:
            line = log_queue.get_nowait()
            if line is None:
                if status_var is not None:
                    status_var.set("Idle")
                if on_done:
                    on_done()
                return
            text.insert(tk.END, line)
            text.see(tk.END)
            if status_var is not None:
                m = _PROCESSING_RE.search(line)
                if m:
                    status_var.set(f"Processing {m.group(1)}/{m.group(2)}")
    except queue.Empty:
        pass
    text.after(100, lambda: _poll_log(log_queue, text, on_done, status_var))


def _build_config_from_gui(vars: dict) -> RenamerConfig:
    """Build RenamerConfig from GUI variables. Uses shared build_config_from_flat_dict."""
    preset = (vars["preset"].get() or "").strip()
    skip_llm_score = None
    skip_llm_gap = None
    if preset == "high-confidence-heuristic":
        skip_llm_score = 0.5
        skip_llm_gap = 0.3
    data = {
        "language": (vars["language"].get() or "de").strip(),
        "desired_case": (vars["case"].get() or "kebabCase").strip(),
        "project": (vars["project"].get() or "").strip() or "",
        "version": (vars["version"].get() or "").strip() or "",
        "dry_run": vars["dry_run"].get(),
        "use_llm": vars["use_llm"].get(),
        "use_ocr": vars["use_ocr"].get(),
        "date_locale": (vars["date_format"].get() or "dmy").strip(),
        "use_pdf_metadata_for_date": vars["use_pdf_metadata_date"].get(),
        "use_structured_fields": vars["use_structured_fields"].get(),
        "skip_if_already_named": vars["skip_already_named"].get(),
        "recursive": vars["recursive"].get(),
        "backup_dir": (vars["backup_dir"].get() or "").strip() or None,
        "rename_log_path": (vars["rename_log"].get() or "").strip() or None,
        "write_pdf_metadata": vars["write_pdf_metadata"].get(),
        "filename_template": (vars["template"].get() or "").strip() or None,
        "use_timestamp_fallback": True,
        "timestamp_fallback_segment": "document",
        "simple_naming_mode": vars["simple_naming_mode"].get(),
        "use_vision_fallback": vars["use_vision_fallback"].get(),
        "vision_fallback_min_text_len": 50,
        "vision_model": None,
        "vision_first": vars["vision_first"].get(),
        "workers": 1,
        "skip_llm_category_if_heuristic_score_ge": skip_llm_score,
        "skip_llm_category_if_heuristic_gap_ge": skip_llm_gap,
    }
    if preset == "scanned":
        data["use_vision_fallback"] = True
        data["simple_naming_mode"] = True
    return build_config_from_flat_dict(data)


def main() -> None:
    setup_logging(level=logging.INFO)

    root = tk.Tk()
    root.title("AI-PDF-Renamer")
    root.minsize(560, 520)
    root.geometry("720x620")

    # Style
    style = ttk.Style()
    style.configure(".", font=(FONT_FAMILY, FONT_SIZE), padding=PAD_SM)
    style.configure("TFrame", padding=PAD)
    style.configure("TLabelframe", padding=PAD)
    style.configure("TLabelframe.Label", font=(FONT_FAMILY, FONT_SIZE, "bold"))
    style.configure("TButton", padding=(12, 6))

    # Variables (for _build_config_from_gui)
    vars: dict = {}
    vars["language"] = tk.StringVar(value="de")
    vars["case"] = tk.StringVar(value="kebabCase")
    vars["template"] = tk.StringVar(value="")
    vars["project"] = tk.StringVar(value="")
    vars["version"] = tk.StringVar(value="")
    vars["backup_dir"] = tk.StringVar(value="")
    vars["rename_log"] = tk.StringVar(value="")
    vars["date_format"] = tk.StringVar(value="dmy")
    vars["dry_run"] = tk.BooleanVar(value=True)
    vars["use_llm"] = tk.BooleanVar(value=True)
    vars["use_ocr"] = tk.BooleanVar(value=False)
    vars["use_pdf_metadata_date"] = tk.BooleanVar(value=True)
    vars["use_structured_fields"] = tk.BooleanVar(value=True)
    vars["skip_already_named"] = tk.BooleanVar(value=False)
    vars["recursive"] = tk.BooleanVar(value=False)
    vars["write_pdf_metadata"] = tk.BooleanVar(value=False)
    vars["preset"] = tk.StringVar(value="")
    vars["use_vision_fallback"] = tk.BooleanVar(value=False)
    vars["simple_naming_mode"] = tk.BooleanVar(value=False)
    vars["vision_first"] = tk.BooleanVar(value=False)
    vars["dark_theme"] = tk.BooleanVar(value=False)

    # --- Directory ---
    f_dir = ttk.LabelFrame(root, text="Directory", padding=PAD)
    f_dir.pack(fill=tk.X, padx=PAD, pady=(PAD, PAD_SM))
    dir_var = tk.StringVar(value="")
    row_dir = ttk.Frame(f_dir)
    row_dir.pack(fill=tk.X)
    ttk.Label(row_dir, text="Folder:").pack(side=tk.LEFT, padx=(0, PAD_SM))
    entry_dir = ttk.Entry(row_dir, textvariable=dir_var, width=52)
    entry_dir.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM)

    def choose_dir() -> None:
        path = filedialog.askdirectory(title="Select folder with PDFs")
        if path:
            dir_var.set(path)

    ttk.Button(row_dir, text="Browse…", command=choose_dir).pack(side=tk.LEFT)

    single_file_var = tk.StringVar(value="")
    row_single = ttk.Frame(f_dir)
    row_single.pack(fill=tk.X, pady=(PAD_SM, 0))
    ttk.Label(row_single, text="Single file (optional):").pack(side=tk.LEFT, padx=(0, PAD_SM))
    entry_single = ttk.Entry(row_single, textvariable=single_file_var, width=52)
    entry_single.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM)

    def choose_single_file() -> None:
        path = filedialog.askopenfilename(
            title="Select a single PDF",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
        )
        if path:
            single_file_var.set(path)

    ttk.Button(row_single, text="Browse…", command=choose_single_file).pack(side=tk.LEFT)

    # --- Naming ---
    f_naming = ttk.LabelFrame(root, text="Naming", padding=PAD)
    f_naming.pack(fill=tk.X, padx=PAD, pady=PAD_SM)
    grid_n = ttk.Frame(f_naming)
    grid_n.pack(fill=tk.X)
    ttk.Label(grid_n, text="Language:").grid(row=0, column=0, sticky=tk.W, padx=(0, PAD_SM), pady=2)
    ttk.Combobox(
        grid_n,
        textvariable=vars["language"],
        values=("de", "en"),
        state="readonly",
        width=8,
    ).grid(row=0, column=1, sticky=tk.W, pady=2)
    ttk.Label(grid_n, text="Case:").grid(row=1, column=0, sticky=tk.W, padx=(0, PAD_SM), pady=2)
    ttk.Combobox(
        grid_n,
        textvariable=vars["case"],
        values=("kebabCase", "snakeCase", "camelCase"),
        state="readonly",
        width=12,
    ).grid(row=1, column=1, sticky=tk.W, pady=2)
    ttk.Label(grid_n, text="Date format:").grid(row=2, column=0, sticky=tk.W, padx=(0, PAD_SM), pady=2)
    ttk.Combobox(
        grid_n,
        textvariable=vars["date_format"],
        values=("dmy", "mdy"),
        state="readonly",
        width=8,
    ).grid(row=2, column=1, sticky=tk.W, pady=2)
    ttk.Label(grid_n, text="Template (optional):").grid(row=3, column=0, sticky=tk.NW, padx=(0, PAD_SM), pady=2)
    ttk.Entry(
        grid_n,
        textvariable=vars["template"],
        width=48,
    ).grid(row=3, column=1, sticky=tk.EW, pady=2)
    ttk.Label(
        grid_n,
        text="Placeholders: {date} {category} {keywords} {summary} {invoice_id} {amount} {company}",
        font=(FONT_FAMILY, FONT_SIZE - 1),
    ).grid(row=4, column=1, sticky=tk.W, pady=(0, 2))
    ttk.Label(grid_n, text="Project / Version:").grid(row=5, column=0, sticky=tk.W, padx=(0, PAD_SM), pady=2)
    f_pv = ttk.Frame(grid_n)
    f_pv.grid(row=5, column=1, sticky=tk.W, pady=2)
    ttk.Entry(f_pv, textvariable=vars["project"], width=18).pack(side=tk.LEFT, padx=(0, PAD_SM))
    ttk.Entry(f_pv, textvariable=vars["version"], width=18).pack(side=tk.LEFT)
    grid_n.columnconfigure(1, weight=1)

    # --- Recognition & behaviour ---
    f_rec = ttk.LabelFrame(root, text="Recognition & behaviour", padding=PAD)
    f_rec.pack(fill=tk.X, padx=PAD, pady=PAD_SM)
    rec_inner = ttk.Frame(f_rec)
    rec_inner.pack(fill=tk.X)
    ttk.Checkbutton(
        rec_inner,
        text="Use LLM (summary, keywords, category). Uncheck for heuristics-only.",
        variable=vars["use_llm"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        rec_inner,
        text="Extract structured fields (invoice_id, amount, company) for template placeholders",
        variable=vars["use_structured_fields"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        rec_inner,
        text="Use PDF CreationDate/ModDate when no date in text",
        variable=vars["use_pdf_metadata_date"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        rec_inner,
        text="OCR for scanned PDFs (requires [ocr] + Tesseract)",
        variable=vars["use_ocr"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        rec_inner,
        text="Recursive (include subdirectories)",
        variable=vars["recursive"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        rec_inner,
        text="Skip files already named (YYYYMMDD-*.pdf)",
        variable=vars["skip_already_named"],
    ).pack(anchor=tk.W)
    row_preset = ttk.Frame(rec_inner)
    row_preset.pack(anchor=tk.W, pady=(PAD_SM, 0))
    ttk.Label(row_preset, text="Preset:").pack(side=tk.LEFT, padx=(0, PAD_SM))
    ttk.Combobox(
        row_preset,
        textvariable=vars["preset"],
        values=("", "high-confidence-heuristic", "scanned"),
        state="readonly",
        width=24,
    ).pack(side=tk.LEFT)
    ttk.Label(row_preset, text=" (heuristic-led / scanned)").pack(side=tk.LEFT)

    # --- Safety & output ---
    f_safe = ttk.LabelFrame(root, text="Safety & output", padding=PAD)
    f_safe.pack(fill=tk.X, padx=PAD, pady=PAD_SM)
    safe_inner = ttk.Frame(f_safe)
    safe_inner.pack(fill=tk.X)
    ttk.Checkbutton(
        safe_inner,
        text="Dry run (preview only, do not rename)",
        variable=vars["dry_run"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        safe_inner,
        text="Write new filename into PDF /Title metadata after rename",
        variable=vars["write_pdf_metadata"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        safe_inner,
        text="Use vision fallback when text is short",
        variable=vars["use_vision_fallback"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        safe_inner,
        text="Simple naming (single LLM call for filename)",
        variable=vars["simple_naming_mode"],
    ).pack(anchor=tk.W)
    ttk.Checkbutton(
        safe_inner,
        text="Vision first (skip text extraction; use first-page image only; requires vision model)",
        variable=vars["vision_first"],
    ).pack(anchor=tk.W)
    row_backup = ttk.Frame(safe_inner)
    row_backup.pack(fill=tk.X, pady=(PAD_SM, 0))
    ttk.Label(row_backup, text="Backup dir (optional):").pack(side=tk.LEFT, padx=(0, PAD_SM))
    ttk.Entry(row_backup, textvariable=vars["backup_dir"], width=32).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM
    )

    def choose_backup() -> None:
        path = filedialog.askdirectory(title="Backup directory (copy before rename)")
        if path:
            vars["backup_dir"].set(path)

    ttk.Button(row_backup, text="…", command=choose_backup, width=3).pack(side=tk.LEFT)
    row_log = ttk.Frame(safe_inner)
    row_log.pack(fill=tk.X, pady=(PAD_SM, 0))
    ttk.Label(row_log, text="Rename log (optional):").pack(side=tk.LEFT, padx=(0, PAD_SM))
    ttk.Entry(row_log, textvariable=vars["rename_log"], width=32).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM
    )

    def choose_rename_log() -> None:
        path = filedialog.asksaveasfilename(
            title="Rename log file (old→new for undo)",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if path:
            vars["rename_log"].set(path)

    ttk.Button(row_log, text="…", command=choose_rename_log, width=3).pack(side=tk.LEFT)

    # --- Theme ---
    def apply_theme(dark: bool) -> None:
        if dark:
            root.configure(bg=BG_DARK)
            log_text.configure(bg=BG_LOG_DARK, fg=FG_DARK, insertbackground=FG_DARK)
            entry_dir.configure(background=BG_LOG_DARK, foreground=FG_DARK)
        else:
            root.configure(bg=BG_LIGHT)
            log_text.configure(bg=BG_LOG_LIGHT, fg=FG_LIGHT, insertbackground=FG_LIGHT)
            entry_dir.configure(background=BG_LOG_LIGHT, foreground=FG_LIGHT)

    ttk.Checkbutton(
        root,
        text="Dark theme",
        variable=vars["dark_theme"],
        command=lambda: apply_theme(vars["dark_theme"].get()),
    ).pack(anchor=tk.W, padx=PAD, pady=(0, PAD_SM))

    # --- Status & log ---
    status_var = tk.StringVar(value="Idle")
    ttk.Label(root, textvariable=status_var).pack(anchor=tk.W, padx=PAD + 4, pady=(0, 2))
    f_log = ttk.LabelFrame(root, text="Log", padding=PAD_SM)
    f_log.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=PAD_SM)
    log_text = scrolledtext.ScrolledText(f_log, wrap=tk.WORD, height=12, state=tk.NORMAL, font=(FONT_FAMILY, 9))
    log_text.pack(fill=tk.BOTH, expand=True)

    # --- Buttons ---
    f_btn = ttk.Frame(root, padding=PAD)
    f_btn.pack(fill=tk.X)
    running: list[bool] = [False]

    def run(dry: bool) -> None:
        directory = dir_var.get().strip()
        if not directory:
            messagebox.showwarning("No directory", "Please select a directory with PDFs.")
            return
        if running[0]:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        running[0] = True
        status_var.set("Running…")
        log_text.insert(tk.END, f"\n--- {'Preview (dry run)' if dry else 'Apply renames'} ---\n")
        log_text.see(tk.END)
        try:
            vars["dry_run"].set(dry)
            config = _build_config_from_gui(vars)
        except ValueError as e:
            messagebox.showerror("Invalid options", str(e))
            running[0] = False
            return
        log_queue: queue.Queue[str] = queue.Queue()

        def work() -> None:
            _run_renamer(directory, config, log_queue)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def done() -> None:
            running[0] = False

        def poll() -> None:
            _poll_log(log_queue, log_text, on_done=done, status_var=status_var)

        root.after(100, poll)

    single_result_queue: queue.Queue = queue.Queue()

    def process_one_file() -> None:
        path_str = single_file_var.get().strip()
        if not path_str:
            path_str = filedialog.askopenfilename(
                title="Select a PDF to process",
                filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            )
            if path_str:
                single_file_var.set(path_str)
        if not path_str:
            messagebox.showwarning("No file", "Select a file first or choose one when prompted.")
            return
        if running[0]:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        file_path = Path(path_str)
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            messagebox.showerror("Invalid file", "Not a PDF file or file not found.")
            return
        try:
            config = _build_config_from_gui(vars)
        except ValueError as e:
            messagebox.showerror("Invalid options", str(e))
            return
        running[0] = True
        status_var.set("Processing one file…")

        def work() -> None:
            new_base, meta, error = suggest_rename_for_file(file_path, config)
            single_result_queue.put((file_path, new_base, meta, error))

        threading.Thread(target=work, daemon=True).start()

        def poll_single() -> None:
            try:
                fp, new_base, meta, err = single_result_queue.get_nowait()
            except queue.Empty:
                root.after(100, poll_single)
                return
            running[0] = False
            status_var.set("Idle")
            if err is not None:
                messagebox.showerror("Error", str(err))
                return
            if new_base is None:
                messagebox.showinfo("Skipped", "File had no content or could not be processed.")
                return
            # Dialog: suggest name (editable), Apply / Cancel
            dlg = tk.Toplevel(root)
            dlg.title("Rename suggestion")
            dlg.transient(root)
            dlg.grab_set()
            ttk.Label(dlg, text=f"File: {fp.name}").pack(anchor=tk.W, padx=PAD, pady=(PAD, PAD_SM))
            meta_parts = []
            if meta:
                for k in ("category", "summary", "keywords"):
                    if meta.get(k):
                        meta_parts.append(f"{k}: {meta.get(k)}")
            if meta_parts:
                ttk.Label(dlg, text=" | ".join(meta_parts[:3]), font=(FONT_FAMILY, FONT_SIZE - 1)).pack(
                    anchor=tk.W, padx=PAD, pady=(0, PAD_SM)
                )
            suggested = new_base + fp.suffix
            edit_var = tk.StringVar(value=suggested)
            row_edit = ttk.Frame(dlg)
            row_edit.pack(fill=tk.X, padx=PAD, pady=PAD_SM)
            ttk.Label(row_edit, text="New name:").pack(side=tk.LEFT, padx=(0, PAD_SM))
            entry_edit = ttk.Entry(row_edit, textvariable=edit_var, width=50)
            entry_edit.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def do_apply() -> None:
                raw = edit_var.get().strip()
                if not raw:
                    messagebox.showwarning("Empty", "Enter a filename.", parent=dlg)
                    return
                base = raw.removesuffix(fp.suffix) if raw.lower().endswith(fp.suffix.lower()) else raw
                base = sanitize_filename_base(base)
                if not base:
                    messagebox.showwarning("Invalid", "Filename is invalid after sanitization.", parent=dlg)
                    return
                try:
                    success, target = apply_single_rename(
                        fp,
                        base,
                        plan_file_path=None,
                        plan_entries=[],
                        dry_run=False,
                        backup_dir=config.backup_dir,
                        on_success=None,
                        max_filename_chars=config.max_filename_chars,
                    )
                    if success:
                        log_text.insert(tk.END, f"Renamed to {target.name}\n")
                        log_text.see(tk.END)
                        messagebox.showinfo("Done", f"Renamed to {target.name}", parent=dlg)
                    else:
                        messagebox.showerror("Failed", "Rename did not succeed.", parent=dlg)
                except Exception as e:
                    messagebox.showerror("Error", str(e), parent=dlg)
                    return
                dlg.destroy()

            def do_cancel() -> None:
                dlg.destroy()

            f_btns = ttk.Frame(dlg)
            f_btns.pack(fill=tk.X, padx=PAD, pady=PAD)
            ttk.Button(f_btns, text="Apply", command=do_apply).pack(side=tk.LEFT, padx=(0, PAD_SM))
            ttk.Button(f_btns, text="Cancel", command=do_cancel).pack(side=tk.LEFT)
            dlg.protocol("WM_DELETE_WINDOW", do_cancel)

        root.after(100, poll_single)

    ttk.Button(f_btn, text="Preview (dry run)", command=lambda: run(True)).pack(side=tk.LEFT, padx=(0, PAD))
    ttk.Button(f_btn, text="Apply renames", command=lambda: run(False)).pack(side=tk.LEFT, padx=(0, PAD))
    ttk.Button(f_btn, text="Process one file", command=process_one_file).pack(side=tk.LEFT)

    root.mainloop()


if __name__ == "__main__":
    main()
    sys.exit(0)
