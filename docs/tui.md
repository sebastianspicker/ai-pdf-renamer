# Terminal interface

Folionym `0.4.0a1` includes a Textual interface for configuring and supervising
PDF rename operations.

Install and launch it:

```bash
python -m pip install -e '.[tui,pdf]'
folionym-tui
```

## Workflow

The interface has three tabs:

1. Setup selects a directory or one PDF and configures common naming and
   extraction options.
2. Fine-tune configures output files, limits, rules, endpoint settings, vision,
   hooks, and worker count.
3. Review and rename displays progress, proposals, skips, failures, and the
   activity log.

Preview is a dry run. Apply starts a separate run from the current settings and
calculates proposals again, so a prior LLM-backed preview and a later Apply can
differ. Apply to folder and Rename one PDF open a keyboard-contained
confirmation whose first action is Cancel.

## Controls

| Shortcut | Action |
| --- | --- |
| `Ctrl+P` | start a directory Preview |
| `Ctrl+A` | open confirmation for a directory Apply |
| `Ctrl+C` | request cancellation after the current file |
| `Ctrl+Q` | request cancellation, then quit after the active file |

Starting a valid directory Preview or confirmed Apply run saves the current
form values to `~/.folionym_ui.json`. Edits are not autosaved on exit or by
Rename one PDF. An existing `~/.folionym_tui.json` is migrated without deleting
the old file. On POSIX systems, the current settings file is written with
owner-only permissions where supported.

## Status and terminal requirements

The Review and rename tab uses source-to-target rows and a bounded chronological
activity log. Keyboard row focus updates the selected-file details on wide
terminals. Status words include `IDLE`, `RUN`, `DONE`, `FAIL`, and `STOP`, so
state does not depend on color alone.

Use a color-capable terminal at least 80 columns by 24 rows. Wide layouts show
the selected-file details. Compact layouts hide that panel and arrange actions
in two columns. Terminal and screen-reader behavior depends on the platform and
requires manual verification.

## Endpoint and data handling

The TUI uses the shared HTTP LLM configuration. Endpoint URLs must be valid
HTTP(S) URLs without embedded credentials. LLM requests do not follow
redirects. Remote plain HTTP emits a warning unless HTTPS is required.

A configured LLM endpoint or post-rename hook can receive document-derived
content or metadata. Review [SECURITY.md](../SECURITY.md) before enabling
either. The TUI does not load models inside the process.

## Operating limits

Use the TUI for supervised processing of trusted documents. There is no hard
input byte limit, page extraction is unlimited by default, worker count has no
hard maximum, and OCR has no application-level timeout. Keep backups of
important documents.
