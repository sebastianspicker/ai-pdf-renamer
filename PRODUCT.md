# Product scope

Folionym is a local PDF naming application for users who need to inspect and
rename individual files or directory batches. It extracts document text and
metadata, scores categories with heuristics, and can request additional naming
fields from an HTTP LLM endpoint.

## Supported workflows

- CLI dry run, apply, interactive confirmation, manual single-file, and watch
- browser Source, Preview, and Apply workflow
- Textual Setup, Fine-tune, and Review and rename workflow
- rename logs and constrained undo
- optional OCR and vision for low-text documents
- export of plans, metadata, and run summaries

## Operating model

The user selects local files and remains responsible for reviewing proposed
names. The browser Preview plan is retained for exact Apply operations. The TUI
Preview and Apply actions are independent runs. CLI mutation behavior depends
on `--dry-run`, `--interactive`, and the selected input mode.

Folionym does not provide document storage, user accounts, a hosted API, or a
remote deployment. Optional LLM endpoints and post-rename hooks are separate
network trust boundaries.

## Interface requirements

- Keep source paths, proposed names, status, and mutation consequences visible.
- Make preview actions available before rename actions.
- Pair color with text labels for status.
- Put Cancel before destructive confirmation actions.
- Preserve keyboard access to browser and terminal controls.
- Keep advanced extraction, endpoint, and output settings separate from the
  primary source selection workflow.

The browser implementation in `frontend/` and the Textual implementation in
`src/folionym/tui.py`, `src/folionym/tui_forms.py`,
`src/folionym/tui_state.py`, and `src/folionym/tui_assets.py` are the current
references. Accessibility support still requires manual verification across
browsers, terminals, and assistive technology combinations.
