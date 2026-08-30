# Product scope

Folionym is a local PDF naming application. It extracts document text and
metadata, applies deterministic naming rules and optional HTTP LLM enrichment,
then lets a user supervise filesystem changes.

## Supported workflows

- CLI dry run, batch apply, interactive confirmation, manual single-file, and watch
- browser Source, Preview, and exact reviewed Apply
- Textual Setup, Fine-tune, Review and rename, and confirmed single-file rename
- rename logs and constrained undo
- optional OCR and vision for low-text documents
- plan, metadata, and summary exports

## Operating model

The browser and directory TUI build immutable reviewed plans and apply exact
selected or included ready targets without another naming pass. A changed source
or occupied target fails rather than receiving a replacement name. Directory
TUI material edits invalidate its plan, and cancellation leaves it
unapplyable. TUI Rename one PDF deliberately remains an immediate confirmed
unique-available operation. CLI `--dry-run` and apply are independent,
recomputing runs; `--plan-file` exports proposals only.

Folionym has no document storage, accounts, hosted API, remote deployment, or
interprocess writer lock. Optional LLM endpoints and post-rename hooks are
separate network trust boundaries.

## Interface requirements

- Keep source paths, proposed names, statuses, and mutation consequences visible.
- Require Preview before reviewed Apply and put Cancel first in destructive confirmation.
- Pair color with text labels and preserve keyboard access.
- Keep advanced extraction, endpoint, and output settings outside initial source selection.
- Bind the browser service to loopback only; do not represent it as remote authentication.

Implementation references are `frontend/`, `src/folionym/interfaces/tui/`, and
`src/folionym/interfaces/web/`. See [DESIGN.md](DESIGN.md) for code placement
and shared-plan architecture.
