# Folionym architecture

Folionym is a local-first PDF naming system. It accepts a PDF or directory,
extracts document signals, generates a filename proposal, and either reports
or applies guarded filesystem changes. The CLI, Textual TUI, and loopback web
application are adapters over the same application contracts.

## Flow

```text
interface input -> settings -> discovery -> extraction -> naming + optional LLM
                -> proposal or reviewed plan -> rename policy -> artifacts/hooks
```

`settings` creates the canonical `RenamerConfig`. `application` discovers
files, schedules proposal production, owns reviewed plans, and writes optional
artifacts. A proposal combines `extraction` results with `naming` rules and
optional `llm` enrichment. `rename_ops` is the sole production boundary for
backup creation and atomic rename attempts.

## Packages and boundaries

| Package | Responsibility |
| --- | --- |
| `settings` | typed configuration and precedence resolution |
| `naming` | rules, heuristics, templates, tokens, and filename composition |
| `extraction` | PDF text, metadata, OCR, and extraction strategies |
| `llm` | LLM request/response models, parsing, protocols, cache, and HTTP client |
| `application` | discovery, proposals, scheduling, batch/watch workflows, reviewed plans, artifacts, and hooks |
| `rename_ops` | filename safety, backups, collision handling, and atomic filesystem mutation |
| `infrastructure` | low-level file, private-I/O, HTTP-validation, logging, error, and resource primitives |
| `interfaces` | CLI, Textual, web, and shared interactive settings adapters |

Dependencies point inward: interfaces and application may depend on domain
packages and infrastructure; infrastructure does not import application or
interfaces. Keep transport construction at a composition boundary and keep
naming deterministic for its supplied dependencies. Do not move filesystem
work out of `rename_ops`.

## Facades and placement

`folionym.config`, `folionym.filename`, `folionym.heuristics`,
`folionym.renamer`, and `folionym.rename_ops` preserve public import and
workflow contracts while implementation lives in the packages above. New
internal code belongs with its responsibility; add a facade only for an
intentional compatibility commitment.

The four console interfaces are `folionym` (CLI), `folionym-tui` (Textual),
`folionym-undo` (undo CLI), and `folionym-web` (loopback web server). Browser
source lives in `frontend/`; its production assets are built into
`src/folionym/web_dist/`.

## Apply semantics

There are two explicit collision policies:

- **Unique available** is used by conventional batch CLI runs and the TUI
  single-file action. An occupied candidate may be retried with a suffix.
- **Exact reviewed** is used by browser and directory-TUI plans. Apply uses the
  reviewed target without recomputing a name. It rejects a changed source,
  duplicate selected target, or occupied target rather than selecting another.

`application.reviewed_plan` is shared by the browser and TUI. Its immutable
plan records source fingerprints, configuration, proposal status, inclusion,
and targets. Directory TUI Preview must complete before Apply; material source
or configuration changes invalidate it, and a cancelled preview cannot apply.
The immediate TUI single-file action deliberately does not create a reviewed
plan. CLI dry-run and apply remain independent invocations, while
`--plan-file` is an export artifact only.

## State and artifacts

Reviewed plans and web reports are process-local. UI settings are stored in
`~/.folionym_ui.json` with private-write handling where supported. Optional
rename logs, backups, metadata exports, summary JSON, and plan files are
application outputs, not a persistent reviewed-plan store. External LLM
endpoints and post-rename hooks are separate network trust boundaries; see
`SECURITY.md`.
