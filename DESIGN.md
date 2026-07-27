# Interface reference

This file records the visual and interaction conventions implemented by the
browser and terminal interfaces. The browser CSS in `frontend/src/` and the
Textual theme in `src/folionym/tui_assets.py` are authoritative when this
summary differs from code.

## Color

| Role | Value | Use |
| --- | --- | --- |
| Canvas | `#E9EEF2` | application background |
| Surface | `#FFFFFF` | inputs, ledger, and dialogs |
| Rail | `#E2E8EE` | filters and evidence regions |
| Ink | `#121820` | primary text |
| Muted | `#5C6774` | notes and metadata |
| Rule | `#C5CED8` | dividers |
| Strong rule | `#A8B3C0` | control borders |
| Primary | `#0B6A5A` | primary actions and ready state |
| Active | `#2C4A7C` | current stage and selected row |
| Warning | `#9A5A00` | review and external endpoint states |
| Failure | `#A51F2D` | failures and destructive confirmation |

Status color is paired with a word, label, or control state.

## Typography

The browser uses IBM Plex Sans for interface text and IBM Plex Mono for
filenames and paths. System sans-serif and monospace families are fallbacks.
Labels use sentence case. The TUI uses short uppercase status words such as
`IDLE`, `RUN`, `DONE`, `FAIL`, and `STOP`.

## Browser layout

- The header shows the application name, current scope, and endpoint locality.
- Source, Preview, and Apply form the primary workflow.
- Preview includes filters, a source-to-target ledger, and document evidence.
- The Apply action reports the exact selected count.
- Fine-tune settings use a side panel on wide screens and a full-width panel on
  narrow screens.
- Narrow preview rows place the source name above the proposed name.

## Interaction states

- Preview does not rename files.
- Browser Apply uses selected entries from the retained plan.
- Destructive confirmation puts the cancel action first.
- Empty, running, cancelled, skipped, failed, and completed states use explicit
  text.
- Keyboard focus remains visible.
- Reduced-motion preferences disable nonessential transitions.

## Implementation map

| Area | Path |
| --- | --- |
| Browser shell and routes | `frontend/src/App.tsx` |
| Browser components | `frontend/src/components/` |
| Browser pages | `frontend/src/pages/` |
| Browser global CSS | `frontend/src/styles.css`, `frontend/src/styles/` |
| Browser API client | `frontend/src/api.ts` |
| Textual application | `src/folionym/tui.py` |
| Textual forms and state | `src/folionym/tui_forms.py`, `src/folionym/tui_state.py` |
| Textual theme and formatters | `src/folionym/tui_assets.py` |

The browser supports responsive layouts represented by the tracked desktop,
tablet, and mobile screenshots. Terminal support targets color-capable
terminals at 80 by 24 cells or larger. Screen-reader and terminal combinations
remain platform-dependent and require manual testing.
