# ADR 0001: Preserve the rename filesystem boundary

Status: Accepted  
Date: 2026-08-09

## Context

Rename operations combine filename policy, collision handling, backups,
cross-filesystem fallbacks, dry runs, plan output, and completion callbacks.
Keeping those responsibilities in one module made the mutation boundary hard
to review, while callers already depended on imports from
`folionym.rename_ops`.

## Decision

`folionym.rename_ops` remains the stable public facade. Its implementation is
split into naming, backup, filesystem, and execution modules, but callers use
only the names exported by `rename_ops.__all__`.

Every implementation change must preserve these filesystem invariants:

- an existing target is never overwritten; collisions select a unique suffix
  unless an exact reviewed target was required;
- dry-run and plan output choose the same collision target as apply;
- backups are created once before mutation, use unique names, and are private
  where the platform supports POSIX permissions;
- cross-filesystem fallback reserves and validates the destination before
  unlinking the source, and cleans incomplete targets after failure;
- completion callbacks cannot turn an already completed rename into a failed
  filesystem operation.

## Consequences

New rename behavior belongs in the owning implementation module without
bypassing the facade. Changes to exports or invariants require an explicit
contract update and focused tests. The retained direct contracts are covered
by `tests/test_core.py`.
