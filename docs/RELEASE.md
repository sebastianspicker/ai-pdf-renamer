# Release Procedure

Manual GitHub release process for AI-PDF-Renamer.

## Preconditions

1. You are on the intended release branch.
2. CI is green on latest commit.
3. Working tree is clean or only contains intended release edits.

## Release checklist

1. Run local gates:

```bash
make clean
make release-check
```

2. Verify repository health artifacts exist:

- `LICENSE`
- `SECURITY.md`
- `.github/ISSUE_TEMPLATE/*`
- `.github/pull_request_template.md`
- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`

3. Verify canonical docs are current:

- `README.md`
- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK.md`
- `docs/product-specs/PRD.md`
- `BUGS_AND_FIXES.md`
- `SECURITY.md`
- `CHANGELOG.md`

4. Verify version consistency:

- `pyproject.toml` (`[project].version`)
- `src/ai_pdf_renamer/__init__.py` (`__version__`)

5. Update changelog:

- Move relevant entries from `Unreleased` to a new version section.
- Add release date in `YYYY-MM-DD`.

## Tag and release

1. Create a version commit (if needed):

```bash
git add -A
git commit -m "release: vX.Y.Z"
```

2. Create annotated tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

3. Push branch and tag:

```bash
git push origin <branch>
git push origin vX.Y.Z
```

4. Create GitHub Release from the tag:

- Title: `vX.Y.Z`
- Notes source: `CHANGELOG.md` version section
- Include highlights, compatibility notes, and known limitations.

## Post-release verification

1. Confirm CI and security workflows complete successfully for the tagged commit.
2. Validate README quick-start commands on a clean environment.
3. Open one smoke issue if release notes need immediate correction; patch in follow-up release.

