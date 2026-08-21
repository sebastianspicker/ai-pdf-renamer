# Releasing Folionym

This checklist defines the public prerelease path. It is intentionally manual:
publishing requires a clean, reviewable commit and explicit maintainer action.
The current alpha identity is `0.4.0a1`, tagged as `v0.4.0a1`.

## 1. Freeze the candidate

1. Confirm every runtime import, test, and public document is tracked. A tag
   records commits, not uncommitted working-tree files.
2. Confirm `src/folionym/__init__.py`, `CHANGELOG.md`, and the intended tag
   all use the same PEP 440 version. Keep the changes under `Unreleased` until
   the release date is chosen, then create the dated `0.4.0a1` heading in the
   final candidate commit.
3. Remove local build products and inspect the tree:

   ```bash
   make clean
   git status --short
   git diff --check
   ```

4. Confirm the candidate has no untracked public assets and that the release
   documents are part of the commit:

   ```bash
   test -z "$(git ls-files --others --exclude-standard)"
   git ls-files --error-unmatch \
     .github/release.yml \
     RELEASING.md \
     docs/releases/0.4.0a1.md \
     docs/tui.md
   ```

5. Review the complete diff. Do not include PDFs, document-derived output,
   logs, caches, local status files, private review packets, or local automation
   workspaces.

## 2. Validate the candidate before review

Run the project gates without external LLM, OCR, or PDF data:

```bash
uv lock --check
make release-check
uv run folionym --validate-config --dir . --no-llm --dry-run
```

`make release-check` builds one wheel and one source distribution, verifies
their public contents, installs the wheel in a disposable environment, and
checks all four console entry-point mappings.

The alpha requires CPython 3.14 or later; release verification uses exact
CPython 3.14.6. Linux runs the complete release gate, while macOS and Windows
run targeted smoke checks. Do not broaden the public support claim from local
results alone.

These checks validate the proposed candidate. Do not retain or publish their
artifacts: the release artifacts must be rebuilt after merge from the exact
commit that will be tagged.

## 3. Review and merge

1. Commit the complete candidate and open a focused pull request.
2. Wait for CI and Security checks on the exact pull-request commit.
3. Merge according to branch protection. Do not tag from the pre-merge working
   tree or reuse its `dist/` directory.

## 4. Rebuild the exact release tree

Start from a clean checkout of the commit that will receive the tag. The
commands below assume that commit is the current `main` tip:

```bash
git switch main
git pull --ff-only
RELEASE_COMMIT=$(git rev-parse HEAD)
make clean
test -z "$(git status --porcelain)"
uv lock --check
make release-check
uv run folionym --validate-config --dir . --no-llm --dry-run
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git rev-parse HEAD > dist/RELEASE_COMMIT
```

If any command changes a tracked or untracked file, stop and review it. Repeat
this section after committing the correction; never publish artifacts from a
different tree than the tag target.

## 5. Record artifacts

After the final build, retain checksums beside the artifacts:

```bash
(cd dist && shasum -a 256 *.whl *.tar.gz > SHA256SUMS)
python -m zipfile --list dist/*.whl
tar -tzf dist/*.tar.gz
```

Review the listings. They must contain the current runtime modules and bundled
data, and must not contain local docs, secrets, PDFs, caches, or tool state.

## 6. Tag and verify remotely

Create an annotated tag on the exact commit recorded during section 4:

```bash
RELEASE_COMMIT=$(sed -n '1p' dist/RELEASE_COMMIT)
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git tag -a v0.4.0a1 "$RELEASE_COMMIT" -m "Folionym 0.4.0a1"
git push origin v0.4.0a1
```

Wait for the tag-triggered CI and Security workflows to pass. If they fail, fix
forward with a new prerelease version; do not move a published tag.

## 7. Publish the GitHub prerelease

Create the prerelease only after the remote tag exists and its checks are green:

```bash
gh release create v0.4.0a1 \
  dist/*.whl dist/*.tar.gz dist/SHA256SUMS \
  --verify-tag \
  --prerelease \
  --latest=false \
  --notes-file docs/releases/0.4.0a1.md \
  --title "Folionym 0.4.0a1"
```

Verify the rendered authored notes, artifact names, checksums, install command,
alpha limitations, and security-reporting link on GitHub. Package-index
publication is a separate action and is out of scope until trusted publishing
and the package-name/version checks are configured.

## Rollback rule

Do not overwrite an uploaded artifact or move a public tag. For a release
defect, mark the affected prerelease clearly, document the issue, and publish a
new increment such as `0.4.0a2` after the full checklist passes again.
