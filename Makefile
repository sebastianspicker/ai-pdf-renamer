UV ?= uv

.PHONY: install-dev lint format typecheck test e2e cov clean hygiene-check build-check release-check ci

install-dev:
	$(UV) sync --extra dev --extra pdf --extra tui

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy src/ai_pdf_renamer/

test:
	$(UV) run pytest -q

e2e:
	$(UV) run pytest -q tests/e2e

cov:
	$(UV) run pytest --cov=ai_pdf_renamer --cov-report=term-missing --cov-fail-under=85 -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache
	rm -rf .coverage .coverage.* coverage.xml coverage htmlcov test-results
	rm -rf build dist
	rm -rf src/ai_pdf_renamer.egg-info *.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete

hygiene-check:
	git ls-files -z | $(UV) run python scripts/repository_hygiene.py --null-stdin

build-check:
	rm -rf dist
	$(UV) build --out-dir dist
	@set -eu; \
	ENV_DIR=$$(mktemp -d); \
	trap 'rm -rf "$$ENV_DIR"' EXIT; \
	set -- dist/*.whl; \
	[ "$$#" -eq 1 ]; \
	WHEEL=$$1; \
	PYTHON=$$($(UV) python find 3.11); \
	$(UV) venv --no-project --python "$$PYTHON" "$$ENV_DIR"; \
	$(UV) pip install --python "$$ENV_DIR/bin/python" "$${WHEEL}[tui]"; \
	"$$ENV_DIR/bin/python" scripts/verify_distributions.py dist --installed-wheel

release-check: hygiene-check lint typecheck cov build-check

ci: release-check
