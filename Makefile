UV ?= uv

.PHONY: install-dev frontend-install frontend-check lint format typecheck test e2e cov clean docs-screenshots hygiene-check build-check release-check ci

install-dev:
	$(UV) sync --all-extras
	cd frontend && npm ci

frontend-install:
	cd frontend && npm ci

frontend-check:
	cd frontend && npm run typecheck
	cd frontend && npm run test
	cd frontend && npm run build

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy src/folionym/ scripts/

test:
	$(UV) run pytest -q

e2e:
	$(UV) run pytest -q tests/e2e

cov:
	$(UV) run pytest --cov=folionym --cov-report=term-missing --cov-fail-under=85 -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache
	rm -rf .coverage .coverage.* coverage.xml coverage htmlcov test-results
	rm -rf build dist
	rm -rf frontend/node_modules
	rm -rf src/folionym.egg-info *.egg-info
	find src tests scripts -type d -name '__pycache__' -prune -exec rm -rf {} +
	find src tests scripts -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete

docs-screenshots:
	PYTHONHASHSEED=0 $(UV) run python scripts/capture_tui_screenshots.py

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
	PYTHON=$$($(UV) python find 3.14.6); \
	$(UV) venv --no-project --python "$$PYTHON" "$$ENV_DIR"; \
	$(UV) pip install --python "$$ENV_DIR/bin/python" "$${WHEEL}[pdf,tokens,ocr,tui,web]"; \
	"$$ENV_DIR/bin/python" scripts/verify_distributions.py dist --installed-wheel

release-check: frontend-check hygiene-check lint typecheck cov build-check

ci: release-check
