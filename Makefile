PYTHON ?= python
PIP ?= $(PYTHON) -m pip
RUFF ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

.PHONY: install-dev lint format test cov clean hygiene-check release-check ci

install-dev:
	$(PIP) install -U pip
	$(PIP) install -e '.[dev,pdf]'

lint:
	$(RUFF) format --check .
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	$(PYTEST) -q

cov:
	$(PYTEST) --cov --cov-report=term-missing -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache
	rm -rf build dist
	rm -rf src/ai_pdf_renamer.egg-info *.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '.DS_Store' -delete

hygiene-check:
	@set -euo pipefail; \
	BAD=$$(git ls-files | grep -E '(^|/)\.DS_Store$$|(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache)(/|$$)|(^|/).*\.egg-info(/|$$)|(^|/)(build|dist)(/|$$)' || true); \
	if [ -n "$$BAD" ]; then \
		echo "Forbidden generated artifacts are tracked:"; \
		echo "$$BAD"; \
		exit 1; \
	fi

release-check: hygiene-check lint test

ci: release-check
