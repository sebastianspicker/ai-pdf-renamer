UV ?= uv

.PHONY: install-dev lint format typecheck test cov clean hygiene-check release-check ci

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

cov:
	$(UV) run pytest --cov=ai_pdf_renamer --cov-report=term-missing --cov-fail-under=85 -q

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

release-check: hygiene-check lint typecheck cov

ci: release-check
