.PHONY: help clean install test test-cov test-pkg test-contract test-artifacts test-e2e test-live build-lookup lint format type-check docs docs-serve build upload dev-install

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

install: ## Install package
	uv pip install -e .

dev-install: ## Install package with development dependencies
	uv sync --dev
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=indicate --cov-report=html --cov-report=term

test-pkg: ## Test only the shipped package (skip gazetteer/, which does not ship)
	pytest tests --ignore=tests/gazetteer

test-contract: ## What must pass on a fresh clone with no network
	HF_HUB_OFFLINE=1 pytest tests --ignore=tests/gazetteer

test-artifacts: ## Fail instead of skipping when weights or tables are missing
	pytest tests --require-artifacts

test-e2e: ## Build the wheel, install it, run the console script
	pytest -m "e2e and not live"

test-live: ## Reach Hugging Face and check the model repo has what we claim
	pytest -m live

build-lookup: ## Build both lookup tables from the committed corpora
	uv run --group train python training/build_lookup.py --lang hindi
	uv run --group train python training/build_lookup.py --lang punjabi

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff format .
	ruff check --fix .

type-check: ## Run type checker
	pyright

docs: ## Build documentation
	cd docs && make clean && make html

docs-serve: ## Serve documentation locally
	cd docs/build/html && python -m http.server 8000

build: ## Build package
	uv build

upload: ## Upload to PyPI
	uv publish

ci: lint type-check test ## Run CI checks

# Legacy aliases for compatibility
install-dev: dev-install ## Legacy alias for dev-install
typecheck: type-check ## Legacy alias for type-check