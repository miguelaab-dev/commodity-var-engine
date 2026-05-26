.PHONY: install dev test lint type-check run backtest clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install 2>/dev/null || true

test:
	pytest tests/ -v --cov=varengine --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --cov=varengine

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/varengine/

run:
	varengine run --config configs/portfolio.yaml

backtest:
	varengine backtest --config configs/portfolio.yaml --window 500

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
