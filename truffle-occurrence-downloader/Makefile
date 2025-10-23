# Makefile for Truffle Occurrence Data Downloader

.PHONY: help install install-dev test lint format clean docs

help:  ## Show this help message
	@echo "Truffle Occurrence Data Downloader - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e .[dev,geospatial]

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=truffle_downloader --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 src/ tests/
	black --check src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Generate documentation
	@echo "Documentation is available in the README.md and examples/ directory"

install-test:  ## Test installation
	python test_installation.py

example-basic:  ## Run basic example
	python examples/basic_usage.py

example-advanced:  ## Run advanced example
	python examples/advanced_filtering.py

example-batch:  ## Run batch processing example
	python examples/batch_processing.py

cli-test:  ## Test CLI
	truffle-downloader --help
	truffle-downloader search --species "Tuber melanosporum" --limit 5

build:  ## Build package
	python setup.py sdist bdist_wheel

check-build:  ## Check built package
	twine check dist/*

all: clean install-dev test lint  ## Run all checks