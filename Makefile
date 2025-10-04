.PHONY: help install install-dev test test-bicycle test-unit plots clean

# Use venv by default
PYTHON := venv/bin/python
PIP := venv/bin/pip

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in venv"
	@echo "  make install-dev  - Install package with dev dependencies in venv"
	@echo "  make test         - Run all tests and generate plots"
	@echo "  make test-bicycle - Run bicycle (unicycle) test and generate plot"
	@echo "  make test-unit    - Run unit test (quadratic) and generate plot"
	@echo "  make plots        - Generate all plots from tests"
	@echo "  make clean        - Remove generated files and cache"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test: test-bicycle test-unit
	@echo "All tests completed!"
	@echo "Check figures/ directory for generated plots"

test-bicycle:
	@echo "Running bicycle (unicycle) test..."
	$(PYTHON) tests/test_ilqr_bicycle.py

test-unit:
	@echo "Running unit (quadratic) test..."
	$(PYTHON) tests/test_ilqr_unit.py

plots: test

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned build files and cache"
