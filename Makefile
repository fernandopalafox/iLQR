.PHONY: help install test test-bicycle test-unit clean

# Use venv by default
PYTHON := venv/bin/python
PIP := venv/bin/pip

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in venv"
	@echo "  make test          - Run unit test"
	@echo "  make test-all      - Run all tests and generate animations"
	@echo "  make test-parking  - Run parking test with animation"
	@echo "  make test-cartpole - Run cart-pole test with animation"
	@echo "  make test-rocket   - Run rocket test with animation"
	@echo "  make test-bicycle  - Run bicycle test with animation"
	@echo "  make test-unit     - Run unit test"
	@echo "  make clean         - Remove build files and cache"

install:
	$(PIP) install -e .

test: test-unit
	@echo "All tests completed!"
	@echo "Check figures/ directory for generated plots"

test-all: test-parking test-cartpole test-rocket test-bicycle test-unit
	@echo "All tests and animations completed!"
	@echo "Check figures/ directory for generated plots and animations"

test-bicycle:
	@echo "Running bicycle (unicycle) test..."
	$(PYTHON) tests/test_ilqr_bicycle.py

test-cartpole:
	@echo "Running cart-pole test..."
	$(PYTHON) tests/test_ilqr_cartpole.py

test-rocket:
	@echo "Running rocket test..."
	$(PYTHON) tests/test_ilqr_rocket.py

test-parking:
	@echo "Running parking test..."
	$(PYTHON) tests/test_ilqr_parking.py

test-unit:
	@echo "Running unit (quadratic) test..."
	$(PYTHON) tests/test_ilqr_unit.py

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned build files and cache"
