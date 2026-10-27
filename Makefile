.PHONY: install verify process test lint format clean

VENV ?= .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
RUFF = $(VENV)/bin/ruff
PYTEST = $(VENV)/bin/pytest

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

verify:
	$(PYTHON) -m src.cli verify

process:
	$(PYTHON) -m src.cli process

test:
	$(PYTEST)

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ __pycache__/ data/*.parquet
