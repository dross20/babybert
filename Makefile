.PHONY: style, check, install, test

check:
	ruff check
	ruff format --diff

style:
	ruff check --fix
	ruff format

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -v -s

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache/

all: style test