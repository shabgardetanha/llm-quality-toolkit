.PHONY: install test lint clean build publish

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*