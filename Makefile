POETRY ?= $(HOME)/.poetry/bin/poetry

.PHONY: install-poetry
install-poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

.PHONY: install-packages
install-packages:
	$(POETRY) install

.PHONY: install
install: install-poetry install-packages

.PHONY: fmt
fmt:
	$(POETRY) run isort .
	$(POETRY) run black .

.PHONY: test
test:
	$(POETRY) run pytest

