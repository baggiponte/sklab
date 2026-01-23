set quiet := true

format:
    uvx ruff format -- src
    uvx ruff check --fix --select=I,UP,F401 -- src

lint: format
    uvx ruff check --fix -- src
    uvx ty check

test:
    uv run --all-extras -- pytest tests/

test-docs:
    uv run --all-extras -- pytest docs/

test-all: test test-docs

docs:
    uv run zensical serve
