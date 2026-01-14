format:
    uvx ruff format -- src
    uvx ruff check --fix --select=I,UP -- src

lint: format
    uvx ruff check -- src
    uvx ty check

test:
    uv run --all-extras -- pytest

docs:
    uv run zensical serve
