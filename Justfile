format:
    uvx ruff format -- src
    uvx ruff check --fix --select=I,UP -- src

lint:
    uvx ruff check -- src
    uvx ruff format --check -- src
    uvx ty check
