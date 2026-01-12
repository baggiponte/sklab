# AGENTS.md

## Project intro
Eksperiment is a lightweight experiment runner for Python ML (starting with scikit‑learn). The core idea is an `Experiment` class that accepts a sklearn `Pipeline`, an optional Optuna config dict, and a logger utility for structured experiment logging.

## How to collaborate (process)
- Iterate first: sketch interfaces, examples, and tradeoffs before writing a design doc.
- Prefer small, reviewable changes over big jumps.
- Keep plans/specs in `plans/` as markdown; promote to a design doc only after the API stabilizes.
- Plan specs in `plans/` must include sections: Goal, References, Design, How to test.
- If you defer items, add a `Future considerations` section (canonical place for scope‑outs).
- Ask a short clarification question only when decisions are blocking.
- Keep scope tight: focus on sklearn-first, optional optuna + wandb integrations.

## Style and conventions
- Prefer Python 3.11+ type hint syntax (`|`, `Self`, `list[str]`, etc.).
- Keep public APIs minimal and backend‑agnostic; avoid coupling core types to vendor SDKs.
- Favor explicit lifecycle methods and context‑managed runs for loggers.
- Use clear, short names; avoid over‑abstracting.
- Leave `__init__.py` files empty.
- Prefer integration tests; avoid monkey patching as much as possible.

## Logging design direction (current)
- Logger protocol should return a context‑managed run handle.
- Minimal run API: `log_params`, `log_metrics`, `set_tags`, `log_artifact`, `log_model`, `finish`.
- Keep adapter examples minimal; avoid deep backend behavior in the core API.

## Future considerations (canonical)
- Use this section in plans to capture deferred or potentially complex items (e.g., run IDs, backend-specific behaviors).

## Tooling expectations
- Use `uv` for all Python env and package operations; do not use the `uv pip` interface.
- Example commands:
  - Run: `uv run python -m module` or `uv run script.py`
  - Add deps: `uv add <pkg>` or `uv add --dev <pkg>`
  - Add optional deps: `uv add --optional <extra> <pkg>`
- Ruff for linting (see `pyproject.toml`); target modern syntax.
- `prek` for git hooks (see `.pre-commit-config.yaml`).

## Keep this file updated
- Update the project intro and process notes as the scope evolves.
