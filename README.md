![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-2f6feb)
![Ruff Format](https://img.shields.io/badge/format-ruff-2f6feb)
![Ty](https://img.shields.io/badge/type%20check-ty-6e56cf)

# Eksperiment

A lightweight experiment runner for Python ML (starting with scikit‑learn). The core idea is an `Experiment` class that accepts a sklearn `Pipeline`, an optional Optuna config dict, and a logger utility for structured experiment logging.

## Goals
- Keep the API small, explicit, and backend‑agnostic.
- Provide first‑class support for context‑managed logger runs.
- Start sklearn‑first, with optional integrations (Optuna, W&B).
