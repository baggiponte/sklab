---
title: Rename distribution to scikit-lab
description: Publish package as scikit-lab while keeping import path as sklab
date: 2026-02-07
---

# Rename distribution to scikit-lab

## Goal
Ship version `0.0.1` with PyPI distribution name `scikit-lab`, while preserving Python imports as `sklab`.

## References
- `pyproject.toml`
- `uv.lock`
- `.github/workflows/publish.yml`
- `README.md`
- `docs/index.md`

## Design
- Keep source package directory as `src/sklab` and all imports unchanged.
- Rename distribution metadata from `sklab` to `scikit-lab`.
- Update self-references in dependency groups/lock metadata to use `scikit-lab[...]`.
- Update publish workflow PyPI project URL to `https://pypi.org/project/scikit-lab/`.
- Update install snippets in docs and README to use `scikit-lab`.
- Keep release version at `0.0.1`.

## How to test
- Run `uv run pytest --ignore=docs` to validate runtime tests.
- Run `uv run pytest docs` to validate documentation code fences.
- Build package with `uv build --no-sources` and verify artifact name starts with `scikit_lab-0.0.1`.

## Future considerations
- Update repository name/links from `sklab` to `scikit-lab` if/when repo rename happens.
- Add a short note in docs clarifying "install as scikit-lab, import as sklab."
