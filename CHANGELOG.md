# Changelog

All notable changes to the ROSE project will be documented in this file.


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **RHAPSODY backend integration**: Execution backends (`RadicalExecutionBackend`, `ConcurrentExecutionBackend`) are now imported from `rhapsody-py` (`from rhapsody.backends import ...`) instead of `radical.asyncflow`. `WorkflowEngine` remains in `radical.asyncflow`. Updated all examples, tutorials, docs, and notebooks accordingly.
- **Pre-commit hooks**: Added `.pre-commit-config.yaml` with docformatter, ruff, standard file checks, actionlint, GitHub workflow validation, and typos. The `examples/use_cases/` directory is excluded from linting.
- **CI pre-commit gate**: The `tests.yml` workflow now runs pre-commit as a required job before unit and integration tests, replacing the separate `lint` job.
- **New tutorials**: Added `03-highly-parallel-surrogates` and `04-al-algorithm-selector` tutorials with corresponding optional dependencies in `tutorials/pyproject.toml` and `tutorials/README.md`.

### Changed
- **Dependency update**: Added `rhapsody-py` in project dependencies; `radical.asyncflow` is retained for `WorkflowEngine`.
- **Package discovery**: Explicitly scoped setuptools to the `rose` package to prevent accidental inclusion of `tutorials/` and `examples/` in the distribution.
- **Ruff configuration**: Raised line length to 100 (aligned with docformatter), added ML naming convention rules to the ignore list (`N803`, `N806`, `N801`, `N812`–`N817`), and scoped the `B006` exception to example `run_me.py` files where `task_description={"shell": True}` is a required API pattern.
- **GitHub Actions**: Fixed unquoted `$GITHUB_ENV` shell variable in `tests.yml` and `ci.yml` (shellcheck SC2086).

---

## [Unreleased — previous]

### Added
- **New `start()` API**: Replaced the blocking `teach()` method with an asynchronous iterator `start()`. This allows users to instrument the loop, log metrics in real-time (e.g., to MLflow), and implement custom early stopping or adaptive logic.
- **IterationState**: Granular state reporting after each iteration, providing metrics, labeled/unlabeled counts, and statistics in a structured dataclass.
- **Dynamic Configuration**: Added ability to update learner configuration (batch sizes, task arguments, etc.) between iterations using `learner.set_next_config()`.
- **Added `Mlflow` integration: `rose.learner()` is now compatible with `mlflow` tracking. This feature is to support the need by the diffusion model community to track the training process via ROSE.

### Changed
- **Async-First Execution**: The core learner logic is now `asyncio` based, enabling better concurrency and integration with modern Python stacks.
- **Separation of Concerns**: Orchestration logic (ROSE) is more clearly separated from task execution (AsyncFlow).

### Deprecated
- `learner.teach()`: This method is deprecated and will be removed in a future version. Users should migrate to the `async for state in learner.start()` pattern.

---
