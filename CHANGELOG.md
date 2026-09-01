# Changelog

All notable changes to the ROSE project will be documented in this file.


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.3.0] - 2026-09-01

### Added
- **`IterationState.learner_id`**: New field (`int | str | None`, default `None`) on `IterationState`
  identifying which parallel learner produced a given state. Integer index for
  `ParallelActiveLearner` and `ParallelReinforcementLearner`; learner name string for
  `ParallelUQLearner`.
- **`rose run --remote`**: spec-driven remote execution against an ORBIT broker, driven
  entirely by a `remote:` block in the workflow YAML (`remote.target`, `remote.broker_url`) —
  no separate manual endpoint-bootstrap step required.
- **`remote.embedded`**: host the ORBIT broker in-process instead of requiring a standalone
  broker deployment.
- **`rose setup`**: interactive first-time wizard that provisions broker TLS cert/token and
  IRI/SFAPI credentials, then verifies everything end-to-end by bringing up a real ORBIT
  endpoint — ending with a ready-to-paste `remote:` block on success.
- **Configurable HPC wait timeout**: `rose setup --wait-timeout SECONDS` and the spec's
  `remote.target.endpoint_timeout_min` control how long ROSE waits on HPC job/endpoint
  registration; a warning is printed with the default used when neither is set.
- **CLI reference docs**: a new Command-Line Interface page documenting `rose run` and
  `rose setup` and their flags.
- **Two canonical example specs** under `examples/spec/` — `sequential_python` (Python
  tasks, sequential learner) and `parallel_shell_learners` (shell tasks, parallel
  learners) — each with a local (`workflow.yaml`) and a `--remote` (`workflow-remote.yaml`)
  variant, together covering Python/shell, local/remote, and sequential/parallel.

### Changed
- **Unified async-iterator API for parallel learners**: `ParallelActiveLearner.start()`,
  `ParallelReinforcementLearner.start()`, and `ParallelUQLearner.start()` now return
  `AsyncIterator[IterationState]` instead of blocking until all learners finish and returning
  `list[Any]`. States stream in real time as each parallel learner completes an iteration,
  using the same `async for state in learner.start():` interface as `SequentialActiveLearner`.
- **Shared `_stream_parallel` helper**: The internal `asyncio.Queue`-based fan-in pattern
  is extracted into a single module-level async generator in `rose/learner.py`, eliminating
  identical code that was previously duplicated across all three parallel learner classes.
- **`rose/bootstrap.py` and `rose/remote.py` reorganized into a `rose/remote/` package**
  (`rose/remote/bootstrap.py`, `rose/remote/execute.py`), matching the existing
  `active_learning`/`reinforcement_learning`/`spec` subpackage convention.
  `from rose.remote import run_remote, run_setup_wizard` continues to work unchanged via
  the package's re-export.
- **Default `--local` execution backend** switched from the deprecated
  `DragonExecutionBackendV3`/`dragon_v3` to `DragonExecutionBackend`/`dragon` (rhapsody
  deprecated the former). Also fixed `--backend`'s help text, which incorrectly stated the
  default was `concurrent`.
- **PyPI distribution renamed** from `ROSE` to `rose-surrogate-explorer` (the `ROSE` name was
  already taken on PyPI by an unrelated package). `import rose` and the `rose` CLI command
  are unaffected. Added `license`/`classifiers` metadata to `pyproject.toml` for the first
  PyPI release.
- **Documentation site restyled** to match RHAPSODY's mkdocs-material conventions (flat
  header/tabs, forced heading color, pill buttons, `pymdownx.highlight`) while keeping
  ROSE's rose/pink brand palette.

### Fixed
- `typeguard` (used directly by `rose/learner.py`'s `@typeguard.typechecked`) is now an
  explicit dependency instead of only being available transitively through
  `radical.asyncflow`/`rhapsody-py`.
- Several broken internal documentation anchor links in `spec-api.md`/`rose-aas.md`
  (double-hyphen slugs that never matched mkdocs' actual generated anchors).
- A broken inline `<figure>` positioning hack on the docs homepage.

### Deprecated
- **`ParallelActiveLearner.teach()`**, **`ParallelReinforcementLearner.learn()`**, and
  **`ParallelUQLearner.teach()`** still work but now internally iterate `start()` and
  collect final states into a list. Migrate to `async for state in learner.start():`.

---

## [0.2.0] - 2026-02-27

### Added
- **RHAPSODY backend integration**: Execution backends (`RadicalExecutionBackend`, `ConcurrentExecutionBackend`) are now imported from `rhapsody-py` (`from rhapsody.backends import ...`) instead of `radical.asyncflow`. `WorkflowEngine` remains in `radical.asyncflow`. Updated all examples, tutorials, docs, and notebooks accordingly.
- **Pre-commit hooks**: Added `.pre-commit-config.yaml` with docformatter, ruff, standard file checks, actionlint, GitHub workflow validation, and typos. The `examples/use_cases/` directory is excluded from linting.
- **CI pre-commit gate**: The `tests.yml` workflow now runs pre-commit as a required job before unit and integration tests, replacing the separate `lint` job.
- **New tutorials**: Added `03-highly-parallel-surrogates` and `04-al-algorithm-selector` tutorials with corresponding optional dependencies in `tutorials/pyproject.toml` and `tutorials/README.md`.
- **New `start()` API**: Replaced the blocking `teach()` method with an asynchronous iterator `start()`. This allows users to instrument the loop, log metrics in real-time (e.g., to MLflow), and implement custom early stopping or adaptive logic.
- **IterationState**: Granular state reporting after each iteration, providing metrics, labeled/unlabeled counts, and statistics in a structured dataclass.
- **Dynamic Configuration**: Added ability to update learner configuration (batch sizes, task arguments, etc.) between iterations using `learner.set_next_config()`.
- **MLflow integration**: `rose.learner()` is now compatible with MLflow tracking to support the diffusion model community's need to monitor the training process via ROSE.

### Changed
- **Dependency update**: `rhapsody-py[radical_pilot]` added as a core dependency; Dragon HPC backend (`rhapsody-py[dragon]`) auto-installed on Python ≤3.12 via PEP 508 environment marker. `radical.asyncflow` retained for `WorkflowEngine`.
- **Python support**: Minimum Python version is 3.10; Python 3.9 dropped from all tooling, CI, and tox environments.
- **Async-first execution**: The core learner logic is now `asyncio`-based, enabling better concurrency and integration with modern Python stacks.
- **Separation of concerns**: Orchestration logic (ROSE) is more clearly separated from task execution (AsyncFlow/RHAPSODY).
- **Package discovery**: Explicitly scoped setuptools to the `rose` package to prevent accidental inclusion of `tutorials/` and `examples/` in the distribution.
- **Ruff configuration**: Raised line length to 100, added ML naming convention rules to the ignore list (`N803`, `N806`, `N801`, `N812`–`N817`), and scoped the `B006` exception to example `run_me.py` files where `task_description={"shell": True}` is a required API pattern.
- **GitHub Actions**: Fixed unquoted `$GITHUB_ENV` shell variable in `tests.yml` and `ci.yml` (shellcheck SC2086).

### Deprecated
- `learner.teach()`: This method is deprecated and will be removed in a future version. Users should migrate to the `async for state in learner.start()` pattern.

---
