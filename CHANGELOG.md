# Changelog

All notable changes to the ROSE project will be documented in this file.

## [Unreleased]

### Added
- **New `start()` API**: Replaced the blocking `teach()` method with an asynchronous iterator `start()`. This allows users to instrument the loop, log metrics in real-time (e.g., to MLflow), and implement custom early stopping or adaptive logic.
- **AsyncFlow Integration**: Deep integration with `radical.asyncflow` for flexible task execution across various backends (Concurrent, Slurm, etc.).
- **IterationState**: Granular state reporting after each iteration, providing metrics, labeled/unlabeled counts, and statistics in a structured dataclass.
- **Dynamic Configuration**: Added ability to update learner configuration (batch sizes, task arguments, etc.) between iterations using `learner.set_next_config()`.
- **Decorator-based Task Registration**: Streamlined API for registering simulation, training, and active learning tasks.

### Changed
- **Async-First Execution**: The core learner logic is now `asyncio` based, enabling better concurrency and integration with modern Python stacks.
- **Separation of Concerns**: Orchestration logic (ROSE) is more clearly separated from task execution (AsyncFlow).

### Deprecated
- `learner.teach()`: This method is deprecated and will be removed in a future version. Users should migrate to the `async for state in learner.start()` pattern.

---
*Generated on 2026-01-30*
