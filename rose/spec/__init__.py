from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from .builder import LearnerBuilder
from .schema import WorkflowConfig

__all__ = ["load_spec", "WorkflowSpec"]


def load_spec(path: str | Path, validate_imports: bool = False) -> WorkflowSpec:
    """Load and validate a YAML workflow spec. Raises ValueError on schema errors.

    Args:
        path: Path to the YAML file.
        validate_imports: If True, verify that every python task's ``module:callable``
            is importable from the current environment (sys.path extended with
            ``remote.pythonpath``). Use during development when task files are
            locally accessible. Leave False (default) when ``remote.pythonpath``
            points to paths that only exist on the remote worker.
    """
    spec = WorkflowSpec(WorkflowConfig.from_yaml(path))
    if validate_imports:
        _validate_task_imports(spec.config)
    return spec


def _collect_python_specs(cfg: WorkflowConfig) -> list[str]:
    """Return deduplicated 'module:callable' strings for all Python tasks in the spec."""
    specs: list[str] = []
    if cfg.learners is not None:
        for ld in cfg.learners:
            for td in ld.tasks.values():
                if td.type == "python":
                    specs.append(td.function)
    else:
        for td in cfg.tasks.values():
            if td.type == "python":
                specs.append(td.function)
    if cfg.stop_criterion.evaluator.type == "python":
        specs.append(cfg.stop_criterion.evaluator.function)
    return list(dict.fromkeys(specs))  # deduplicate, preserve order


def _validate_task_imports(cfg: WorkflowConfig) -> None:
    """Try to import every Python task callable.

    Raises ValueError listing all failures.
    """
    import importlib
    import sys as _sys

    added = [p for p in cfg.remote.pythonpath if p not in _sys.path]
    for p in added:
        _sys.path.insert(0, p)
    try:
        failures: list[str] = []
        for spec_str in _collect_python_specs(cfg):
            mod_path, fn_name = spec_str.rsplit(":", 1)
            try:
                mod = importlib.import_module(mod_path)
                if not hasattr(mod, fn_name):
                    failures.append(f"  {spec_str!r}: module has no attribute '{fn_name}'")
            except ImportError as exc:
                failures.append(f"  {spec_str!r}: {exc}")
        if failures:
            raise ValueError(
                "validate_imports failed — the following function specs are not importable:\n"
                + "\n".join(failures)
            )
    finally:
        for p in added:
            if p in _sys.path:
                _sys.path.remove(p)


class WorkflowSpec:
    """Validated workflow spec that produces a coroutine compatible with service_utils.run()."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

    def workflow_with(self, **overrides: Any) -> WorkflowSpec:
        """Return a new WorkflowSpec with selective overrides applied.

        Accepted keys:
        - ``parameters``: dict merged (not replaced) into the existing parameters block
        - Any ``LearnerSpec`` field: ``max_iter``, ``parallel_learners``
        - Any other top-level ``WorkflowConfig`` field

        Example::

            spec.workflow_with(max_iter=3, parameters={"dataset": "test_ds"})
        """
        data = self.config.model_dump()
        learner_fields = set(data["learner"].keys())
        for key, value in overrides.items():
            if key == "parameters" and isinstance(value, dict):
                data["parameters"] = {**data["parameters"], **value}
            elif key in learner_fields:
                data["learner"][key] = value
            elif key in data:
                data[key] = value
            else:
                raise ValueError(f"workflow_with: unknown spec field '{key}'")
        return WorkflowSpec(WorkflowConfig.model_validate(data))

    @property
    def workflow(self) -> Callable[..., Coroutine[Any, Any, None]]:
        cfg = self.config

        async def _workflow(bridge_url: str, edge_name: str) -> None:
            import rhapsody
            from radical.asyncflow import WorkflowEngine

            engine = await rhapsody.get_backend("edge", bridge_url=bridge_url, edge_name=edge_name)
            asyncflow = await WorkflowEngine.create(engine)

            builder = LearnerBuilder(cfg, asyncflow)
            learner = builder.build()

            start_kwargs: dict[str, Any] = {"max_iter": cfg.learner.max_iter}
            if cfg.learner.type == "parallel_active_learner":
                lcs = builder.build_learner_configs()
                if lcs is not None:
                    start_kwargs["parallel_learners"] = len(lcs)
                    start_kwargs["learner_configs"] = lcs
                else:
                    start_kwargs["parallel_learners"] = cfg.learner.parallel_learners
            else:
                # Sequential learners accept initial_config — the same ROSE-native
                # mechanism used by the parallel path, giving tasks access to
                # parameters and iteration via kwargs at every iteration.
                ic = builder.build_learner_config()
                if ic is not None:
                    start_kwargs["initial_config"] = ic

            try:
                async for state in learner.start(**start_kwargs):
                    print(
                        f"[iter {state.iteration}]  metric={state.metric_value}",
                        flush=True,
                    )
            finally:
                await asyncflow.shutdown()

        return _workflow
