from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Coroutine

from .builder import LearnerBuilder
from .schema import SpecConfig

__all__ = ["load_spec", "WorkflowSpec"]


def load_spec(path: str | Path) -> "WorkflowSpec":
    """Load and validate a YAML workflow spec. Raises ValueError on schema errors."""
    return WorkflowSpec(SpecConfig.from_yaml(path))


class WorkflowSpec:
    """Validated workflow spec that produces a coroutine compatible with service_utils.run()."""

    def __init__(self, config: SpecConfig) -> None:
        self.config = config

    @property
    def workflow(self) -> Callable[..., Coroutine[Any, Any, None]]:
        cfg = self.config

        async def _workflow(bridge_url: str, edge_name: str) -> None:
            import rhapsody
            from radical.asyncflow import WorkflowEngine

            engine = await rhapsody.get_backend(
                "edge", bridge_url=bridge_url, edge_name=edge_name
            )
            asyncflow = await WorkflowEngine.create(engine)

            builder = LearnerBuilder(cfg, asyncflow)
            learner = builder.build()

            start_kwargs: dict[str, Any] = {"max_iter": cfg.learner.max_iter}
            if cfg.learner.type == "parallel_active_learner":
                lc = builder.build_learner_configs()
                if lc is not None:
                    start_kwargs["parallel_learners"] = len(lc)
                    start_kwargs["learner_configs"] = lc
                else:
                    start_kwargs["parallel_learners"] = cfg.learner.parallel_learners

            try:
                async for state in learner.start(**start_kwargs):
                    print(
                        f"[iter {state.iteration}]  metric={state.metric_value}",
                        flush=True,
                    )
            finally:
                await asyncflow.shutdown()

        return _workflow
