"""ClearML tracker integration for ROSE.

Install the optional dependency before use::

    pip install rose[clearml]

Usage::

    from rose.integrations.clearml_tracker import ClearMLTracker

    learner.add_tracker(ClearMLTracker(project_name="ROSE", task_name="al-run-01"))
    async for state in learner.start(max_iter=20):
        ...  # tracking happens automatically
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rose.learner import IterationState
    from rose.tracking import PipelineManifest

_TASK_NAMES = (
    "simulation",
    "training",
    "prediction",
    "active_learn",
    "environment",
    "update",
    "criterion",
)


class ClearMLTracker:
    """TrackerBase implementation backed by ClearML.

    Logs pipeline manifest as hyperparameters on start, scalar metrics per
    iteration, and stop reason as a task tag on stop. Task outputs (returned
    as dicts) are captured automatically via ``on_iteration``.

    Args:
        project_name: ClearML project name (created if it does not exist).
        task_name: Display name for the ClearML task.
        learner_names: Optional list of human-readable names for parallel
            learners. When provided, integer ``learner_id`` values (0, 1, ...)
            are mapped to these names as the ClearML scalar series, so the UI
            shows ``"ensemble-A"`` instead of ``0``.
    """

    def __init__(
        self,
        project_name: str,
        task_name: str,
        learner_names: list[str] | None = None,
    ) -> None:
        try:
            import clearml  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "clearml is required for ClearMLTracker. Install it with: pip install rose[clearml]"
            ) from e

        self._project_name = project_name
        self._task_name = task_name
        self._learner_names: list[str] = learner_names or []
        self._task = None
        self._logger = None

    def on_start(self, manifest: PipelineManifest) -> None:
        from clearml import Task

        self._task = Task.init(
            project_name=self._project_name,
            task_name=self._task_name,
        )
        self._logger = self._task.get_logger()

        params: dict[str, Any] = {
            "learner_type": manifest.learner_type,
            "parallel_learners": manifest.parallel_count,
            "criterion/metric_name": manifest.criterion.metric_name if manifest.criterion else None,
            "criterion/threshold": manifest.criterion.threshold if manifest.criterion else None,
            "criterion/operator": manifest.criterion.operator if manifest.criterion else None,
        }
        for task_key, task_manifest in manifest.tasks.items():
            params[f"task/{task_key}/as_executable"] = task_manifest.as_executable
            for k, v in task_manifest.log_params.items():
                params[f"task/{task_key}/{k}"] = v

        self._task.connect(params)

    def on_iteration(self, state: IterationState) -> None:
        if self._logger is None:
            return

        learner_id = state.learner_id
        if learner_id is None:
            series: str = "value"
        elif (
            self._learner_names
            and isinstance(learner_id, int)
            and learner_id < len(self._learner_names)
        ):
            series = self._learner_names[learner_id]
        else:
            series = str(learner_id)

        if state.metric_value is not None:
            metric_name = getattr(state, "metric_name", None) or "metric"
            self._logger.report_scalar(
                title=metric_name,
                series=series,
                value=state.metric_value,
                iteration=state.iteration,
            )

        for key, value in state.state.items():
            if isinstance(value, (int, float)):
                self._logger.report_scalar(
                    title=key,
                    series=series,
                    value=value,
                    iteration=state.iteration,
                )

        if state.current_config is not None:
            string_params: dict[str, str] = {}
            for task_name in _TASK_NAMES:
                task_cfg = state.current_config.get_task_config(task_name, state.iteration)
                if task_cfg is None:
                    continue
                for k, v in task_cfg.kwargs.items():
                    if k.startswith("--"):
                        continue
                    if isinstance(v, (int, float)):
                        self._logger.report_scalar(
                            title=f"config/{task_name}/{k}",
                            series=series,
                            value=v,
                            iteration=state.iteration,
                        )
                    elif isinstance(v, str):
                        string_params[f"{task_name}/{k}"] = v
            if string_params and self._task is not None:
                self._task.connect(string_params, name="current_config")

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        if self._task is None:
            return

        self._task.add_tags([f"stop:{reason}"])
        if final_state is not None:
            self._task.add_tags([f"final_iter:{final_state.iteration}"])
        self._task.close()
