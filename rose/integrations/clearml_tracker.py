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


class ClearMLTracker:
    """TrackerBase implementation backed by ClearML.

    Logs pipeline manifest as hyperparameters on start, scalar metrics per
    iteration, and stop reason as a task tag on stop. Task outputs (returned
    as dicts) are captured automatically via ``on_iteration``.

    Args:
        project_name: ClearML project name (created if it does not exist).
        task_name: Display name for the ClearML task.
    """

    def __init__(self, project_name: str, task_name: str) -> None:
        try:
            import clearml  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "clearml is required for ClearMLTracker. Install it with: pip install rose[clearml]"
            ) from e

        self._project_name = project_name
        self._task_name = task_name
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

        series = state.learner_id or "value"

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
            _TASK_NAMES = (
                "simulation",
                "training",
                "prediction",
                "active_learn",
                "environment",
                "update",
                "criterion",
            )
            for task_name in _TASK_NAMES:
                task_cfg = state.current_config.get_task_config(task_name, state.iteration)
                if task_cfg is None:
                    continue
                for k, v in task_cfg.kwargs.items():
                    if isinstance(v, (int, float)) and not k.startswith("--"):
                        self._logger.report_scalar(
                            title=f"config/{task_name}/{k}",
                            series=series,
                            value=v,
                            iteration=state.iteration,
                        )

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        if self._task is None:
            return

        self._task.add_tags([f"stop:{reason}"])
        if final_state is not None:
            self._task.add_tags([f"final_iter:{final_state.iteration}"])
        self._task.close()
