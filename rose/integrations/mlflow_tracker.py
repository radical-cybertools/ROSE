"""MLflow tracker integration for ROSE.

Install the optional dependency before use::

    pip install rose[mlflow]

Usage::

    from rose.integrations.mlflow_tracker import MLflowTracker

    learner.add_tracker(MLflowTracker(experiment_name="surrogate-v1"))
    async for state in learner.start(max_iter=20):
        ...  # tracking happens automatically
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rose.learner import IterationState
    from rose.tracking import PipelineManifest


class MLflowTracker:
    """TrackerBase implementation backed by MLflow.

    Logs pipeline manifest as run parameters on start, metric values per
    iteration, and stop reason as a tag on stop. Task outputs (returned as
    dicts) are captured automatically via ``on_iteration``.

    Args:
        experiment_name: MLflow experiment name (created if it does not exist).
        run_name: Optional display name for the run.
    """

    def __init__(self, experiment_name: str, run_name: str | None = None) -> None:
        try:
            import mlflow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mlflow is required for MLflowTracker. Install it with: pip install rose[mlflow]"
            ) from e

        self._experiment_name = experiment_name
        self._run_name = run_name
        self._run = None

    def on_start(self, manifest: PipelineManifest) -> None:
        import mlflow

        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name)

        params: dict[str, Any] = {
            "learner_type": manifest.learner_type,
            "parallel_learners": manifest.parallel_count,
            "criterion/metric_name": manifest.criterion.metric_name if manifest.criterion else None,
            "criterion/threshold": manifest.criterion.threshold if manifest.criterion else None,
            "criterion/operator": manifest.criterion.operator if manifest.criterion else None,
        }
        for task_key, task_manifest in manifest.tasks.items():
            params[f"task.{task_key}.as_executable"] = task_manifest.as_executable
            for k, v in task_manifest.log_params.items():
                params[f"task.{task_key}.{k}"] = v

        mlflow.log_params(params)

    def on_iteration(self, state: IterationState) -> None:
        import mlflow

        step = state.iteration
        prefix = f"{state.learner_id}/" if state.learner_id is not None else ""

        if state.metric_value is not None:
            metric_name = getattr(state, "metric_name", None) or "metric"
            mlflow.log_metric(f"{prefix}{metric_name}", state.metric_value, step=step)

        for key, value in state.state.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}{key}", value, step=step)

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        import mlflow

        if self._run is None:
            return

        mlflow.set_tag("stop_reason", reason)
        if final_state is not None:
            mlflow.set_tag("final_iteration", str(final_state.iteration))
        mlflow.end_run()
