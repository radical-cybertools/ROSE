from __future__ import annotations

from rose.learner import Learner

from .adapters import TaskAdapterFactory
from .schema import SpecConfig, TrackingConfig, _REQUIRED_SLOTS


def _get_learner_class(learner_type: str):
    if learner_type == "sequential_active_learner":
        from rose.al.active_learner import SequentialActiveLearner

        return SequentialActiveLearner
    if learner_type == "parallel_active_learner":
        from rose.al.active_learner import ParallelActiveLearner

        return ParallelActiveLearner
    if learner_type == "sequential_reinforcement_learner":
        from rose.rl.reinforcement_learner import SequentialReinforcementLearner

        return SequentialReinforcementLearner
    if learner_type == "uq_active_learner":
        from rose.uq.uq_active_learner import SeqUQLearner

        return SeqUQLearner
    raise ValueError(f"No learner class registered for type '{learner_type}'")


class LearnerBuilder:
    def __init__(self, config: SpecConfig, asyncflow) -> None:
        self.config = config
        self.asyncflow = asyncflow

    def build(self) -> Learner:
        cfg = self.config
        learner = _get_learner_class(cfg.learner.type)(self.asyncflow)

        if cfg.candidates is not None:
            self._register_dispatched_tasks(learner, cfg)
        else:
            self._register_flat_tasks(learner, cfg)

        crit = cfg.stop_criterion
        c_closure = TaskAdapterFactory.make_closure(crit.evaluator, cfg.remote)
        c_as_exec = TaskAdapterFactory.as_executable(crit.evaluator)
        learner.as_stop_criterion(
            metric_name=crit.metric,
            threshold=crit.threshold,
            operator=crit.operator,
            as_executable=c_as_exec,
        )(c_closure)

        _attach_tracker(learner, cfg.tracking)
        return learner

    def _register_flat_tasks(self, learner: Learner, cfg: SpecConfig) -> None:
        for slot_name, task_def in cfg.tasks.items():
            closure = TaskAdapterFactory.make_closure(task_def, cfg.remote)
            as_exec = TaskAdapterFactory.as_executable(task_def)
            getattr(learner, f"{slot_name}_task")(as_executable=as_exec)(closure)

    def _register_dispatched_tasks(self, learner: Learner, cfg: SpecConfig) -> None:
        required = _REQUIRED_SLOTS[cfg.learner.type]
        for slot_name in required:
            task_defs = [c.tasks[slot_name] for c in cfg.candidates]
            as_exec = task_defs[0].type == "shell"
            closure = TaskAdapterFactory.make_dispatch_closure(slot_name, task_defs, cfg.remote)
            getattr(learner, f"{slot_name}_task")(as_executable=as_exec)(closure)

    def build_learner_configs(self):
        """Return auto-generated LearnerConfig list for parallel candidates, or None."""
        cfg = self.config
        if cfg.candidates is None:
            return None
        from rose.learner import LearnerConfig, TaskConfig

        required = _REQUIRED_SLOTS[cfg.learner.type]
        configs = []
        max_iter = cfg.learner.max_iter
        for i, _candidate in enumerate(cfg.candidates):
            schedule = {n: TaskConfig(kwargs={"learner_id": i}) for n in range(max_iter + 1)}
            schedule[-1] = TaskConfig(kwargs={"learner_id": i})
            configs.append(
                LearnerConfig(**{slot: schedule for slot in required}, criterion=schedule)
            )
        return configs


def _attach_tracker(learner: Learner, tracking: TrackingConfig) -> None:
    if tracking.backend == "mlflow":
        from rose.integrations.mlflow_tracker import MLflowTracker

        learner.add_tracker(
            MLflowTracker(
                experiment_name=tracking.experiment,
                run_name=tracking.run_name,
            )
        )
    elif tracking.backend == "clearml":
        from rose.integrations.clearml_tracker import ClearMLTracker

        learner.add_tracker(
            ClearMLTracker(
                project_name=tracking.experiment,
                task_name=tracking.run_name or "rose-spec-run",
            )
        )
