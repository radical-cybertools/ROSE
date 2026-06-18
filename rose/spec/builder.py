from __future__ import annotations

from pathlib import Path

from rose.learner import Learner

from .adapters import TaskAdapterFactory
from .schema import WorkflowConfig, TrackingConfig, _REQUIRED_SLOTS

# Slots that exist as fields on LearnerConfig; uncertainty is required by uq_active_learner
# but is not a LearnerConfig field — filter it out when constructing LearnerConfig.
_LEARNER_CONFIG_SLOTS: frozenset[str] = frozenset(
    {"simulation", "training", "prediction", "active_learn", "environment", "update"}
)


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
    def __init__(self, source: str | Path | WorkflowConfig, asyncflow) -> None:
        if isinstance(source, (str, Path)):
            source = WorkflowConfig.from_yaml(source)
        elif hasattr(source, "config"):  # duck-typed: also accepts a WorkflowSpec
            source = source.config
        self.config = source
        self.asyncflow = asyncflow

    def build(self) -> Learner:
        cfg = self.config
        learner = _get_learner_class(cfg.learner.type)(self.asyncflow)

        if cfg.learners is not None:
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

    def _register_flat_tasks(self, learner: Learner, cfg: WorkflowConfig) -> None:
        for slot_name, task_def in cfg.tasks.items():
            closure = TaskAdapterFactory.make_closure(task_def, cfg.remote)
            as_exec = TaskAdapterFactory.as_executable(task_def)
            getattr(learner, f"{slot_name}_task")(as_executable=as_exec)(closure)

    def _register_dispatched_tasks(self, learner: Learner, cfg: WorkflowConfig) -> None:
        required = _REQUIRED_SLOTS[cfg.learner.type]
        for slot_name in required:
            task_defs = [l.tasks[slot_name] for l in cfg.learners]
            as_exec = task_defs[0].type == "shell"
            closure = TaskAdapterFactory.make_dispatch_closure(slot_name, task_defs, cfg.remote)
            getattr(learner, f"{slot_name}_task")(as_executable=as_exec)(closure)

    def build_learner_config(self):
        """Single LearnerConfig for sequential learners.

        Passed as initial_config to learner.start() so that parameters and
        iteration reach task kwargs via the same ROSE-native mechanism used
        by the parallel path.  Returns None when no parameters are defined.
        """
        cfg = self.config
        if not cfg.parameters and not cfg.remote.pythonpath:
            return None
        from rose.learner import LearnerConfig, TaskConfig

        required = _REQUIRED_SLOTS[cfg.learner.type]
        max_iter = cfg.learner.max_iter
        params   = dict(cfg.parameters)
        params["pythonpath"] = list(cfg.remote.pythonpath)
        schedule = {n: TaskConfig(kwargs={**params, "iteration": n}) for n in range(max_iter + 1)}
        schedule[-1] = TaskConfig(kwargs={**params, "iteration": max_iter})
        lc_slots = required & _LEARNER_CONFIG_SLOTS
        return LearnerConfig(**{slot: schedule for slot in lc_slots}, criterion=schedule)

    def build_learner_configs(self):
        """LearnerConfig list for parallel learners, or None if no learners defined."""
        cfg = self.config
        if cfg.learners is None:
            return None
        from rose.learner import LearnerConfig, TaskConfig

        required = _REQUIRED_SLOTS[cfg.learner.type]
        configs  = []
        max_iter = cfg.learner.max_iter
        params   = dict(cfg.parameters)
        params["pythonpath"] = list(cfg.remote.pythonpath)
        lc_slots = required & _LEARNER_CONFIG_SLOTS
        for i, learner_def in enumerate(cfg.learners):
            # learner_id    → dispatch routing key (popped by closure, never reaches user fn)
            # iteration     → per-entry so get_task_config(slot, n) returns the right value
            # learner_label → human-readable learner name; only injected when non-empty
            # parameters    → user-defined values from the YAML parameters: block
            base = {"learner_id": i, **params}
            if learner_def.label:
                base["learner_label"] = learner_def.label
            schedule = {n: TaskConfig(kwargs={**base, "iteration": n}) for n in range(max_iter + 1)}
            schedule[-1] = TaskConfig(kwargs={**base, "iteration": max_iter})
            configs.append(
                LearnerConfig(**{slot: schedule for slot in lc_slots}, criterion=schedule)
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
