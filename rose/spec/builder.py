from __future__ import annotations

from pathlib import Path

from rose.learner import Learner

from .adapters import TaskAdapterFactory
from .schema import _REQUIRED_SLOTS, TrackingConfig, WorkflowConfig

# Task slots that are LearnerConfig fields (the UQ uncertainty task is added separately)
_LEARNER_CONFIG_SLOTS: frozenset[str] = frozenset(
    {"simulation", "training", "prediction", "active_learn", "environment", "update"}
)


def _iteration_schedule(base: dict):
    """Schedule giving every iteration ``base`` kwargs plus its ``iteration`` number.

    A factory, so each learner's schedule binds its own ``base`` (no late binding in loops).
    """
    import copy

    from rose.learner import adaptive_schedule

    # deepcopy: tasks must not share mutable values such as the pythonpath list
    return adaptive_schedule(lambda n: {"kwargs": {**copy.deepcopy(base), "iteration": n}})


def _get_learner_class(learner_type: str):
    if learner_type == "sequential_active_learner":
        from rose.active_learning.active_learner import SequentialActiveLearner

        return SequentialActiveLearner
    if learner_type == "parallel_active_learner":
        from rose.active_learning.active_learner import ParallelActiveLearner

        return ParallelActiveLearner
    if learner_type == "sequential_reinforcement_learner":
        from rose.reinforcement_learning.reinforcement_learner import SequentialReinforcementLearner

        return SequentialReinforcementLearner
    if learner_type == "uq_active_learner":
        from rose.uncertainty_quantification.uq_active_learner import SeqUQLearner

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
        if crit is not None:  # optional with a stream
            c_closure = TaskAdapterFactory.make_closure(crit.evaluator, cfg.remote)
            c_as_exec = TaskAdapterFactory.as_executable(crit.evaluator)
            learner.as_stop_criterion(
                metric_name=crit.metric,
                threshold=crit.threshold,
                operator=crit.operator,
                as_executable=c_as_exec,
            )(c_closure)

        uq_crit = cfg.uncertainty_criterion
        if uq_crit is not None:  # uq_active_learner
            learner.uncertainty_quantification(
                uq_metric_name=uq_crit.metric,
                query_size=None,  # unused by the learner; pass via parameters if a task needs it
                threshold=uq_crit.threshold,
                operator=uq_crit.operator,  # resolved by the schema
                as_executable=TaskAdapterFactory.as_executable(uq_crit.evaluator),
            )(TaskAdapterFactory.make_closure(uq_crit.evaluator, cfg.remote))

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
            task_defs = [ld.tasks[slot_name] for ld in cfg.learners]
            as_exec = task_defs[0].type == "shell"
            closure = TaskAdapterFactory.make_dispatch_closure(slot_name, task_defs, cfg.remote)
            getattr(learner, f"{slot_name}_task")(as_executable=as_exec)(closure)

    def build_learner_config(self):
        """Single LearnerConfig for sequential learners.

        Passed as initial_config to learner.start() so that parameters and iteration reach task
        kwargs via the same ROSE-native mechanism used by the parallel path.  Returns None when no
        parameters are defined.
        """
        cfg = self.config
        uq = cfg.learner.type == "uq_active_learner"  # UQ tasks always get `iteration`
        if not cfg.parameters and not cfg.remote.pythonpath and not uq:
            return None
        from rose.learner import LearnerConfig

        required = _REQUIRED_SLOTS[cfg.learner.type]
        params = dict(cfg.parameters)
        params["pythonpath"] = list(cfg.remote.pythonpath)
        schedule = _iteration_schedule(params)
        slots = {slot: schedule for slot in required & _LEARNER_CONFIG_SLOTS}
        if cfg.learner.type == "uq_active_learner":
            from rose.uncertainty_quantification.uq_learner import UQLearnerConfig

            return UQLearnerConfig(**slots, criterion=schedule, uncertainty=schedule)
        return LearnerConfig(**slots, criterion=schedule)

    def build_learner_configs(self):
        """LearnerConfig list for parallel learners.

        Returns None for non-parallel learner types (use build_learner_config
        instead) or when there's nothing to inject: no per-learner `learners:`
        block, no `parameters`, and no `remote.pythonpath`.

        When `learners:` is absent (shared tasks across learners), the list
        length falls back to `learner.parallel_learners` so that `parameters`/
        `pythonpath` still reach every learner's task kwargs.
        """
        cfg = self.config
        if cfg.learner.type != "parallel_active_learner":
            return None
        if cfg.learners is None and not cfg.parameters and not cfg.remote.pythonpath:
            return None
        from rose.learner import LearnerConfig

        required = _REQUIRED_SLOTS[cfg.learner.type]
        configs = []
        params = dict(cfg.parameters)
        params["pythonpath"] = list(cfg.remote.pythonpath)
        lc_slots = required & _LEARNER_CONFIG_SLOTS
        learner_defs = cfg.learners or [None] * cfg.learner.parallel_learners
        for i, learner_def in enumerate(learner_defs):
            # learner_id    → dispatch routing key (popped by closure, never reaches user fn)
            # iteration     → set per lookup by the schedule: get_task_config(slot, n)
            # learner_label → human-readable learner name; only injected when non-empty
            # parameters    → user-defined values from the YAML parameters: block
            base = {"learner_id": i, **params}
            if learner_def is not None and learner_def.label:
                base["learner_label"] = learner_def.label
            schedule = _iteration_schedule(base)
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
