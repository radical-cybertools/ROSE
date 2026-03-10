"""Tracking protocol and pipeline manifest types for ROSE.

This module defines the ``TrackerBase`` protocol — a pluggable observability interface that
learners call at three lifecycle points:

* ``on_start``     — once, immediately after ``add_tracker()`` is called (manifest is complete)
* ``on_iteration`` — once per iteration, with the full ``IterationState`` snapshot
* ``on_stop``      — once, when the learning loop exits for any reason

Task outputs flow into the tracker via ``on_iteration``: when a task returns a ``dict``,
ROSE automatically extracts each key-value pair into ``IterationState.state``, making
them available in the ``state`` argument passed to every ``on_iteration`` call.

Concrete implementations live in ``rose/integrations/`` (MLflow, ClearML) and are
optional — ROSE core never imports them.

Example::

    from rose.tracking import TrackerBase, PipelineManifest
    from rose.learner import IterationState

    class PrintTracker:
        def on_start(self, manifest: PipelineManifest) -> None:
            print(f"Starting {manifest.learner_type}")

        def on_iteration(self, state: IterationState) -> None:
            print(f"  iter {state.iteration}: metric={state.metric_value}")

        def on_stop(self, final_state, reason: str) -> None:
            print(f"Stopped: {reason}")

    learner.add_tracker(PrintTracker())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rose.learner import IterationState


# ---------------------------------------------------------------------------
# Pipeline manifest — captures the registered pipeline at decoration time
# ---------------------------------------------------------------------------


@dataclass
class TaskManifest:
    """Snapshot of a single registered task captured at decoration time.

    Attributes:
        func_name: The decorated function's ``__name__``.
        func_module: The decorated function's ``__module__``.
        as_executable: ``True`` when submitted via ``asyncflow.executable_task``,
            ``False`` when submitted via ``asyncflow.function_task``.
        decor_kwargs: Extra keyword arguments passed to the task decorator
            (e.g. ``num_gpus``, ``ranks``).
    """

    func_name: str
    func_module: str
    as_executable: bool
    decor_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionManifest(TaskManifest):
    """Snapshot of the stop-criterion task, extending ``TaskManifest`` with metric metadata.

    Attributes:
        metric_name: Name of the metric being tracked (e.g. ``"MSE"``).
        threshold: Target threshold value for the stopping condition.
        operator: Comparison operator string (``"<"``, ``">"``, etc.).
    """

    metric_name: str = ""
    threshold: float = 0.0
    operator: str = ""


@dataclass
class PipelineManifest:
    """Complete snapshot of the registered pipeline, built at ``add_tracker()`` time.

    All information comes from the decorator metadata already stored on the learner —
    no user annotation required.

    Attributes:
        learner_type: The learner class name (e.g. ``"SequentialActiveLearner"``).
        tasks: Dictionary of registered tasks keyed by task name
            (``"simulation"``, ``"training"``, ``"active_learn"``, etc.).
        criterion: Stop-criterion manifest, or ``None`` if no criterion is registered.
        parallel_count: Number of parallel learners for parallel runs, ``None`` for
            sequential runs.
    """

    learner_type: str
    tasks: dict[str, TaskManifest] = field(default_factory=dict)
    criterion: CriterionManifest | None = None
    parallel_count: int | None = None


# ---------------------------------------------------------------------------
# TrackerBase protocol
# ---------------------------------------------------------------------------


class TrackerBase:
    """Protocol defining the tracking interface for ROSE learners.

    Implement any subset of these methods to observe a learner's run. All
    methods have default no-op implementations so you only override what you
    need.

    The three lifecycle methods map to the outer learning loop:

    - ``on_start``     → pipeline manifest, fired once at ``add_tracker()``
    - ``on_iteration`` → ``IterationState`` snapshot after each iteration completes
    - ``on_stop``      → final state and stop reason when the loop exits

    Task outputs are captured via return values: when a task returns a ``dict``,
    ROSE extracts each key-value pair into ``IterationState.state`` automatically,
    making them available in every ``on_iteration`` call.
    """

    def on_start(self, manifest: PipelineManifest) -> None:
        """Called once immediately when ``add_tracker()`` is invoked.

        The pipeline manifest is already fully populated at this point (all
        task decorators have fired), so ``manifest`` contains the complete
        pipeline definition.

        Args:
            manifest: Full pipeline snapshot (task names, functions, criterion
                metadata, parallel count).
        """

    def on_iteration(self, state: IterationState) -> None:
        """Called once per iteration with the complete ``IterationState`` snapshot.

        Invoked just before ``yield state`` inside ``start()``, so it fires
        before the user's ``async for`` loop body runs. The state contains
        the consolidated snapshot of all task outputs registered via
        ``register_state()`` during this iteration.

        Args:
            state: Complete iteration snapshot including ``iteration``,
                ``metric_value``, ``should_stop``, ``current_config``,
                ``learner_id``, and ``state`` dict with all registered task data.
        """

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        """Called once when the learning loop exits for any reason.

        Invoked from the ``finally`` block of ``start()``, so it fires whether
        the loop completed normally, hit a stop criterion, was externally
        stopped, or raised an exception.

        Args:
            final_state: The last ``IterationState`` that was yielded, or
                ``None`` if no iterations ran (e.g. validation error on start).
            reason: One of:
                - ``"criterion_met"`` — stop criterion threshold was reached
                - ``"max_iter_reached"`` — all iterations completed normally
                - ``"stopped"`` — ``learner.stop()`` was called or user broke
                  out of the ``async for`` loop
                - ``"error"`` — an unhandled exception occurred
        """
