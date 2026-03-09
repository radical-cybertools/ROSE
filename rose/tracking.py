"""Tracking protocol and pipeline manifest types for ROSE.

This module defines the ``TrackerBase`` protocol — a pluggable observability interface that
learners call at three lifecycle points:

* ``on_start``     — once, immediately after ``add_tracker()`` is called (manifest is complete)
* ``on_iteration`` — once per iteration, with the full ``IterationState`` snapshot
* ``on_stop``      — once, when the learning loop exits for any reason

An optional fourth method, ``on_state_update``, is called in real-time each time
``register_state()`` fires inside a task, giving trackers access to mid-iteration
process data before the iteration snapshot is built.

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

        def on_state_update(self, key: str, value) -> None:
            pass  # optional — no-op is fine

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
    """Snapshot of the stop-criterion task, extending ``TaskManifest`` with
    metric metadata.

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

    The four methods map to two distinct data channels:

    **Snapshot channel** (iteration-level):
        - ``on_start``     → pipeline manifest before the first iteration
        - ``on_iteration`` → ``IterationState`` snapshot after each iteration
        - ``on_stop``      → final state and stop reason when the loop exits

    **Streaming channel** (sub-iteration, real-time):
        - ``on_state_update`` → fires for every ``register_state(key, value)``
          call inside a task, before ``build_iteration_state()`` consolidates
          them into the ``IterationState`` snapshot.

    Note:
        For parallel learners, ``on_state_update`` is **not** called for
        sub-learner task code — only ``on_iteration`` is called (with the
        already-consolidated ``IterationState.state`` dict). This is sufficient
        for all standard tracking needs; streaming sub-learner updates would
        require injecting the tracker into each sub-learner instance.
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

    def on_state_update(self, key: str, value: Any) -> None:
        """Called in real-time for every ``register_state(key, value)`` call.

        Fires inside the task execution, before ``build_iteration_state()``
        consolidates all registered state into the ``IterationState`` snapshot.
        Useful for streaming per-task metrics (e.g. training loss per batch)
        without waiting for the full iteration to complete.

        For parallel learners this method is **not** called — use
        ``on_iteration`` and read ``state.state`` instead.

        Args:
            key: State key (e.g. ``"loss"``, ``"labeled_count"``).
            value: Associated value.
        """
