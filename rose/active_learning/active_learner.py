import asyncio
import contextlib
import dataclasses
import itertools
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from radical.asyncflow import WorkflowEngine

from ..learner import IterationState, Learner, LearnerConfig, _stream_parallel
from ..stream import Stream, _Subscription


class SequentialActiveLearner(Learner):
    """Sequential active learner that runs iterations one after another.

    This class implements a sequential active learning approach where each iteration
    consists of simulation, training, and active learning phases that run in sequence.
    The learner must be started with the `start()` method, which returns an async
    iterator that yields state at each iteration.

    Attributes:
        learner_id (Optional[int]): Identifier for the learner, used for logging.

    Example:
        Basic usage::

            learner = SequentialActiveLearner(asyncflow)

            @learner.simulation_task(as_executable=False)
            async def simulation(*args):
                ...

            @learner.training_task(as_executable=False)
            async def training(*args):
                ...

            @learner.active_learn_task(as_executable=False)
            async def active_learn(*args):
                ...

            async for state in learner.start(max_iter=10):
                print(f"Iteration {state.iteration}: {state.metric_value}")
                if state.metric_value < 0.01:
                    break
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Sequential Active Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks.
        """
        super().__init__(asyncflow, register_and_submit=True)
        self.learner_id: int | None = None

        self._iteration_state: IterationState | None = None
        self._pending_config: LearnerConfig | None = None
        self._max_iter: int | None = None

    async def start(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_simulation_step: bool = False,
        initial_config: LearnerConfig | None = None,
        stream: Stream | None = None,
    ) -> AsyncIterator[IterationState]:
        """Start the learner and yield state at each iteration.

        This is the main entry point for running the learner. It returns an
        async iterator that yields IterationState at each iteration, giving
        the caller full control over the learning loop.

        Args:
            max_iter: Maximum number of iterations to run. If 0, runs until
                stop criterion is met (requires criterion_function to be set).
            skip_pre_loop: If True, skips the initial simulation and training
                phases before the main learning loop.
            skip_simulation_step: If True, simulation tasks will be skipped and
                managed externally.
            initial_config: Initial configuration object. Can be modified
                between iterations via set_next_config().
            stream: Optional :class:`~rose.stream.Stream`. Each iteration then
                waits for a window of streamed items, which is passed as first
                argument to the simulation task (to the training task with
                ``skip_simulation_step``), i.e. ``simulation(window, *args)``
                and, from the second window on, ``simulation(window, *args,
                acl_result)``. A met criterion does not end the loop: it marks
                the state ``model_ready`` (always set without a criterion) and
                fires :meth:`on_model_ready` callbacks. The loop ends on
                ``stop()``, stream exhaustion, or ``max_iter``. Executable tasks
                get the window as first argument of their command builder; YAML
                spec shell tasks ignore it.

        Yields:
            IterationState containing current iteration info, metrics, and
            all registered state from tasks.

        Raises:
            ValueError: If required functions are not set, if neither
                max_iter nor criterion_function nor stream is provided, or if
                stream is combined with skip_pre_loop.

        Example:
            Basic usage::

                async for state in learner.start(max_iter=20):
                    print(f"Iteration {state.iteration}, metric={state.metric_value}")

                    # Stop early based on custom condition
                    if state.metric_value and state.metric_value < 0.01:
                        break

            Modifying configuration between iterations::

                async for state in learner.start(max_iter=20):
                    # Adjust config based on state
                    if state.mean_uncertainty and state.mean_uncertainty < 0.2:
                        learner.set_next_config(LearnerConfig(
                            training=TaskConfig(kwargs={'--lr': '0.0001'})
                        ))
        """
        # Validation
        if not skip_simulation_step and not self.simulation_function:
            raise ValueError("Simulation function must be set when not using simulation pool!")
        if not self.training_function or not self.active_learn_function:
            raise ValueError("Training and Active Learning functions must be set!")
        if max_iter == 0 and not self.criterion_function and stream is None:
            raise ValueError("Either max_iter > 0 or criterion_function must be provided.")
        if stream is not None and skip_pre_loop:
            raise ValueError("stream cannot be combined with skip_pre_loop")

        self._max_iter = max_iter if max_iter > 0 else None
        learner_config = initial_config
        _stop_reason = "max_iter_reached"

        # a Subscription is passed in by ParallelActiveLearner
        sub = None
        if stream is not None:
            sub = stream if isinstance(stream, _Subscription) else stream._subscribe()
            self._subscriptions.append(sub)
            if self.is_stopped:
                sub.wake()  # stop() came before the subscription existed
        window_size: int | None = None

        learner_suffix = f" (Learner-{self.learner_id})" if self.learner_id is not None else ""
        print(f"Starting Active Learner{learner_suffix}")

        # Initialize task references
        sim_task: Any = ()
        train_task: Any = ()

        # Pre-loop phase: register and await sim/train, but don't extract state yet
        # State extraction happens inside the loop after clear_state()
        sim_result: Any = None
        train_result: Any = None

        try:
            if not skip_pre_loop:
                window = await self._pull_window(sub)
                if window == []:
                    _stop_reason = "stopped" if self.is_stopped else "stream_exhausted"
                    return

                train_config = self._get_iteration_task_config(
                    self.training_function, learner_config, "training", 0
                )
                sim_config = None
                if not skip_simulation_step:
                    sim_config = self._get_iteration_task_config(
                        self.simulation_function, learner_config, "simulation", 0
                    )
                window_size = self._inject_window(window, sim_config or train_config)

                if skip_simulation_step:
                    train_task = self._register_task(train_config)
                else:
                    sim_task = self._register_task(sim_config)
                    train_task = self._register_task(train_config, deps=sim_task)

                    # Await simulation result (extract state later, inside loop)
                    sim_result = await sim_task

                # Await training result (extract state later, inside loop)
                train_result = await train_task

            # Determine iteration range
            iteration_range: Iterator[int] | range
            if max_iter == 0:
                iteration_range = itertools.count()
            else:
                iteration_range = range(max_iter)

            # Main iteration loop
            for i in iteration_range:
                learner_prefix = (
                    f"[Learner-{self.learner_id}] " if self.learner_id is not None else ""
                )
                if self.is_stopped:
                    print(f"{learner_prefix}Stop requested, exiting learning loop.")
                    _stop_reason = "stopped"
                    break

                # Check for pending config
                if self._pending_config is not None:
                    learner_config = self._pending_config
                    self._pending_config = None

                # Clear transient state from previous iteration
                self.clear_state()

                # Extract state from sim/train results
                # (prepared in previous iteration or pre-loop)
                if window_size is not None:
                    self.register_state("window_size", window_size)  # task results may override
                if not skip_simulation_step and sim_result is not None:
                    self._extract_state_from_result(sim_result)
                if train_result is not None:
                    self._extract_state_from_result(train_result)

                print(f"{learner_prefix}Starting Iteration-{i}")

                # Get iteration-specific AL config
                acl_config = self._get_iteration_task_config(
                    self.active_learn_function, learner_config, "active_learn", i
                )

                # Register AL task with dependencies
                if skip_simulation_step:
                    acl_task = self._register_task(acl_config, deps=train_task)
                else:
                    acl_task = self._register_task(acl_config, deps=(sim_task, train_task))

                # Await AL task and extract state from dict result
                acl_result = await acl_task
                if self.is_stopped:
                    _stop_reason = "stopped"
                    break
                self._extract_state_from_result(acl_result)

                # Check stop criterion if configured
                metric_value: float | None = None
                should_stop = False

                if self.criterion_function:
                    criterion_config = self._get_iteration_task_config(
                        self.criterion_function, learner_config, "criterion", i
                    )
                    stop_task = self._register_task(criterion_config)
                    stop_result = await stop_task
                    if self.is_stopped:
                        _stop_reason = "stopped"
                        break
                    should_stop, metric_value = self._check_stop_criterion(stop_result)

                # Build iteration state
                self._iteration_state = self.build_iteration_state(
                    iteration=i,
                    metric_value=metric_value,
                    should_stop=should_stop,
                    current_config=learner_config,
                    model_ready=sub is not None and (should_stop or not self.criterion_function),
                )

                # Notify trackers and model callbacks, then yield control to caller
                self._notify_trackers_iteration(self._iteration_state)
                if self._iteration_state.model_ready:
                    await self._notify_model_ready(self._iteration_state)
                yield self._iteration_state

                # Check if user loop broke or criterion met
                # (in stream mode the criterion only gates publishing)
                if should_stop and sub is None:
                    _stop_reason = "criterion_met"
                    break
                if sub is not None and self._max_iter is not None and i + 1 == self._max_iter:
                    break

                # Wait for data first, so a config set meanwhile applies to the whole iteration
                window = await self._pull_window(sub)
                if window == []:
                    _stop_reason = "stopped" if self.is_stopped else "stream_exhausted"
                    break

                # Prepare next iteration using potentially updated config
                next_config = self._pending_config or learner_config
                next_train_config = self._get_iteration_task_config(
                    self.training_function, next_config, "training", i + 1
                )
                next_sim_config = None
                if not skip_simulation_step:
                    next_sim_config = self._get_iteration_task_config(
                        self.simulation_function, next_config, "simulation", i + 1
                    )

                window_size = self._inject_window(window, next_sim_config or next_train_config)

                if skip_simulation_step:
                    sim_task = ()
                    sim_result = None
                    train_task = self._register_task(next_train_config, deps=acl_task)
                else:
                    sim_task = self._register_task(next_sim_config, deps=acl_task)
                    train_task = self._register_task(next_train_config, deps=sim_task)

                    # Await simulation result (extract state in next iteration)
                    sim_result = await sim_task
                    if self.is_stopped:
                        _stop_reason = "stopped"
                        break

                # Await training result (extract state in next iteration)
                train_result = await train_task
                if self.is_stopped:
                    _stop_reason = "stopped"
                    break
        except GeneratorExit:
            # consumer left the async for loop, e.g. on should_stop as documented for streams
            last = self._iteration_state
            _stop_reason = "criterion_met" if last is not None and last.should_stop else "stopped"
            raise
        except asyncio.CancelledError:
            _stop_reason = "stopped"
            raise
        except Exception:
            _stop_reason = "error"
            raise
        finally:
            if sub is not None:
                sub.unsubscribe()
                self._subscriptions.remove(sub)
            self._notify_trackers_stop(self._iteration_state, _stop_reason)

    @staticmethod
    async def _pull_window(sub: _Subscription | None) -> list[Any] | None:
        """Next window, None without stream; empty means stopped or exhausted."""
        return await sub.next_window() if sub is not None else None

    @staticmethod
    def _inject_window(window: list[Any] | None, task_config: dict[str, Any]) -> int | None:
        """Prepend the window to the task's args; return its size (None without stream)."""
        if window is None:
            return None
        task_config["args"] = (window, *task_config["args"])
        return len(window)

    def set_next_config(self, config: LearnerConfig) -> None:
        """Set configuration for the next iteration.

        Called between iterations to modify hyperparameters, sample selection
        strategy, or other task configurations.

        Args:
            config: Configuration to apply in the next iteration. This will
                override the initial_config for subsequent iterations.

        Example::

            async for state in learner.start(max_iter=20):
                if state.iteration > 10:
                    # Reduce learning rate after iteration 10
                    learner.set_next_config(LearnerConfig(
                        training=TaskConfig(kwargs={'--lr': '0.0001'})
                    ))
        """
        self._pending_config = config

    def get_current_state(self) -> IterationState | None:
        """Get the current iteration state.

        Returns:
            Current IterationState if in iteration loop, None otherwise.
        """
        return self._iteration_state

    def get_max_iterations(self) -> int | None:
        """Get the maximum iterations configured for current run.

        Returns:
            Maximum iterations or None if running until criterion met.
        """
        return self._max_iter

    async def teach(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_simulation_step: bool = False,
        learner_config: LearnerConfig | None = None,
        stream: Stream | None = None,
    ) -> IterationState | None:
        """Run active learning loop to completion.

        .. deprecated::
            Use :meth:`start` instead. This method will be removed in a future
            version. The `start()` method returns an async iterator giving you
            full control over each iteration.

        Args:
            max_iter: Maximum number of iterations to run.
            skip_pre_loop: If True, skips the initial simulation and training.
            skip_simulation_step: If True, skips simulation tasks.
            learner_config: Configuration for the learner.
            stream: Optional Stream, see :meth:`start`. With an unbounded
                stream, teach() returns only after stop() or Stream.close().

        Returns:
            Final IterationState after completion, or None if no iterations ran.
        """
        warnings.warn(
            "teach() is deprecated and will be removed in a future version. "
            "Use start() instead which returns an async iterator for full control.",
            DeprecationWarning,
            stacklevel=2,
        )
        final_state = None
        async for state in self.start(
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            skip_simulation_step=skip_simulation_step,
            initial_config=learner_config,
            stream=stream,
        ):
            final_state = state
        return final_state


class ParallelActiveLearner(Learner):
    """Parallel active learner that runs multiple SequentialActiveLearners concurrently.

    This class orchestrates multiple SequentialActiveLearner instances to run in parallel, allowing
    for concurrent exploration of the learning space. Each learner can be configured independently
    through per-learner LearnerConfig objects.

    The parallel learner manages the lifecycle of all sequential learners and collects their results
    when all have completed their learning processes.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Parallel Active Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks
                across all parallel learners.
        """
        super().__init__(asyncflow, register_and_submit=False)

    def _create_sequential_learner(
        self, learner_id: int, config: LearnerConfig | None
    ) -> SequentialActiveLearner:
        """Create a SequentialActiveLearner instance for a parallel learner.

        Creates and configures a new SequentialActiveLearner with the same base
        functions as the parent parallel learner, but with a unique identifier
        for logging and debugging purposes.

        Args:
            learner_id: Unique identifier for the learner, used in logging
                and debugging output.
            config: Configuration object for this specific learner. Can be None
                to use default configuration.

        Returns:
            A fully configured SequentialActiveLearner instance ready to run
            independently in the parallel learning environment.
        """
        # Create a new sequential learner with the same asyncflow
        sequential_learner: SequentialActiveLearner = SequentialActiveLearner(self.asyncflow)

        # Copy the base functions from the parent learner
        sequential_learner.simulation_function = self.simulation_function
        sequential_learner.training_function = self.training_function
        sequential_learner.active_learn_function = self.active_learn_function
        sequential_learner.criterion_function = self.criterion_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_id = learner_id

        return sequential_learner

    def _convert_to_sequential_config(
        self, parallel_config: LearnerConfig | None
    ) -> LearnerConfig | None:
        """Convert a LearnerConfig to a LearnerConfig.

        Note: This method currently performs a direct copy as both parallel and
        sequential learners use the same LearnerConfig type. This method exists
        to provide a clear interface for potential future differences in
        configuration handling.

        Args:
            parallel_config: Configuration object designed for parallel learner
                usage. Contains simulation, training, active_learn, and criterion
                parameters.

        Returns:
            Equivalent LearnerConfig suitable for use with SequentialActiveLearner,
            or None if input was None.
        """
        if parallel_config is None:
            return None

        # Create LearnerConfig with same parameters
        return LearnerConfig(
            simulation=parallel_config.simulation,
            training=parallel_config.training,
            active_learn=parallel_config.active_learn,
            criterion=parallel_config.criterion,
        )

    async def start(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_simulation_step: bool = False,
        learner_configs: list[LearnerConfig | None] | None = None,
        stream: Stream | None = None,
    ) -> AsyncIterator[IterationState]:
        """Run parallel active learning by launching multiple SequentialActiveLearners.

        Orchestrates multiple SequentialActiveLearner instances to run concurrently,
        each with potentially different configurations. States are streamed in real
        time as each learner completes an iteration — use ``async for`` to consume them.

        Each yielded ``IterationState`` includes a ``learner_id`` field (int) indicating
        which parallel learner produced it.

        Args:
            parallel_learners: Number of parallel learners to run concurrently.
                Must be >= 2 (use SequentialActiveLearner directly for single learner).
            max_iter: Maximum number of iterations for each learner. If 0,
                learners run until their individual stop criteria are met.
            skip_pre_loop: If True, all learners skip their initial simulation
                and training phases.
            learner_configs: list of configuration objects, one for each learner.
                If None, all learners use default configuration. Length must
                match parallel_learners if provided.
            skip_simulation_step: if True, all learners will skip the simulation
                step and the learner will consider a simulation pool already exist.
            stream: Optional Stream; every learner receives every streamed item
                (see SequentialActiveLearner.start). on_model_ready callbacks
                registered on this learner fire with ``learner_id`` set. A
                failing learner stops all others.

        Yields:
            IterationState for each iteration of each learner, in arrival order.
            Each state has ``learner_id`` set to the integer index of the learner.

        Raises:
            ValueError: If parallel_learners < 2 (use SequentialActiveLearner instead).
            ValueError: If learner_configs length doesn't match parallel_learners.
            Exception: Re-raises any exception from a learner after all learners finish.

        Example::

            async for state in learner.start(parallel_learners=3, max_iter=10):
                print(f"Learner {state.learner_id}, iter {state.iteration}: {state.metric_value}")
        """
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialActiveLearner")

        # Prepare learner configurations
        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")
        if stream is not None and skip_pre_loop:
            raise ValueError("stream cannot be combined with skip_pre_loop")

        # subscribe all learners before any pulls a window, so all see the backlog
        subs = [stream._subscribe() if stream is not None else None for _ in learner_configs]
        self._subscriptions.extend(sub for sub in subs if sub is not None)

        print(f"Starting Parallel Active Learning with {parallel_learners} learners")

        # Factory required: plain closure in a loop would capture the same variable reference.
        def make_run_fn(learner_id: int):
            async def run_learner(queue: asyncio.Queue) -> None:
                try:
                    sequential_learner: SequentialActiveLearner = self._create_sequential_learner(
                        learner_id, learner_configs[learner_id]
                    )
                    if stream is not None:
                        # parent stop() wakes the subscription and stops the learner
                        sequential_learner._stop_event = self._stop_event
                    async for state in sequential_learner.start(
                        max_iter=max_iter,
                        skip_pre_loop=skip_pre_loop,
                        skip_simulation_step=skip_simulation_step,
                        initial_config=learner_configs[learner_id],
                        stream=subs[learner_id],
                    ):
                        if self.is_stopped:
                            sequential_learner.stop()
                        state = dataclasses.replace(state, learner_id=learner_id)
                        if state.model_ready:
                            await self._notify_model_ready(state)
                        await queue.put(("state", state))
                    self.metric_values_per_iteration[f"learner-{learner_id}"] = (
                        sequential_learner.metric_values_per_iteration
                    )
                except Exception as e:
                    print(f"[ActiveLearner-{learner_id}] failed with error: {e}")
                    await queue.put(("error", e))
                    if stream is not None:
                        self.stop()  # others would wait for windows forever
                finally:
                    await queue.put(("done", None))

            return run_learner

        _stop_reason = "max_iter_reached"
        try:
            # aclosing: a consumer break must reap the learner tasks before unsubscribing
            async with contextlib.aclosing(
                _stream_parallel([make_run_fn(i) for i in range(parallel_learners)])
            ) as states:
                async for state in states:
                    self._notify_trackers_iteration(state)
                    yield state
            if self.is_stopped:
                _stop_reason = "stopped"
            elif stream is not None and stream._ended:
                _stop_reason = "stream_exhausted"
        except (GeneratorExit, asyncio.CancelledError):
            _stop_reason = "stopped"  # consumer left the async for loop, or cancelled
            raise
        except Exception:
            _stop_reason = "error"
            raise
        finally:
            for sub in subs:
                if sub is not None:
                    sub.unsubscribe()
                    self._subscriptions.remove(sub)
            self._notify_trackers_stop(self._iteration_state, _stop_reason)

    async def teach(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_simulation_step: bool = False,
        learner_configs: list[LearnerConfig | None] | None = None,
        stream: Stream | None = None,
    ) -> list[Any]:
        """Run parallel active learning loop to completion.

        .. deprecated::
            Use :meth:`start` instead. This method will be removed in a future
            version.

        Args:
            parallel_learners: Number of parallel learners to run concurrently.
            max_iter: Maximum number of iterations for each learner.
            skip_pre_loop: If True, skips the initial simulation and training.
            skip_simulation_step: If True, skips simulation tasks.
            learner_configs: Configuration for each learner.
            stream: Optional Stream, see :meth:`start`.

        Returns:
            List of final IterationState from each learner.
        """
        warnings.warn(
            "teach() is deprecated and will be removed in a future version. Use start() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        final_states: dict[int, IterationState | None] = {}
        async for state in self.start(
            parallel_learners=parallel_learners,
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            skip_simulation_step=skip_simulation_step,
            learner_configs=learner_configs,
            stream=stream,
        ):
            final_states[state.learner_id] = state
        return [final_states.get(i) for i in range(parallel_learners)]
