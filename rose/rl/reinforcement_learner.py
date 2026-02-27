import asyncio
import itertools
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from functools import wraps
from typing import Any

import typeguard
from radical.asyncflow import WorkflowEngine

from rose.learner import IterationState, Learner, LearnerConfig


class ReinforcementLearner(Learner):
    """Base class for reinforcement learning implementations.

    This class provides the foundation for implementing reinforcement learning
    algorithms with asynchronous workflow management and configuration support.
    It maintains dictionaries for test, update, and environment functions that
    can be registered and executed as tasks with per-iteration configuration.

    Attributes:
        test_function (dict): Dictionary storing test function configurations.
        update_function (dict): Dictionary storing update function configurations.
        environment_function (dict): Dictionary storing environment function
        configurations.
        update_task (Callable): Registered update task callable.
        environment_task (Callable): Registered environment task decorator.
        learner_id (Optional[int]): Identifier for the learner, used for logging.
    """

    @typeguard.typechecked
    def __init__(self, asyncflow: WorkflowEngine, register_and_submit: bool = True) -> None:
        """Initialize the ReinforcementLearner.

        Args:
            asyncflow (WorkflowEngine): The workflow engine for managing
            asynchronous tasks.
            register_and_submit (bool, optional): Whether to automatically register and
                submit tasks. Defaults to True.
        """
        super().__init__(asyncflow, register_and_submit)

        self.update_function = {}
        self.environment_function = {}
        self.learner_id: int | None = None
        self.test_function = self.criterion_function

        # Create custom decorators that immediately register functions
        self.update_task: Callable = self.register_decorator("update")
        self.environment_task: Callable = self.register_decorator("environment")


class SequentialReinforcementLearner(ReinforcementLearner):
    """Sequential reinforcement learning implementation with configuration support.

    This class implements a sequential reinforcement learning loop where the learner
    interacts with the environment in a series of steps, updating its policy based
    on the rewards received from the environment. The learner must be started with
    the `start()` method, which returns an async iterator that yields state at each
    iteration.

    The learning process follows this sequential pattern:

        Iteration 1:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration 2:
        [Env] -> [Update] -> [Test]
                    |
                    v
        ...

    Each iteration consists of three sequential steps:
    1. Environment interaction to collect experiences
    2. Policy update based on collected experiences
    3. Testing/evaluation of the updated policy

    Example:
        Basic usage::

            learner = SequentialReinforcementLearner(asyncflow)

            @learner.environment_task(as_executable=False)
            async def environment(*args):
                ...

            @learner.update_task(as_executable=False)
            async def update(*args):
                ...

            async for state in learner.start(max_iter=100):
                print(f"Iteration {state.iteration}: reward={state.episode_reward}")
                if state.episode_reward > 200:
                    break
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the SequentialReinforcementLearner.

        Args:
            asyncflow (WorkflowEngine): The workflow engine for
            managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=True)
        self._iteration_state: IterationState | None = None
        self._pending_config: LearnerConfig | None = None
        self._max_iter: int | None = None

    async def start(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_environment_step: bool = False,
        initial_config: LearnerConfig | None = None,
    ) -> AsyncIterator[IterationState]:
        """Start the learner and yield state at each iteration.

        This is the main entry point for running the learner. It returns an
        async iterator that yields IterationState at each iteration, giving
        the caller full control over the learning loop.

        Args:
            max_iter: Maximum number of iterations to run. If 0, runs until
                stop criterion is met (requires criterion_function to be set).
            skip_pre_loop: If True, skips the initial environment and update
                phases before the main learning loop.
            skip_environment_step: If True, skips the environment task and assumes
                experiences are provided externally. Defaults to False.
            initial_config: Initial configuration object. Can be modified
                between iterations via set_next_config().

        Yields:
            IterationState containing current iteration info, metrics, and
            all registered state from tasks.

        Raises:
            ValueError: If environment function is not set when needed.
            ValueError: If update function is not set.
            ValueError: If neither max_iter nor criterion_function is provided.

        Example:
            Basic usage::

                async for state in learner.start(max_iter=100):
                    print(f"Iteration {state.iteration}, reward={state.episode_reward}")

                    # Stop early based on custom condition
                    if state.episode_reward and state.episode_reward > 200:
                        break

            Modifying configuration between iterations::

                async for state in learner.start(max_iter=100):
                    if state.iteration > 50:
                        learner.set_next_config(LearnerConfig(
                            update=TaskConfig(kwargs={'learning_rate': 0.0001})
                        ))
        """
        # Validation
        if not skip_environment_step and not self.environment_function:
            raise ValueError("Environment function must be set unless using external experiences!")
        if not self.update_function:
            raise ValueError("Update function must be set!")
        if not max_iter and not self.criterion_function:
            raise ValueError("Either max_iter > 0 or criterion_function must be provided.")

        self._max_iter = max_iter if max_iter > 0 else None
        learner_config = initial_config

        learner_suffix = f" (Learner-{self.learner_id})" if self.learner_id is not None else ""
        print(f"Starting Sequential RL Learner{learner_suffix}")

        # Initialize task references
        env_task: Any = ()
        update_task: Any = ()

        # Pre-loop phase: register and await env/update, but don't extract state yet
        # State extraction happens inside the loop after clear_state()
        env_result: Any = None
        update_result: Any = None

        if not skip_pre_loop:
            if skip_environment_step:
                update_config = self._get_iteration_task_config(
                    self.update_function, learner_config, "update", 0
                )
                update_task = self._register_task(update_config)
            else:
                env_config = self._get_iteration_task_config(
                    self.environment_function, learner_config, "environment", 0
                )
                update_config = self._get_iteration_task_config(
                    self.update_function, learner_config, "update", 0
                )
                env_task = self._register_task(env_config)
                update_task = self._register_task(update_config, deps=env_task)

                # Await environment result (extract state later, inside loop)
                env_result = await env_task

            # Await update result (extract state later, inside loop)
            update_result = await update_task

        # Determine iteration range
        iteration_range: Iterator[int] | range
        if max_iter == 0:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main iteration loop
        for i in iteration_range:
            learner_prefix = f"[Learner-{self.learner_id}] " if self.learner_id is not None else ""
            if self.is_stopped:
                print(f"{learner_prefix}Stop requested, exiting learning loop.")
                break

            # Check for pending config update
            if self._pending_config is not None:
                learner_config = self._pending_config
                self._pending_config = None

            # Clear transient state from previous iteration
            self.clear_state()

            # Extract state from env/update results
            # (prepared in previous iteration or pre-loop)
            if not skip_environment_step and env_result is not None:
                self._extract_state_from_result(env_result)
            if update_result is not None:
                self._extract_state_from_result(update_result)

            learner_prefix = f"[Learner-{self.learner_id}] " if self.learner_id is not None else ""
            print(f"{learner_prefix}Starting Iteration-{i}")

            # Check stop criterion if configured
            metric_value: float | None = None
            should_stop = False

            if self.criterion_function:
                criterion_config = self._get_iteration_task_config(
                    self.criterion_function, learner_config, "criterion", i
                )
                stop_task = self._register_task(criterion_config, deps=update_task)
                stop_result = await stop_task

                # Extract state from criterion result, excluding handled keys
                self._extract_state_from_result(
                    stop_result, exclude_keys={"metric_value", "should_stop"}
                )

                should_stop, metric_value = self._check_stop_criterion(stop_result)

            # Build iteration state
            self._iteration_state = self.build_iteration_state(
                iteration=i,
                metric_value=metric_value,
                should_stop=should_stop,
                current_config=learner_config,
            )

            # YIELD CONTROL TO CALLER
            yield self._iteration_state

            # Check if caller broke out or criterion met
            if should_stop:
                break

            # Prepare next iteration using potentially updated config
            next_config = self._pending_config or learner_config
            next_update_config = self._get_iteration_task_config(
                self.update_function, next_config, "update", i + 1
            )

            if skip_environment_step:
                env_task = ()
                env_result = None
                update_task = self._register_task(next_update_config, deps=stop_task)
            else:
                next_env_config = self._get_iteration_task_config(
                    self.environment_function, next_config, "environment", i + 1
                )
                env_task = self._register_task(next_env_config, deps=stop_task)
                update_task = self._register_task(next_update_config, deps=env_task)

                # Await environment result (extract state in next iteration)
                env_result = await env_task
                if self.is_stopped:
                    break

            # Await update result (extract state in next iteration)
            update_result = await update_task

    def set_next_config(self, config: LearnerConfig) -> None:
        """Set configuration for the next iteration.

        Called between iterations to modify hyperparameters or other
        task configurations.

        Args:
            config: Configuration to apply in the next iteration.

        Example::

            async for state in learner.start(max_iter=100):
                if state.iteration > 50:
                    learner.set_next_config(LearnerConfig(
                        update=TaskConfig(kwargs={'learning_rate': 0.0001})
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

    async def learn(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        skip_simulation_step: bool = False,
        learner_config: LearnerConfig | None = None,
    ) -> IterationState | None:
        """Run reinforcement learning loop to completion.

        .. deprecated::
            Use :meth:`start` instead. This method will be removed in a future
            version. The `start()` method returns an async iterator giving you
            full control over each iteration.

        Args:
            max_iter: Maximum number of iterations to run.
            skip_pre_loop: If True, skips the initial environment and update phases.
            skip_simulation_step: If True, skips environment tasks.
            learner_config: Configuration for the learner.

        Returns:
            Final IterationState after completion, or None if no iterations ran.
        """
        warnings.warn(
            "learn() is deprecated and will be removed in a future version. "
            "Use start() instead which returns an async iterator for full control.",
            DeprecationWarning,
            stacklevel=2,
        )
        final_state = None
        async for state in self.start(
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            skip_environment_step=skip_simulation_step,
            initial_config=learner_config,
        ):
            final_state = state
        return final_state


class ParallelExperience(ReinforcementLearner):
    """Parallel experience collection reinforcement learning implementation.

    This class implements a parallel reinforcement learning loop where multiple
    environments run in parallel to collect experiences. After all environments
    complete their data collection, the experiences are merged and a single
    update step is performed on the aggregated data.

    The learning process follows this parallel pattern:

        Environment 1       Environment 2     Environment 3
            |                   |                 |
          [Collect]         [Collect]         [Collect]
            |                   |                 |
                --->   [Merge Experiences]   <---
                                |
                                v
                        [Update Policy]
                                |
                                v
                        [Test Policy]

    This approach is particularly useful for:
    - Collecting diverse experiences from multiple environment instances
    - Improving sample efficiency through parallel data collection
    - Implementing distributed reinforcement learning algorithms

    Attributes:
        environment_functions (dict[str, dict]): Dictionary mapping environment names
            to their function configurations.
        work_dir (str): Working directory for saving and loading experience banks.

    Example:
        Basic usage::

            learner = ParallelExperience(asyncflow)

            @learner.environment_task(name='env_1')
            def environment_1(*args):
                ...

            @learner.environment_task(name='env_2')
            def environment_2(*args):
                ...

            @learner.update_task(as_executable=False)
            async def update(*args):
                ...

            async for state in learner.start(max_iter=100):
                print(f"Iteration {state.iteration}")
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the ParallelExperience learner.

        Args:
            asyncflow (WorkflowEngine): The workflow engine for
            managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=False)
        self.environment_functions: dict[str, dict] = {}
        self.work_dir = "."
        self._iteration_state: IterationState | None = None
        self._pending_config: LearnerConfig | None = None
        self._max_iter: int | None = None

    def environment_task(self, name: str) -> Callable:
        """Decorator to register an environment task under a given name.

        This decorator allows registering multiple environment functions that will
        be executed in parallel during the learning process. Each environment
        should collect experiences independently.

        Args:
            name (str): Unique name identifier for the environment task.

        Returns:
            Callable: Decorator function that wraps the environment function.

        Example:
            @par_exp.environment_task(name='env_1')
            def environment_1(*args):
                # Environment 1 logic here
                pass

            @par_exp.environment_task(name='env_2')
            def environment_2(*args):
                # Environment 2 logic here
                pass
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.environment_functions[name] = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                }
                if self.register_and_submit:
                    return self._register_task(self.environment_functions[name])

            return wrapper

        return decorator

    def merge_banks(self) -> None:
        """Merge all experience banks from parallel environments.

        This method searches for experience bank files in the working directory,
        loads them, merges them into a single consolidated experience bank, and
        then removes the individual bank files to clean up the workspace.

        The merged experience bank is saved as "experience_bank.pkl" in the
        working directory.

        Note:
            Experience bank files are expected to follow the naming pattern
            "experience_bank_*.pkl" where * can be any string identifier.
        """
        import os

        from .experience import ExperienceBank

        # Find all experience bank files
        bank_files = []
        for filename in os.listdir(self.work_dir):
            if filename.startswith("experience_bank_") and filename.endswith(".pkl"):
                bank_files.append(os.path.join(self.work_dir, filename))

        if not bank_files:
            print("No experience banks found!")
            return

        print(f"Found {len(bank_files)} experience banks")

        # Create merged bank and load all files
        merged = ExperienceBank()
        total = 0

        for bank_file in bank_files:
            try:
                bank = ExperienceBank.load(bank_file)
                merged.merge_inplace(bank)
                total += len(bank)
                print(f"  Merged {len(bank)} from {os.path.basename(bank_file)}")
            except Exception as e:
                print(f"  Failed to load {bank_file}: {e}")

        # Clean up individual bank files
        for bank_file in bank_files:
            try:
                os.remove(bank_file)
            except Exception as e:
                print(f"  Failed to delete {bank_file}: {e}")

        # Save merged bank
        merged.save(self.work_dir, "experience_bank.pkl")
        print(f"  Saved merged bank with {total} total experiences")

    async def start(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        initial_config: LearnerConfig | None = None,
    ) -> AsyncIterator[IterationState]:
        """Start the learner and yield state at each iteration.

        This is the main entry point for running the learner. It returns an
        async iterator that yields IterationState at each iteration, giving
        the caller full control over the learning loop.

        Args:
            max_iter: Maximum number of iterations to run. If 0, runs until
                stop criterion is met (requires criterion_function to be set).
            skip_pre_loop: If True, skips the initial environment collection
                and update phases before the main learning loop.
            initial_config: Initial configuration object. Can be modified
                between iterations via set_next_config().

        Yields:
            IterationState containing current iteration info, metrics, and
            all registered state from tasks.

        Raises:
            ValueError: If environment functions or update function are not set.
            ValueError: If neither max_iter nor criterion_function is provided.
        """
        # Validation
        if not self.environment_functions or not self.update_function:
            raise ValueError("Environment and Update functions must be set!")

        if not max_iter and not self.criterion_function:
            raise ValueError("Either max_iter > 0 or criterion_function must be provided.")

        self._max_iter = max_iter if max_iter > 0 else None
        learner_config = initial_config

        learner_suffix = f" (Learner-{self.learner_id})" if self.learner_id is not None else ""
        print(f"Starting Parallel Experience RL Learner{learner_suffix}")

        update_task: Any = ()

        # Pre-loop phase: register and await env/update, but don't extract state yet
        # State extraction happens inside the loop after clear_state()
        env_results: list[Any] = []
        update_result: Any = None

        if not skip_pre_loop:
            # Pre-loop: collect experiences and update
            env_tasks = []
            for env_name, env_func in self.environment_functions.items():
                env_config = self._get_iteration_task_config(
                    env_func, learner_config, f"environment_{env_name}", 0
                )
                env_task = self._register_task(env_config)
                env_tasks.append(env_task)

            # Wait for all environment tasks (extract state later, inside loop)
            env_results = await asyncio.gather(*env_tasks, return_exceptions=True)

            self.merge_banks()

            update_config = self._get_iteration_task_config(
                self.update_function, learner_config, "update", 0
            )
            update_task = self._register_task(update_config)

            # Await update result (extract state later, inside loop)
            update_result = await update_task

        # Determine iteration range
        iteration_range: Iterator[int] | range
        if max_iter == 0:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main iteration loop
        for i in iteration_range:
            # Check for pending config update
            if self._pending_config is not None:
                learner_config = self._pending_config
                self._pending_config = None

            # Clear transient state from previous iteration
            self.clear_state()

            # Extract state from env/update results
            # (prepared in previous iteration or pre-loop)
            for env_result in env_results:
                if not isinstance(env_result, BaseException):
                    self._extract_state_from_result(env_result)
            if update_result is not None:
                self._extract_state_from_result(update_result)

            learner_prefix = f"[Learner-{self.learner_id}] " if self.learner_id is not None else ""
            print(f"{learner_prefix}Starting Iteration-{i}")

            # Check stop criterion if configured
            metric_value: float | None = None
            should_stop = False
            stop_task = None

            if self.criterion_function:
                criterion_config = self._get_iteration_task_config(
                    self.criterion_function, learner_config, "criterion", i
                )
                stop_task = self._register_task(criterion_config, deps=update_task)
                stop_result = await stop_task

                # Extract state from criterion result, excluding handled keys
                self._extract_state_from_result(
                    stop_result, exclude_keys={"metric_value", "should_stop"}
                )

                should_stop, metric_value = self._check_stop_criterion(stop_result)

            # Build iteration state
            self._iteration_state = self.build_iteration_state(
                iteration=i,
                metric_value=metric_value,
                should_stop=should_stop,
                current_config=learner_config,
            )

            # YIELD CONTROL TO CALLER
            yield self._iteration_state

            # Check if caller broke out or criterion met
            if should_stop:
                break

            # Collect experiences from parallel environments for next iteration
            next_config = self._pending_config or learner_config
            env_tasks = []
            for env_name, env_func in self.environment_functions.items():
                next_env_config = self._get_iteration_task_config(
                    env_func, next_config, f"environment_{env_name}", i + 1
                )
                env_task = self._register_task(next_env_config, deps=stop_task)
                env_tasks.append(env_task)

            # Wait for all environment tasks (extract state in next iteration)
            env_results = await asyncio.gather(*env_tasks, return_exceptions=True)
            if self.is_stopped:
                break

            # Merge all collected experiences
            self.merge_banks()

            # Prepare next iteration update with configuration
            next_update_config = self._get_iteration_task_config(
                self.update_function, next_config, "update", i + 1
            )
            update_task = self._register_task(next_update_config)

            # Await update result (extract state in next iteration)
            update_result = await update_task

            print(f"{learner_prefix}Finished Iteration-{i}")

    def set_next_config(self, config: LearnerConfig) -> None:
        """Set configuration for the next iteration.

        Args:
            config: Configuration to apply in the next iteration.
        """
        self._pending_config = config

    def get_current_state(self) -> IterationState | None:
        """Get the current iteration state."""
        return self._iteration_state

    def get_max_iterations(self) -> int | None:
        """Get the maximum iterations configured for current run."""
        return self._max_iter

    async def learn(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_config: LearnerConfig | None = None,
    ) -> IterationState | None:
        """Run parallel experience RL loop to completion.

        .. deprecated::
            Use :meth:`start` instead. This method will be removed in a future
            version. The `start()` method returns an async iterator giving you
            full control over each iteration.

        Args:
            max_iter: Maximum number of iterations to run.
            skip_pre_loop: If True, skips the initial environment collection
                and update phases.
            learner_config: Configuration for the learner.

        Returns:
            Final IterationState after completion, or None if no iterations ran.
        """
        warnings.warn(
            "learn() is deprecated and will be removed in a future version. "
            "Use start() instead which returns an async iterator for full control.",
            DeprecationWarning,
            stacklevel=2,
        )
        final_state = None
        async for state in self.start(
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            initial_config=learner_config,
        ):
            final_state = state
        return final_state


class ParallelReinforcementLearner(ReinforcementLearner):
    """Parallel reinforcement learner that runs multiple SequentialReinforcementLearners
    concurrently.

    This class orchestrates multiple SequentialReinforcementLearner instances to run in parallel,
    allowing for concurrent exploration of the learning space. Each learner can be configured
    independently through per-learner LearnerConfig objects.

    The parallel learner manages the lifecycle of all sequential learners and collects their results
    when all have completed their learning processes.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Parallel Reinforcement Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks
                across all parallel learners.
        """
        super().__init__(asyncflow, register_and_submit=False)

    def _create_sequential_learner(
        self, learner_id: int, config: LearnerConfig | None
    ) -> SequentialReinforcementLearner:
        """Create a SequentialReinforcementLearner instance for a parallel learner.

        Creates and configures a new SequentialReinforcementLearner with the same base
        functions as the parent parallel learner, but with a unique identifier
        for logging and debugging purposes.

        Args:
            learner_id: Unique identifier for the learner, used in logging
                and debugging output.
            config: Configuration object for this specific learner. Can be None
                to use default configuration.

        Returns:
            A fully configured SequentialReinforcementLearner instance ready to run
            independently in the parallel learning environment.
        """
        # Create a new sequential learner with the same asyncflow
        sequential_learner: SequentialReinforcementLearner = SequentialReinforcementLearner(
            self.asyncflow
        )

        # Copy the base functions from the parent learner
        sequential_learner.environment_function = self.environment_function
        sequential_learner.update_function = self.update_function
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
            parallel_config: Configuration object designed for parallel
            learner usage. Contains environment, update, test, and criterion
            parameters.

        Returns:
            Equivalent LearnerConfig suitable for use with
            SequentialReinforcementLearner,
            or None if input was None.
        """
        if parallel_config is None:
            return None

        # Create LearnerConfig with same parameters
        return LearnerConfig(
            environment=getattr(parallel_config, "environment", None),
            update=getattr(parallel_config, "update", None),
            criterion=getattr(parallel_config, "criterion", None),
        )

    async def start(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: list[LearnerConfig | None] | None = None,
    ) -> list[Any]:
        """Run parallel reinforcement learning by launching multiple
        SequentialReinforcementLearners.

        Orchestrates multiple SequentialReinforcementLearner instances to
        run concurrently, each with potentially different configurations. All
        learners run independently and their results are collected when all
        have completed.

        Args:
            parallel_learners: Number of parallel learners to run concurrently.
                Must be >= 2 (use SequentialReinforcementLearner directly for
                single learner).
            max_iter: Maximum number of iterations for each learner. If 0,
                learners run until their individual stop criteria are met.
            skip_pre_loop: If True, all learners skip their initial environment
                and update phases.
            learner_configs: list of configuration objects, one for each learner.
                If None, all learners use default configuration. Length must
                match parallel_learners if provided.

        Returns:
            list containing the final IterationState from each learner,
            in the same order as the learners were launched.

        Raises:
            ValueError: If parallel_learners < 2.
            ValueError: If required base functions are not set.
            ValueError: If neither max_iter nor criterion_function is provided.
            ValueError: If learner_configs length doesn't match parallel_learners.
        """
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialReinforcementLearner")

        # Validate base functions are set
        if not self.environment_function or not self.update_function:
            raise ValueError("Environment and Update functions must be set!")

        if not max_iter and not self.criterion_function:
            raise ValueError("Either max_iter > 0 or criterion_function must be provided.")

        # Prepare learner configurations
        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        print(f"Starting Parallel Reinforcement Learning with {parallel_learners} learners")

        async def rl_learner_workflow(learner_id: int) -> Any:
            """Run a single SequentialReinforcementLearner.

            Internal async function that manages the lifecycle of a single
            SequentialReinforcementLearner within the parallel learning context.

            Args:
                learner_id: Unique identifier for this learner instance.

            Returns:
                The final IterationState from the sequential learner.

            Raises:
                Exception: Re-raises any exception from the sequential learner
                    with additional context about which learner failed.
            """
            try:
                # Create and configure the sequential learner
                sequential_learner: SequentialReinforcementLearner = (
                    self._create_sequential_learner(learner_id, learner_configs[learner_id])
                )

                # Convert parallel config to sequential config
                sequential_config: LearnerConfig | None = self._convert_to_sequential_config(
                    learner_configs[learner_id]
                )

                # Run the sequential learner by iterating through start()
                final_state = None
                async for state in sequential_learner.start(
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    initial_config=sequential_config,
                ):
                    final_state = state
                    # Let the learner run to completion
                    if self.is_stopped:
                        sequential_learner.stop()

                # Store metrics per learner
                self.metric_values_per_iteration[f"learner-{learner_id}"] = (
                    sequential_learner.metric_values_per_iteration
                )

                return final_state
            except Exception as e:
                print(f"RLLearner-{learner_id}] failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures: list[Coroutine] = [rl_learner_workflow(i) for i in range(parallel_learners)]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*futures)

    async def learn(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: list[LearnerConfig | None] | None = None,
    ) -> list[Any]:
        """Run parallel reinforcement learning to completion.

        .. deprecated::
            Use :meth:`start` instead. This method will be removed in a future
            version.

        Args:
            parallel_learners: Number of parallel learners to run.
            max_iter: Maximum number of iterations for each learner.
            skip_pre_loop: If True, skips initial environment and update phases.
            learner_configs: Configuration objects for each learner.

        Returns:
            list containing the final IterationState from each learner.
        """
        warnings.warn(
            "learn() is deprecated and will be removed in a future version. Use start() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.start(
            parallel_learners=parallel_learners,
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            learner_configs=learner_configs,
        )
