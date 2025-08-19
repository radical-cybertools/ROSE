import asyncio
import itertools
from collections.abc import Iterator
from functools import wraps
from typing import Any, Callable, Optional, Union

import typeguard
from radical.asyncflow import WorkflowEngine

from rose.learner import Learner, LearnerConfig, TaskConfig


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
    def __init__(
        self, asyncflow: WorkflowEngine,
        register_and_submit: bool = True
        ) -> None:
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
        self.learner_id: Optional[int] = None
        self.test_function = self.criterion_function

        # Create custom decorators that immediately register functions
        self.update_task: Callable = self.register_decorator('update')
        self.environment_task: Callable = self.register_decorator('environment')


class SequentialReinforcementLearner(ReinforcementLearner):
    """Sequential reinforcement learning implementation with configuration support.

    This class implements a sequential reinforcement learning loop where the learner
    interacts with the environment in a series of steps, updating its policy based
    on the rewards received from the environment. It supports per-iteration
    configuration through LearnerConfig. This approach is useful for implementing both
    on-policy (PPO, A2C) and off-policy (DQN) learning algorithms.

    The learning process follows this sequential pattern:

        Iteration 1:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration 2:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration 3:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration N:
        [Env] -> [Update] -> [Test]

    Each iteration consists of three sequential steps:
    1. Environment interaction to collect experiences
    2. Policy update based on collected experiences
    3. Testing/evaluation of the updated policy
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the SequentialReinforcementLearner.

        Args:
            asyncflow (WorkflowEngine): The workflow engine for
            managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=True)

    async def learn(self,
                   max_iter: int = 0,
                   skip_pre_loop: bool = False,
                   learner_config: Optional[LearnerConfig] = None) -> Any:
        """Run the sequential reinforcement learning loop with configuration support.

        Executes the reinforcement learning algorithm for a specified number of
        iterations. Each iteration performs environment interaction, policy update,
        and testing in sequence. The loop can be terminated early if stopping
        criteria are met. Supports per-iteration parameter customization.

        Args:
            max_iter (int, optional): The maximum number of iterations for the
                reinforcement learning loop. If 0 or not provided, runs indefinitely.
                Defaults to 0.
            skip_pre_loop (bool): If True, skips the initial environment and update
                phases before the main learning loop.
            learner_config (Optional[LearnerConfig]): Configuration object containing
                per-iteration parameters for environment, update, and test functions.

        Returns:
            The result of the learning process. Type depends on the specific
            implementation of the learning functions.

        Raises:
            Exception: If environment, update, or test functions are not set.
            Exception: If neither max_iter nor criterion_function is provided.
        """
        self.test_function = self.criterion_function
        # Validate that required functions are set
        if not self.environment_function or not self.update_function:
            raise Exception("Environment and Update function must be set!")

        if not self.test_function:
            raise Exception("Test function must be set!")

        if not max_iter and not self.criterion_function:
            mgs = "Either max_iter or stop_criterion_function must be provided."
            raise Exception(mgs)

        learner_suffix: str = (
            f' (Learner-{self.learner_id})'
            if self.learner_id is not None
            else ''
        )

        print(f"Starting Sequential RL Learner{learner_suffix}")


        # Initialize tasks for pre-loop
        env_task: tuple = ()
        update_task: tuple = ()

        if not skip_pre_loop:
            # Pre-loop: use iteration 0 configuration
            env_config: TaskConfig = self._get_iteration_task_config(
                self.environment_function, learner_config, 'environment', 0
            )
            update_config: TaskConfig = self._get_iteration_task_config(
                self.update_function, learner_config, 'update', 0
            )

            env_task = self._register_task(env_config)
            update_task = self._register_task(update_config, deps=env_task)

        # Setup iteration counter
        iteration_range: Union[Iterator[int], range]
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Execute the RL loop with per-iteration configuration
        for i in iteration_range:
            learner_prefix: str = (
                f'[Learner-{self.learner_id}] '
                if self.learner_id is not None
                else ''
            )

            print(f'{learner_prefix}Starting Iteration-{i}')

            # Get iteration-specific test configuration
            test_config: TaskConfig = self._get_iteration_task_config(
                self.test_function, learner_config, 'test', i
            )

            # Register test task with dependencies
            test_task = self._register_task(test_config, deps=(env_task, update_task))

            # Check stop criterion if configured
            if self.criterion_function:
                criterion_config: TaskConfig = self._get_iteration_task_config(
                    self.criterion_function, learner_config, 'criterion', i
                )
                stop_task = self._register_task(criterion_config, deps=test_task)
                stop_result = await stop_task

                should_stop, _ = self._check_stop_criterion(stop_result)
                if should_stop:
                    break

            # Prepare next iteration tasks with iteration-specific configs
            next_env_config: TaskConfig = self._get_iteration_task_config(
                self.environment_function, learner_config, 'environment', i + 1
            )
            next_update_config: TaskConfig = self._get_iteration_task_config(
                self.update_function, learner_config, 'update', i + 1
            )
            env_task = self._register_task(next_env_config, deps=test_task)
            update_task = self._register_task(next_update_config, deps=env_task)

            # Wait for update to complete
            await update_task


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
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the ParallelExperience learner.

        Args:
            asyncflow (WorkflowEngine): The workflow engine for
            managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=False)
        self.environment_functions: dict[str, dict] = {}
        self.work_dir = '.'

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
                    'func': func,
                    'args': args,
                    'kwargs': kwargs
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

    async def learn(self,
                   max_iter: int = 0,
                   skip_pre_loop: bool = False,
                   learner_config: Optional[LearnerConfig] = None) -> Any:
        """Run the parallel reinforcement learning loop with configuration support.

        Executes the reinforcement learning algorithm with parallel experience
        collection for a specified number of iterations. Each iteration runs
        multiple environments in parallel, merges their experiences, updates
        the policy, and tests performance. Supports per-iteration parameter
        customization.

        Args:
            max_iter (int, optional): The maximum number of iterations for the
                reinforcement learning loop. If 0 or not provided, runs indefinitely.
                Defaults to 0.
            skip_pre_loop (bool): If True, skips the initial environment collection
                and update phases before the main learning loop.
            learner_config (Optional[LearnerConfig]): Configuration object containing
                per-iteration parameters for environment, update, and test functions.

        Returns:
            The result of the learning process. Type depends on the specific
            implementation of the learning functions.

        Raises:
            Exception: If environment functions, update function, or test function
                are not properly configured.
            Exception: If neither max_iter nor criterion_function is provided.
        """
        self.test_function = self.criterion_function

        # Validate that required functions are set
        if not self.environment_functions or \
                not self.update_function or \
                not self.test_function:
            raise Exception("Environment, Update, and Test functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or "
            "stop_criterion_function must be provided.")

        learner_suffix: str = (
            f' (Learner-{self.learner_id})'
            if self.learner_id is not None
            else ''
        )

        print(f"Starting Parallel Experience RL Learner{learner_suffix}")


        # Initialize tasks for pre-loop
        update_task: tuple = ()

        if not skip_pre_loop:
            # Pre-loop: collect experiences and update
            env_tasks = []
            for env_name, env_func in self.environment_functions.items():
                env_config: TaskConfig = self._get_iteration_task_config(
                    env_func, learner_config, f'environment_{env_name}', 0
                )
                env_task = self._register_task(env_config)
                env_tasks.append(env_task)

            # Wait for all environment tasks to complete
            await asyncio.gather(*env_tasks, return_exceptions=True)
            self.merge_banks()

            update_config: TaskConfig = self._get_iteration_task_config(
                self.update_function, learner_config, 'update', 0
            )
            update_task = self._register_task(update_config)

        # Setup iteration counter
        iteration_range: Union[Iterator[int], range]
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Execute the parallel RL loop with per-iteration configuration
        for i in iteration_range:
            learner_prefix: str = (
                f'[Learner-{self.learner_id}] '
                if self.learner_id is not None
                else ''
            )
            print(f'{learner_prefix}Starting Iteration-{i}')

            # Get iteration-specific test configuration
            test_config: TaskConfig = self._get_iteration_task_config(
                self.test_function, learner_config, 'test', i
            )

            # Register test task
            test_task = self._register_task(test_config, deps=update_task)

            # Check stop criterion if configured
            if self.criterion_function:
                criterion_config: TaskConfig = self._get_iteration_task_config(
                    self.criterion_function, learner_config, 'criterion', i
                )
                stop_task = self._register_task(criterion_config, deps=test_task)
                stop_result = await stop_task

                should_stop, _ = self._check_stop_criterion(stop_result)
                if should_stop:
                    break

            # Collect experiences from parallel environments for next iteration
            env_tasks = []
            for env_name, env_func in self.environment_functions.items():
                next_env_config: TaskConfig = self._get_iteration_task_config(
                    env_func, learner_config, f'environment_{env_name}', i + 1
                )
                env_task = self._register_task(next_env_config, deps=test_task)
                env_tasks.append(env_task)

            # Wait for all environment tasks to complete
            await asyncio.gather(*env_tasks, return_exceptions=True)

            # Merge all collected experiences
            self.merge_banks()

            # Prepare next iteration update with configuration
            next_update_config: TaskConfig = self._get_iteration_task_config(
                self.update_function, learner_config, 'update', i + 1
            )
            update_task = self._register_task(next_update_config)

            # Wait for update to complete
            await update_task

            print(f'{learner_prefix}Finished Iteration-{i}')


class ParallelReinforcementLearner(ReinforcementLearner):
    """Parallel reinforcement learner that runs multiple
    SequentialReinforcementLearners concurrently.

    This class orchestrates multiple SequentialReinforcementLearner
    instances to run in parallel, allowing for concurrent exploration
    of the learning space. Each learner can be configured independently
    through per-learner LearnerConfig objects.

    The parallel learner manages the lifecycle of all sequential
    learners and collects their results when all have completed their
    learning processes.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Parallel Reinforcement Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks
                across all parallel learners.
        """
        super().__init__(asyncflow, register_and_submit=False)

    def _create_sequential_learner(
        self,
        learner_id: int,
        config: Optional[LearnerConfig]
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
        sequential_learner: SequentialReinforcementLearner = \
             SequentialReinforcementLearner(self.asyncflow)

        # Copy the base functions from the parent learner
        sequential_learner.environment_function = self.environment_function
        sequential_learner.update_function = self.update_function
        sequential_learner.test_function = self.test_function
        sequential_learner.criterion_function = self.criterion_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_id = learner_id

        return sequential_learner

    def _convert_to_sequential_config(
        self,
        parallel_config: Optional[LearnerConfig]
    ) -> Optional[LearnerConfig]:
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
            environment=getattr(parallel_config, 'environment', None),
            update=getattr(parallel_config, 'update', None),
            test=getattr(parallel_config, 'test', None),
            criterion=getattr(parallel_config, 'criterion', None)
        )

    async def learn(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[list[Optional[LearnerConfig]]] = None,
    ) -> list[Any]:
        """Run parallel reinforcement learning by launching multiple
           SequentialReinforcementLearners.

        Orchestrates multiple SequentialReinforcementLearner instances to
        run concurrently,
        each with potentially different configurations. All learners run
        independently and their results are collected when all have completed.

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
            list containing the results from each learner, in the same order
            as the learners were launched. Result types depend on the specific
            implementation of the learning functions.

        Raises:
            ValueError: If parallel_learners < 2
            (use SequentialReinforcementLearner instead).
            Exception: If required base functions are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            ValueError: If learner_configs length doesn't match parallel_learners.
        """
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialReinforcementLearner")

        # Validate base functions are set
        if not self.environment_function or \
            not self.update_function or \
                not self.test_function:
            raise Exception("Environment, Update, and Test functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or "
            "stop_criterion_function must be provided.")

        # Prepare learner configurations
        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        print(f"Starting Parallel Reinforcement Learning "
        f"with {parallel_learners} learners")

        async def rl_learner_workflow(learner_id: int) -> Any:
            """Run a single SequentialReinforcementLearner.

            Internal async function that manages the lifecycle of a single
            SequentialReinforcementLearner within the parallel learning context.

            Args:
                learner_id: Unique identifier for this learner instance.

            Returns:
                The result from the sequential learner's learn method.

            Raises:
                Exception: Re-raises any exception from the sequential learner
                    with additional context about which learner failed.
            """
            try:
                # Create and configure the sequential learner
                sequential_learner: SequentialReinforcementLearner = \
                    self._create_sequential_learner(
                    learner_id, learner_configs[learner_id]
                )

                # Convert parallel config to sequential config
                sequential_config: Optional[LearnerConfig] = \
                     self._convert_to_sequential_config(
                    learner_configs[learner_id]
                )

                # Run the sequential learner
                learner_result = await sequential_learner.learn(
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learner_config=sequential_config
                )

                # Store metrics per learner
                self.metric_values_per_iteration[f'learner-{learner_id}'] = \
                     sequential_learner.metric_values_per_iteration

                return learner_result
            except Exception as e:
                print(f"RLLearner-{learner_id}] failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures: list[Any] = [rl_learner_workflow(i) for i in range(parallel_learners)]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*[f for f in futures])
