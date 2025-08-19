import asyncio
import itertools
from collections.abc import Iterator
from typing import Any, List, Optional, Tuple, Union

from radical.asyncflow import WorkflowEngine

from ..learner import Learner, LearnerConfig, TaskConfig


class SequentialActiveLearner(Learner):
    """Sequential active learner that runs iterations one after another.
    
    This class implements a sequential active learning approach where each iteration
    consists of simulation, training, and active learning phases that run in sequence.
    The learner can be configured with per-iteration parameters through LearnerConfig.
    
    Attributes:
        learner_id (Optional[int]): Identifier for the learner, used for logging.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Sequential Active Learner.
        
        Args:
            asyncflow: The workflow engine instance used to manage async tasks.
        """
        super().__init__(asyncflow, register_and_submit=True)
        self.learner_id: Optional[int] = None

    async def teach(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_config: Optional[LearnerConfig] = None,
    ) -> Any:
        """Run sequential active learning with optional per-iteration configuration.
        
        Executes the active learning loop sequentially, with each iteration containing
        simulation, training, and active learning phases. Supports configurable
        stopping criteria and per-iteration parameter customization.
        
        Args:
            max_iter: Maximum number of iterations to run. If 0, runs until
                stop criterion is met (requires criterion_function to be set).
            skip_pre_loop: If True, skips the initial simulation and training
                phases before the main learning loop.
            learner_config: Configuration object containing per-iteration
                parameters for simulation, training, active learning, and
                criterion functions.
                
        Returns:
            The result of the learning process. Type depends on the specific
            implementation of the learning functions.
            
        Raises:
            Exception: If required functions (simulation_function, training_function,
                active_learn_function) are not set.
            Exception: If neither max_iter nor criterion_function is provided.
        """
        # Validate required functions
        if not self.simulation_function or not self.training_function or not self.active_learn_function:
            raise Exception("Simulation, Training, and Active Learning functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        learner_suffix: str = f' (Learner-{self.learner_id})' if self.learner_id is not None else ''
        print(f"Starting Active Learner{learner_suffix}")

        # Initialize tasks for pre-loop
        sim_task: Tuple = ()
        train_task: Tuple = ()

        if not skip_pre_loop:
            # Pre-loop: use iteration 0 configuration
            sim_config: TaskConfig = self._get_iteration_task_config(
                self.simulation_function, learner_config, 'simulation', 0
            )
            train_config: TaskConfig = self._get_iteration_task_config(
                self.training_function, learner_config, 'training', 0
            )

            sim_task = self._register_task(sim_config)
            train_task = self._register_task(train_config, deps=sim_task)

        # Determine iteration range
        iteration_range: Union[Iterator[int], range]
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main learning loop with per-iteration configuration
        for i in iteration_range:
            learner_prefix: str = f'[Learner-{self.learner_id}] ' if self.learner_id is not None else ''
            print(f'{learner_prefix}Starting Iteration-{i}')

            # Get iteration-specific configurations
            acl_config: TaskConfig = self._get_iteration_task_config(
                self.active_learn_function, learner_config, 'active_learn', i
            )

            acl_task: Any = self._register_task(acl_config, deps=(sim_task, train_task))

            # Check stop criterion if configured
            if self.criterion_function:
                criterion_config: TaskConfig = self._get_iteration_task_config(
                    self.criterion_function, learner_config, 'criterion', i
                )
                stop_task: Any = self._register_task(criterion_config, deps=acl_task)
                stop: Any = await stop_task

                should_stop: bool
                should_stop, _ = self._check_stop_criterion(stop)
                if should_stop:
                    break

            # Prepare next iteration tasks with iteration-specific configs
            next_sim_config: TaskConfig = self._get_iteration_task_config(
                self.simulation_function, learner_config, 'simulation', i + 1
            )
            next_train_config: TaskConfig = self._get_iteration_task_config(
                self.training_function, learner_config, 'training', i + 1
            )

            sim_task = self._register_task(next_sim_config, deps=acl_task)
            train_task = self._register_task(next_train_config, deps=sim_task)

            # Wait for training to complete
            await train_task


class ParallelActiveLearner(Learner):
    """Parallel active learner that runs multiple SequentialActiveLearners concurrently.
    
    This class orchestrates multiple SequentialActiveLearner instances to run in parallel,
    allowing for concurrent exploration of the learning space. Each learner can be
    configured independently through per-learner LearnerConfig objects.
    
    The parallel learner manages the lifecycle of all sequential learners and collects
    their results when all have completed their learning processes.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Parallel Active Learner.

        Args:
            asyncflow: The workflow engine instance used to manage async tasks
                across all parallel learners.
        """
        super().__init__(asyncflow, register_and_submit=False)

    def _create_sequential_learner(
        self,
        learner_id: int,
        config: Optional[LearnerConfig]
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
        self,
        parallel_config: Optional[LearnerConfig]
    ) -> Optional[LearnerConfig]:
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
            criterion=parallel_config.criterion
        )

    async def teach(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[List[Optional[LearnerConfig]]] = None,
    ) -> List[Any]:
        """Run parallel active learning by launching multiple SequentialActiveLearners.
        
        Orchestrates multiple SequentialActiveLearner instances to run concurrently,
        each with potentially different configurations. All learners run
        independently and their results are collected when all have completed.

        Args:
            parallel_learners: Number of parallel learners to run concurrently.
                Must be >= 2 (use SequentialActiveLearner directly for single learner).
            max_iter: Maximum number of iterations for each learner. If 0,
                learners run until their individual stop criteria are met.
            skip_pre_loop: If True, all learners skip their initial simulation
                and training phases.
            learner_configs: List of configuration objects, one for each learner.
                If None, all learners use default configuration. Length must
                match parallel_learners if provided.

        Returns:
            List containing the results from each learner, in the same order
            as the learners were launched. Result types depend on the specific
            implementation of the learning functions.

        Raises:
            ValueError: If parallel_learners < 2 (use SequentialActiveLearner instead).
            Exception: If required base functions are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            ValueError: If learner_configs length doesn't match parallel_learners.
        """
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialActiveLearner")

        # Validate base functions are set
        if not self.simulation_function or not self.training_function or not self.active_learn_function:
            raise Exception("Simulation, Training, and Active Learning functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        # Prepare learner configurations
        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        print(f"Starting Parallel Active Learning with {parallel_learners} learners")

        async def active_learner_workflow(learner_id: int) -> Any:
            """Run a single SequentialActiveLearner.
            
            Internal async function that manages the lifecycle of a single
            SequentialActiveLearner within the parallel learning context.
            
            Args:
                learner_id: Unique identifier for this learner instance.
                
            Returns:
                The result from the sequential learner's teach method.
                
            Raises:
                Exception: Re-raises any exception from the sequential learner
                    with additional context about which learner failed.
            """
            try:
                # Create and configure the sequential learner
                sequential_learner: SequentialActiveLearner = self._create_sequential_learner(
                    learner_id, learner_configs[learner_id]
                )

                # Convert parallel config to sequential config
                sequential_config: Optional[LearnerConfig] = self._convert_to_sequential_config(
                    learner_configs[learner_id]
                )

                # Run the sequential learner
                learner_result = await sequential_learner.teach(
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learner_config=sequential_config
                )

                self.metric_values_per_iteration[f'learner-{learner_id}'] = \
                     sequential_learner.metric_values_per_iteration

                return learner_result
            except Exception as e:
                print(f"ActiveLearner-{learner_id}] failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures: List[Any] = [active_learner_workflow(i) for i in range(parallel_learners)]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*[f for f in futures])
