import asyncio
import typeguard
import itertools
from typing import Callable, Dict, Any, Optional, List, Union
from functools import wraps

from ..learner import Learner
from ..learner import TaskConfig
from ..learner import LearnerConfig

from radical.asyncflow import ThreadExecutionBackend
from radical.asyncflow import RadicalExecutionBackend


class SequentialActiveLearner(Learner):
    """
    Sequential active learner that runs iterations one after another.
    Configure via `LearnerConfig` with per-iteration support.
    """

    def __init__(self, engine: Union[ThreadExecutionBackend,
                                     RadicalExecutionBackend]):
        super().__init__(engine, register_and_submit=True)
        self.learner_id = None  # Set by ParallelActiveLearner for logging

    async def teach(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_config: Optional[LearnerConfig] = None,
    ) -> Any:
        """
        Run sequential active learning with optional per-iteration configuration.
        
        Args:
            max_iter: Maximum number of iterations (0 for infinite)
            skip_pre_loop: Whether to skip pre-loop simulation and training
            learner_config: Configuration for iteration-specific parameters
            
        Returns:
            Result of the learning process
        """
        # Validate required functions
        if not self.simulation_function or not self.training_function or not self.active_learn_function:
            raise Exception("Simulation, Training, and Active Learning functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        print(f"Starting Sequential Active Learner{f' (Learner-{self.learner_id})' if self.learner_id is not None else ''}")
        
        # Initialize tasks for pre-loop
        sim_task, train_task = (), ()
        
        if not skip_pre_loop:
            # Pre-loop: use iteration 0 configuration
            sim_config = self._get_iteration_task_config(
                self.simulation_function, learner_config, 'simulation', 0
            )
            train_config = self._get_iteration_task_config(
                self.training_function, learner_config, 'training', 0
            )
            
            sim_task = self._register_task(sim_config)
            train_task = self._register_task(train_config, deps=sim_task)

        # Determine iteration range
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main learning loop with per-iteration configuration
        for i in iteration_range:
            learner_prefix = f'[Learner-{self.learner_id}] ' if self.learner_id is not None else ''
            print(f'{learner_prefix}Starting Iteration-{i}')
            
            # Get iteration-specific configurations
            acl_config = self._get_iteration_task_config(
                self.active_learn_function, learner_config, 'active_learn', i
            )

            acl_task = self._register_task(acl_config, deps=(sim_task, train_task))

            # Check stop criterion if configured
            if self.criterion_function:
                criterion_config = self._get_iteration_task_config(
                    self.criterion_function, learner_config, 'criterion', i
                )
                stop_task = self._register_task(criterion_config, deps=acl_task)
                stop = await stop_task

                should_stop, _ = self._check_stop_criterion(stop)
                if should_stop:
                    break

            # Prepare next iteration tasks with iteration-specific configs
            next_sim_config = self._get_iteration_task_config(
                self.simulation_function, learner_config, 'simulation', i + 1
            )
            next_train_config = self._get_iteration_task_config(
                self.training_function, learner_config, 'training', i + 1
            )
            
            sim_task = self._register_task(next_sim_config, deps=acl_task)
            train_task = self._register_task(next_train_config, deps=sim_task)

            # Wait for training to complete
            await train_task


class ParallelActiveLearner(Learner):
    """
    Parallel active learner that runs multiple SequentialActiveLearners concurrently.
    Configure each learner via `LearnerConfig` with per-iteration support.
    """

    def __init__(self, engine: Union[ThreadExecutionBackend,
                                     RadicalExecutionBackend]):
        super().__init__(engine, register_and_submit=False)

    def _create_sequential_learner(self, learner_id: int, config: Optional[LearnerConfig]) -> SequentialActiveLearner:
        """
        Create a SequentialActiveLearner instance for a parallel learner.

        Args:
            learner_id: ID of the learner (for logging/debugging)
            config: Configuration for this specific learner

        Returns:
            Configured SequentialActiveLearner instance
        """
        # Create a new sequential learner with the same engine
        sequential_learner = SequentialActiveLearner(self.engine)

        # Copy the base functions from the parent learner
        sequential_learner.simulation_function = self.simulation_function
        sequential_learner.training_function = self.training_function
        sequential_learner.active_learn_function = self.active_learn_function
        sequential_learner.criterion_function = self.criterion_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_id = learner_id

        return sequential_learner

    def _convert_to_sequential_config(self, parallel_config: Optional[LearnerConfig]) -> Optional[LearnerConfig]:
        """
        Convert a LearnerConfig to a LearnerConfig.

        Args:
            parallel_config: Configuration for parallel learner

        Returns:
            Equivalent LearnerConfig
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
        """
        Run parallel active learning by launching multiple SequentialActiveLearners.

        Args:
            parallel_learners: Number of parallel learners to run
            max_iter: Maximum number of iterations (0 for infinite)
            skip_pre_loop: Whether to skip pre-loop simulation and training
            learner_configs: List of configurations for each learner

        Returns:
            List of results from each learner
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

        @self.asyncflow.block
        async def _run_sequential_learner(learner_id: int):
            """Run a single SequentialActiveLearner."""
            try:
                # Create and configure the sequential learner
                sequential_learner = self._create_sequential_learner(learner_id, learner_configs[learner_id])
                
                # Convert parallel config to sequential config
                sequential_config = self._convert_to_sequential_config(learner_configs[learner_id])
                
                print(f"[Parallel-Learner-{learner_id}] Starting sequential learning")

                # Run the sequential learner
                return await sequential_learner.teach(
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learner_config=sequential_config
                )
            except Exception as e:
                print(f"[Parallel-Learner-{learner_id}] Failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures = [_run_sequential_learner(i) for i in range(parallel_learners)]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*[f for f in futures])
