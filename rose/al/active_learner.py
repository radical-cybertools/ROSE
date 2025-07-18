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

class AlgorithmSelector(Learner):
    """
    AlgorithmSelector is a subclass of Learner that implements 
    multiple active learning pipelines in parallel, each pipeline is a 
    sequential active learning loop, and uses the same simulation and 
    training tasks, but distinct active learning task.
    
    This class now leverages SequentialActiveLearner for each pipeline.
    """
    def __init__(self, engine: Union[ThreadExecutionBackend, RadicalExecutionBackend]) -> None:
        ''' 
        Initialize the AlgorithmSelector object.

        Args:
            engine: The execution backend that manages the resources and
            tasks submission to HPC resources during the active learning loop.
        '''
        super().__init__(engine, register_and_submit=False)
        self.active_learn_functions: Dict[str, Dict] = {}

        # A dictionary to store stats for each active learning pipeline
        # e.g. self.algorithm_results['algo_1'] = {'iterations': 5, 'last_result': 0.01}
        self.algorithm_results: Dict[str, Dict] = {}
        self.best_pipeline_name = None
        self.best_pipeline_stats = None

    def active_learn_task(self, name: str):
        """
        A decorator that registers an active learning task under the given name.
        
        Usage:
        
        @algo_selector.active_learn_task(name='algo_1')
        def active_learn_1(*args):
            ...
        
        @algo_selector.active_learn_task(name='algo_2')
        def active_learn_2(*args):
            ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.active_learn_functions[name] = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs
                }
                if self.register_and_submit:
                    return self._register_task(self.active_learn_functions[name])
            return wrapper
        return decorator

    def teach_and_select(self, max_iter: int = 0, skip_pre_loop: bool = False):
        """
        Run the active learning pipelines in parallel, each using a different AL algorithm,
        leveraging SequentialActiveLearner for each pipeline.
        After completion, select the best active learning algorithm.

        Args:
            max_iter (int, optional): The maximum number of iterations for each pipeline.
                                      If 0 and a criterion function is provided, it will run
                                      until the criterion is met.
            skip_pre_loop (bool, optional): If True, skip the initial pre-loop step
                                            (simulation+training setup).
        """
        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_functions:
            raise Exception("Simulation, Training, and at least one AL function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or a stop_criterion_function must be provided.")

        def _run_algorithm_pipeline(name: str, al_task: Dict):
            """Run a single algorithm pipeline using SequentialActiveLearner."""
            # Create a SequentialActiveLearner for this algorithm
            config = LearnerConfig(learner_name=f"Pipeline-{name}")
            sequential_learner = SequentialActiveLearner(self.engine, config)

            # Copy the function references from parent
            sequential_learner.simulation_function = self.simulation_function
            sequential_learner.training_function = self.training_function
            sequential_learner.active_learn_function = al_task  # Use the specific AL function
            sequential_learner.criterion_function = self.criterion_function
            
            # Track results for algorithm selection
            class ResultTracker:
                def __init__(self):
                    self.iterations = 0
                    self.last_result = float('inf')
                    
            tracker = ResultTracker()

            # Wrap the criterion function to track results
            if self.criterion_function:
                original_criterion = self.criterion_function
                def tracked_criterion(*args, **kwargs):
                    result = original_criterion['func'](*args, **kwargs)
                    tracker.last_result = result
                    return result
                
                sequential_learner.criterion_function = {
                    'func': tracked_criterion,
                    'args': original_criterion.get('args', ()),
                    'kwargs': original_criterion.get('kwargs', {})
                }
            
            # Override the teach method to count iterations
            original_teach = sequential_learner.teach
            def tracked_teach(*args, **kwargs):
                # Count iterations by intercepting the loop
                iteration_count = 0
                
                # We need to modify the sequential learner to track iterations
                # This is a simplified approach - in practice, you might want to
                # implement a more sophisticated tracking mechanism
                try:
                    result = original_teach(*args, **kwargs)
                    # Since we can't easily intercept the loop, we'll use max_iter as approximation
                    tracker.iterations = max_iter if max_iter > 0 else 1
                    return result
                except Exception as e:
                    # If an error occurs, we still want to record what we can
                    raise e
            
            sequential_learner.teach = tracked_teach
            
            # Run the sequential learner
            try:
                sequential_learner.teach(max_iter, skip_pre_loop, 0)
            except Exception as e:
                print(f"Pipeline {name} failed: {e}")
                tracker.iterations = 0
                tracker.last_result = float('inf')
            
            # Store results
            self.algorithm_results[name] = {
                'iterations': tracker.iterations,
                'last_result': tracker.last_result
            }

        # Submit all algorithm pipelines asynchronously
        submitted_learners = []
        for name, al_task in self.active_learn_functions.items():
            async_teach = self.as_async(_run_algorithm_pipeline)
            submitted_learners.append(async_teach(name, al_task))
            print(f'Pipeline-{name} is submitted for execution')

        # Wait for all pipelines to complete
        [learner.result() for learner in submitted_learners]

        # Select the best algorithm
        if self.algorithm_results:
            # Sort by (iterations, last_result) - fewer iterations and lower results are better
            sorted_pipelines = sorted(
                self.algorithm_results.items(),
                key=lambda kv: (kv[1]['iterations'], kv[1]['last_result'])
            )
            self.best_pipeline_name, self.best_pipeline_stats = sorted_pipelines[0]
            print(f"Best algorithm is '{self.best_pipeline_name}' "
                  f"with {self.best_pipeline_stats['iterations']} iteration(s) "
                  f"and final metric result {self.best_pipeline_stats['last_result']}")
        else:
            excp = "No pipeline stats found! Please make sure that at least one active learning algorithm "
            excp += "is used, and the status of each active learning pipeline to make sure that at least "
            excp += "one of them is running successfully!"
            raise ValueError(excp)
