import typeguard
import itertools
from typing import Callable, Dict, Any, Optional, List, Union
from functools import wraps
from pydantic import BaseModel

from ..learner import Learner

from radical.asyncflow import ThreadExecutionBackend
from radical.asyncflow import RadicalExecutionBackend


class TaskConfig(BaseModel):
    """Configuration for a single task's arguments"""
    args: tuple = ()
    kwargs: dict = {}


class SequentialLearnerConfig(BaseModel):
    """Configuration for one sequential learner's tasks with per-iteration support."""
    simulation: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    training: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    active_learn: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    criterion: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    
    # Learner-specific metadata
    learner_id: Optional[int] = None
    learner_name: Optional[str] = None
    max_iter_override: Optional[int] = None

    class Config:
        extra = "forbid"
        json_encoders = {
            tuple: list,
        }

    def get_task_config(self, task_name: str, iteration: int) -> Optional[TaskConfig]:
        """
        Get the task configuration for a specific iteration.
        
        Args:
            task_name: Name of the task ('simulation', 'training', 'active_learn', 'criterion')
            iteration: The iteration number (0-based)
            
        Returns:
            TaskConfig for the specified iteration, or None if not configured
        """
        task_config = getattr(self, task_name, None)
        if task_config is None:
            return None
            
        # If it's a TaskConfig, return it directly (same config for all iterations)
        if isinstance(task_config, TaskConfig):
            return task_config
            
        # If it's a dict, look for iteration-specific config
        if isinstance(task_config, dict):
            # Try exact iteration match first
            if iteration in task_config:
                return task_config[iteration]
            # Fall back to default config (key -1 or 'default')
            return task_config.get(-1) or task_config.get('default')
            
        return None


class SequentialActiveLearner(Learner):
    '''
    SequentialActiveLearner is a subclass of Learner that implements
    a sequential active learning loop with per-iteration configuration support.

           Iteration 1:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration 2:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration 3:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration N
    '''
    def __init__(self, engine: Union[ThreadExecutionBackend, RadicalExecutionBackend], 
                 config: Optional[SequentialLearnerConfig] = None) -> None:
        '''
        Initialize the SequentialActiveLearner object.

        Args:
            engine: The execution backend that manages the resources and
            tasks submission to HPC resources during the active learning loop.
            config: Optional configuration for per-iteration task parameters.
        '''
        super().__init__(engine, register_and_submit=False)
        self.config = config

    def _get_iteration_task_config(self, base_task: Dict, task_key: str, iteration: int) -> Dict:
        """
        Get task configuration for a specific iteration, merging base config with iteration-specific overrides.
        
        Args:
            base_task: Base task configuration from parent
            task_key: Task type ('simulation', 'training', 'active_learn', 'criterion')
            iteration: Current iteration number
            
        Returns:
            Merged task configuration
        """
        # Start with base task configuration
        task_config = base_task.copy() if base_task else {"func": None, "args": (), "kwargs": {}}
        
        # Apply iteration-specific overrides if available
        if self.config:
            iter_config = self.config.get_task_config(task_key, iteration)
            if iter_config:
                # Use explicit None checks to allow intentional clearing with empty collections
                if iter_config.args is not None:
                    task_config["args"] = iter_config.args
                if iter_config.kwargs is not None:
                    task_config["kwargs"] = iter_config.kwargs
                    
        return task_config

    def teach(self, max_iter: int = 0, skip_pre_loop: bool = False, learner_id: int = 0):
        '''
        Run the active learning loop for a specified number of iterations.

        Args:
            max_iter (int, optional): The maximum number of iterations for the
            active learning loop. If not provided, the value set during initialization
            will be used. Defaults to 0.
            skip_pre_loop (bool, optional): If True, skip the initial pre-loop step.
            learner_id (int, optional): Identifier for this learner instance (used in logging).
        '''
        # Check for max_iter override in config
        if self.config and self.config.max_iter_override is not None:
            max_iter = self.config.max_iter_override
        
        # Validate required functions
        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_function:
            raise Exception("Simulation and Training function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        # Generate learner name for logging
        learner_name = f"Learner-{learner_id}"
        if self.config and self.config.learner_name:
            learner_name = self.config.learner_name
        elif self.config and self.config.learner_id is not None:
            learner_name = f"Learner-{self.config.learner_id}"

        print(f"Starting {learner_name}")

        sim_task, train_task = (), ()

        if not skip_pre_loop:
            # Pre-loop: use iteration 0 configuration
            sim_config = self._get_iteration_task_config(
                self.simulation_function, 'simulation', 0
            )
            train_config = self._get_iteration_task_config(
                self.training_function, 'training', 0
            )
            
            sim_task = self._register_task(sim_config)
            train_task = self._register_task(train_config, deps=sim_task)

        # Determine iteration range
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main active learning loop with per-iteration configuration
        for i in iteration_range:
            print(f'[{learner_name}] Starting Iteration-{i}')
            
            # Get iteration-specific configurations
            acl_config = self._get_iteration_task_config(
                self.active_learn_function, 'active_learn', i
            )
            
            acl_task = self._register_task(acl_config, deps=(sim_task, train_task))

            # Check stop criterion if configured
            if self.criterion_function:
                criterion_config = self._get_iteration_task_config(
                    self.criterion_function, 'criterion', i
                )
                stop_task = self._register_task(criterion_config, deps=acl_task)
                stop = stop_task.result()

                should_stop, _ = self._check_stop_criterion(stop)
                if should_stop:
                    break

            # Prepare next iteration tasks with iteration-specific configs
            next_sim_config = self._get_iteration_task_config(
                self.simulation_function, 'simulation', i + 1
            )
            next_train_config = self._get_iteration_task_config(
                self.training_function, 'training', i + 1
            )
            
            sim_task = self._register_task(next_sim_config, deps=acl_task)
            train_task = self._register_task(next_train_config, deps=sim_task)

            # Wait for training to complete
            train_task.result()

        print(f"{learner_name} completed")

    def create_iteration_schedule(self, task_name: str, schedule: Dict[int, Dict]) -> Dict[int, TaskConfig]:
        """
        Helper method to create iteration-specific configurations.
        
        Args:
            task_name: Name of the task type
            schedule: Dictionary mapping iteration numbers to args/kwargs
            
        Returns:
            Dictionary mapping iterations to TaskConfig objects
            
        Example:
            schedule = {
                0: {'args': (param1,), 'kwargs': {'lr': 0.01}},
                5: {'args': (param2,), 'kwargs': {'lr': 0.005}},
                -1: {'args': (default_param,), 'kwargs': {'lr': 0.001}}  # default
            }
        """
        return {
            iteration: TaskConfig(
                args=config.get('args', ()),
                kwargs=config.get('kwargs', {})
            )
            for iteration, config in schedule.items()
        }

    def create_adaptive_schedule(self, task_name: str, param_schedule: Callable[[int], Dict]) -> Dict[int, TaskConfig]:
        """
        Helper method to create adaptive iteration schedules using a function.
        
        Args:
            task_name: Name of the task type
            param_schedule: Function that takes iteration number and returns config dict
            
        Returns:
            Dictionary with computed TaskConfig for each iteration
            
        Example:
            def adaptive_lr(iteration):
                lr = 0.01 * (0.9 ** iteration)
                return {'kwargs': {'learning_rate': lr}}
            
            adaptive_config = learner.create_adaptive_schedule('training', adaptive_lr)
        """
        # For now, we'll pre-compute a reasonable range. In practice, you might
        # want to compute this dynamically or use a lazy evaluation approach.
        max_precompute = 100
        return {
            i: TaskConfig(
                args=param_schedule(i).get('args', ()),
                kwargs=param_schedule(i).get('kwargs', {})
            )
            for i in range(max_precompute)
        }


class ParallelActiveLearner(Learner):
    """
    Parallel active learner that runs multiple SequentialActiveLearner instances concurrently.
    Each learner can be configured independently via SequentialLearnerConfig.
    """

    def __init__(self, engine: Union[ThreadExecutionBackend, RadicalExecutionBackend]):
        super().__init__(engine, register_and_submit=False)

    def teach(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[List[Optional[SequentialLearnerConfig]]] = None,
    ) -> List[Any]:
        """
        Run multiple sequential active learners in parallel.
        
        Args:
            parallel_learners: Number of parallel learners to run
            max_iter: Maximum iterations per learner (can be overridden per learner)
            skip_pre_loop: Whether to skip the initial simulation+training step
            learner_configs: List of configurations for each learner
            
        Returns:
            List of results from each learner
        """
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialActiveLearner")

        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        def _run_single_learner(learner_id: int):
            """Run a single sequential learner with its specific configuration."""
            config = learner_configs[learner_id]

            # Create a SequentialActiveLearner with the specific config
            sequential_learner = SequentialActiveLearner(self.engine, config)

            # Copy the function references from parent
            sequential_learner.simulation_function = self.simulation_function
            sequential_learner.training_function = self.training_function
            sequential_learner.active_learn_function = self.active_learn_function
            sequential_learner.criterion_function = self.criterion_function

            # Run the sequential learner
            return sequential_learner.teach(max_iter, skip_pre_loop, learner_id)

        # Submit all learners asynchronously
        futures = [self.as_async(_run_single_learner)(i) for i in range(parallel_learners)]
        return [f.result() for f in futures]

    def create_iteration_schedule(self, task_name: str, schedule: Dict[int, Dict]) -> Dict[int, TaskConfig]:
        """
        Helper method to create iteration-specific configurations.
        This is a convenience method that delegates to SequentialActiveLearner.
        """
        # Create a temporary sequential learner to use its helper method
        temp_learner = SequentialActiveLearner(self.engine)
        return temp_learner.create_iteration_schedule(task_name, schedule)

    def create_adaptive_schedule(self, task_name: str, param_schedule: Callable[[int], Dict]) -> Dict[int, TaskConfig]:
        """
        Helper method to create adaptive iteration schedules using a function.
        This is a convenience method that delegates to SequentialActiveLearner.
        """
        # Create a temporary sequential learner to use its helper method
        temp_learner = SequentialActiveLearner(self.engine)
        return temp_learner.create_adaptive_schedule(task_name, param_schedule)


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
            config = SequentialLearnerConfig(learner_name=f"Pipeline-{name}")
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
