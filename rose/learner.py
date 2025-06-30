import typeguard
import itertools
from typing import Callable, Dict, Any, Optional, List, Union
from functools import wraps
from pydantic import BaseModel


from .engine import ResourceEngine
from .engine import WorkflowEngine
from .metrics import ActiveLearningMetrics as metrics

class ActiveLearner(WorkflowEngine):

    @typeguard.typechecked
    def __init__(self, engine: ResourceEngine, register_and_submit: bool=True) -> None:

        self.criterion_function = {}
        self.training_function = {}
        self.simulation_function = {}
        self.active_learn_function = {}

        super().__init__(engine)

        self.register_and_submit = register_and_submit

        self.simulation_task = self.register_decorator('simulation')
        self.training_task = self.register_decorator('training')
        self.active_learn_task = self.register_decorator('active_learn')
        self.utility_task = self.register_decorator('utility')


    def register_decorator(self, task_attr_name: str):
        """
        Generic decorator factory for registering simulation/training/etc. tasks.
        """

        def decorator(func: Callable):
            # Set the base function reference at decoration time
            setattr(self, f"{task_attr_name}_function", {
                'func': func,
                'args': (),
                'kwargs': {},
            })

            @wraps(func)
            def wrapper(*args, **kwargs):
                task_obj = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                }
                setattr(self, f"{task_attr_name}_function", task_obj)

                if self.register_and_submit:
                    return self._register_task(task_obj)
            return wrapper

        return decorator

    @typeguard.typechecked
    def as_stop_criterion(self, metric_name: str,
                                threshold: float,
                                operator: str = ''):
        @typeguard.typechecked
        def decorator(func: Callable):
            # Register the function reference immediately
            self.criterion_function = {
                'func': func,
                'args': (),
                'kwargs': {},
                'operator': operator,
                'threshold': threshold,
                'metric_name': metric_name
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Update runtime args/kwargs
                self.criterion_function.update({
                    'args': args,
                    'kwargs': kwargs
                })

                if self.register_and_submit:
                    res = self._register_task(self.criterion_function).result()
                    return self._check_stop_criterion(res)
            return wrapper

        return decorator


    def _register_task(self, task_obj, deps=None):
        func = task_obj['func']
        args = task_obj['args']

        # Ensure deps is added as a tuple
        if deps:
            if not isinstance(deps, tuple):  # Check if deps is not a tuple
                deps = (deps,)  # Wrap deps in a tuple if it's a single Task
            args += deps

        kwargs = task_obj['kwargs']

        return super().__call__(func)(*args, **kwargs)

    def compare_metric(self, metric_name, metric_value, threshold, operator=''):
        """
        Compare a metric value against a threshold using a specified operator.
        
        Args:
            metric_name (str): Name of the metric to compare.
            metric_value (float): The value of the metric.
            threshold (float): The threshold to compare against.
            operator (str): The comparison operator. Supported values:
                - '<': metric_value < threshold
                - '>': metric_value > threshold
                - '==': metric_value == threshold
                - '<=': metric_value <= threshold
                - '>=': metric_value >= threshold
        
        Returns:
            bool: The result of the comparison.
        """
        # check for custom/user defined metric
        if not metrics.is_supported_metric(metric_name):
            if not operator:
                excp = f'Operator value must be provided for custom metric {metric_name}, '
                excp += 'and must be one of the following: LESS_THAN_THRESHOLD, GREATER_THAN_THRESHOLD, '
                excp += 'EQUAL_TO_THRESHOLD, LESS_THAN_OR_EQUAL_TO_THRESHOLD, GREATER_THAN_OR_EQUAL_TO_THRESHOLD'
                raise ValueError(excp)

        # standard metric
        else:
            operator = metrics.get_operator(metric_name)

        if operator == "<":
            return metric_value < threshold
        elif operator == ">":
            return metric_value > threshold
        elif operator == "==":
            return metric_value == threshold
        elif operator == "<=":
            return metric_value <= threshold
        elif operator == ">=":
            return metric_value >= threshold
        else:
            raise ValueError(f"Unknown comparison operator for metric {metric_name}")

    def _start_pre_loop(self):
        """
        start the initlial step for active learning by 
        defining and setting simulation and training tasks
        """

        sim_task = self._register_task(self.simulation_function)
        train_task = self._register_task(self.training_function, deps=sim_task)
        return sim_task, train_task

    def _check_stop_criterion(self, stop_task_result):

        try:
            metric_value = eval(stop_task_result)
        except Exception as e:
            raise Exception(f"Failed to obtain a numerical value from criterion task: {e}")

        # check if the metric value is a number
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            operator = self.criterion_function['operator']
            threshold = self.criterion_function['threshold']
            metric_name = self.criterion_function['metric_name']

            if self.compare_metric(metric_name, metric_value, threshold, operator):
                print(f'stop criterion metric: {metric_name} is met with value of: {metric_value}'\
                      '. Breaking the active learning loop')
                return True, metric_value
            else:
                print(f'stop criterion metric: {metric_name} is not met yet ({metric_value}).')
                return False, metric_value
        else:
            raise TypeError(f'Stop criterion task must produce a numerical value, got {type(metric_value)} instead')

    def teach(self, max_iter:int = 0):
        raise NotImplementedError('This is not supported, please define your teach method and invoke it directly')


    def get_result(self, task_name: str):
        '''
        Get the result of a task(s) by its name, tasks might have
        similar name yet different future and task IDs.
        '''
        tasks = [t['future'].result() 
                 for t in self.tasks.values() 
                 if t['description']['name'] == task_name]

        return tasks


class SequentialActiveLearner(ActiveLearner):
    '''
    SequentialActiveLearner is a subclass of ActiveLearner that implements
    a sequential active learning loop.

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
    def __init__(self, engine: ResourceEngine) -> None:
        '''
        Initialize the SequentialActiveLearner object.

        Args:
            engine: The ResourceEngine object that manages the resources and
            tasks submission to HPC resources during the active learning loop.
        '''
        super().__init__(engine, register_and_submit=False)

    def teach(self, max_iter:int = 0, skip_pre_loop:bool = False):
        '''
        Run the active learning loop for a specified number of iterations.

        Args:
            max_iter (int, optional): The maximum number of iterations for the
            active learning loop. If not provided, the value set during initialization
            will be used. Defaults to 0.
        '''
        # start the initial step for active learning by
        # defining and setting simulation and training tasks
        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_function:
            raise Exception("Simulation and Training function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        sim_task, train_task = (), ()

        if not skip_pre_loop:
            # step-1 invoke the pre_step only once
            sim_task, train_task = self._start_pre_loop()

        # if no max_iter is provided, run the loop indefinitely
        # and until the stop criterion is met
        if not max_iter:
            max_iter = itertools.count()

        else:
            max_iter = range(max_iter)

        # step-2 form the ACL loop and workflow
        for i in max_iter:
            print(f'Starting Iteration-{i}')
            acl_task = self._register_task(self.active_learn_function, deps=(sim_task, train_task))

            if self.criterion_function:
                stop_task = self._register_task(self.criterion_function, deps=acl_task)
                stop = stop_task.result()

                should_stop, _ = self._check_stop_criterion(stop)
                if should_stop:
                    break

            sim_task = self._register_task(self.simulation_function, deps=acl_task)
            train_task = self._register_task(self.training_function, deps=sim_task)

            # block/wait for each workflow until it finishes
            train_task.result()

class TaskConfig(BaseModel):
    """Configuration for a single task's arguments"""
    args: tuple = ()
    kwargs: dict = {}


class ParallelLearnerConfig(BaseModel):
    """Configuration for one parallel learner's tasks with per-iteration support."""
    simulation: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    training: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    active_learn: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    criterion: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None

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


class ParallelActiveLearner(ActiveLearner):
    """
    Parallel active learner that runs multiple learners concurrently.
    Configure each learner via `ParallelLearnerConfig` with per-iteration support.
    """

    def __init__(self, engine: ResourceEngine):
        super().__init__(engine, register_and_submit=False)

    def _get_iteration_task_config(self, base_task: Dict, config: Optional[ParallelLearnerConfig], 
                                 task_key: str, iteration: int) -> Dict:
        """
        Get task configuration for a specific iteration, merging base config with iteration-specific overrides.
        
        Args:
            base_task: Base task configuration from parent
            config: Learner-specific configuration
            task_key: Task type ('simulation', 'training', 'active_learn', 'criterion')
            iteration: Current iteration number
            
        Returns:
            Merged task configuration
        """
        # Start with base task configuration
        task_config = base_task.copy() if base_task else {"func": None, "args": (), "kwargs": {}}
        
        # Apply iteration-specific overrides if available
        if config:
            iter_config = config.get_task_config(task_key, iteration)
            if iter_config:
                task_config["args"] = iter_config.args or task_config.get("args", ())
                task_config["kwargs"] = iter_config.kwargs or task_config.get("kwargs", {})
                
        return task_config

    def teach(
        self,
        parallel_learners: int = 2,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[List[Optional[ParallelLearnerConfig]]] = None,
    ) -> List[Any]:
        if parallel_learners < 2:
            raise ValueError("For single learner, use SequentialActiveLearner")

        learner_configs = learner_configs or [None] * parallel_learners
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        def _run_learner_with_iteration_config(learner_id: int):
            """Run a single learner with per-iteration configuration support."""
            config = learner_configs[learner_id]
            
            # Validate required functions
            if not self.simulation_function or not self.training_function or not self.active_learn_function:
                raise Exception("Simulation, Training, and Active Learning functions must be set!")

            if not max_iter and not self.criterion_function:
                raise Exception("Either max_iter or stop_criterion_function must be provided.")

            print(f"Starting Learner-{learner_id}")
            
            # Initialize tasks for pre-loop
            sim_task, train_task = (), ()
            
            if not skip_pre_loop:
                # Pre-loop: use iteration 0 configuration
                sim_config = self._get_iteration_task_config(
                    self.simulation_function, config, 'simulation', 0
                )
                train_config = self._get_iteration_task_config(
                    self.training_function, config, 'training', 0
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
                print(f'[Learner-{learner_id}] Starting Iteration-{i}')
                
                # Get iteration-specific configurations
                acl_config = self._get_iteration_task_config(
                    self.active_learn_function, config, 'active_learn', i
                )
                
                acl_task = self._register_task(acl_config, deps=(sim_task, train_task))

                # Check stop criterion if configured
                if self.criterion_function:
                    criterion_config = self._get_iteration_task_config(
                        self.criterion_function, config, 'criterion', i
                    )
                    stop_task = self._register_task(criterion_config, deps=acl_task)
                    stop = stop_task.result()

                    should_stop, _ = self._check_stop_criterion(stop)
                    if should_stop:
                        break

                # Prepare next iteration tasks with iteration-specific configs
                next_sim_config = self._get_iteration_task_config(
                    self.simulation_function, config, 'simulation', i + 1
                )
                next_train_config = self._get_iteration_task_config(
                    self.training_function, config, 'training', i + 1
                )
                
                sim_task = self._register_task(next_sim_config, deps=acl_task)
                train_task = self._register_task(next_train_config, deps=sim_task)

                # Wait for training to complete
                train_task.result()

            print(f"Learner-{learner_id} completed")

        # Submit all learners asynchronously
        futures = [self.as_async(_run_learner_with_iteration_config)(i) for i in range(parallel_learners)]
        return [f.result() for f in futures]

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


class AlgorithmSelector(ActiveLearner):
    """
    AlgorithmSelector is a subclass of ActiveLearner that implements 
    multiple active learning pipelines in parallel, each pipeline is a 
    sequential active learning loop, and uses the same simulation and 
    training tasks, but distinct active learning task.
    """
    def __init__(self, engine: ResourceEngine) -> None:
        ''' 
        Initialize the AlgorithmSelector object.

        Args:
            engine: The ResourceEngine object that manages the resources and
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

    def teach_and_select(self, max_iter:int = 0, skip_pre_loop:bool = False):
        """
        Run the active learning pipelines in parallel, each using a different AL algorithm,
        for multiple iterations similar to SequentialActiveLearner.
        After that, select the best active learning algorithm

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

        def _parallel_active_learn(al_task, name):
            if not skip_pre_loop:
                sim_task, train_task = self._start_pre_loop()
            else:
                sim_task, train_task = (), ()

            if not max_iter:
                iteration_range = itertools.count()
            else:
                iteration_range = range(max_iter)

            stop_value = float('inf')
            num_iterations = 0

            for i in iteration_range:
                print(f'[Pipeline: {al_task["func"].__name__}] Starting Iteration-{i}')
                acl_task = self._register_task(al_task, deps=(sim_task, train_task))

                if self.criterion_function:
                    stop_task = self._register_task(self.criterion_function, deps=acl_task)
                    stop = stop_task.result()

                    should_stop, stop_value = self._check_stop_criterion(stop)
                    if should_stop:
                        num_iterations = i + 1
                        break

                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)
                # Wait for the training to complete before next iteration
                train_task.result()
                num_iterations = i + 1
            
            self.algorithm_results[name] = {
                'iterations': num_iterations,
                'last_result': stop_value
            }

        submitted_learners = []
        for name, al_task in self.active_learn_functions.items():
            async_teach = self.as_async(_parallel_active_learn)
            submitted_learners.append(async_teach(al_task, name))
            print(f'Pipeline-{name} is submitted for execution')

        # block/wait for each pipeline until it finishes
        [learner.result() for learner in submitted_learners]

        if self.algorithm_results:
            # Sort by (iterations, last_result)
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
