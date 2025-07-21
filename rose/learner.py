import typeguard
import itertools

from abc import ABC, abstractmethod

from typing import Callable, Dict, Any, Optional, List, Union, Tuple, Type
from functools import wraps
from pydantic import BaseModel

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ThreadExecutionBackend
from radical.asyncflow import RadicalExecutionBackend

from .metrics import LearningMetrics as metrics


class TaskConfig(BaseModel):
    """Configuration for a single task.
    
    This class represents the configuration needed to execute a task,
    including its arguments and keyword arguments.
    
    Attributes:
        args: Positional arguments for the task.
        kwargs: Keyword arguments for the task.
    """
    args: tuple = ()
    kwargs: dict = {}
    
    class Config:
        """Pydantic configuration for TaskConfig."""
        extra = "forbid"
        json_encoders = {
            tuple: list,
        }


class LearnerConfig(BaseModel, ABC):
    """Base configuration class for active learners with per-iteration support.
    
    This class provides configuration management for different types of learning tasks
    across multiple iterations. Each task type can have either a single configuration
    applied to all iterations, or iteration-specific configurations.
    
    Attributes:
        simulation: Configuration for simulation tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
        training: Configuration for training tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
        active_learn: Configuration for active learning tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
        criterion: Configuration for criterion tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
    """
    simulation: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    training: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    active_learn: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None
    criterion: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = None

    class Config:
        """Pydantic configuration for LearnerConfig."""
        extra = "forbid"
        json_encoders = {
            tuple: list,
        }

    def get_task_config(self, task_name: str, iteration: int) -> Optional[TaskConfig]:
        """Get the task configuration for a specific iteration.
        
        Args:
            task_name: Name of the task ('simulation', 'training', 'active_learn', 'criterion').
            iteration: The iteration number (0-based).
            
        Returns:
            TaskConfig for the specified iteration, or None if not configured.
            
        Note:
            If a dictionary of configurations is provided, the method will first
            look for an exact iteration match, then fall back to default configs
            (key -1 or 'default').
        """
        task_config: Optional[Union[TaskConfig, Dict[int, TaskConfig]]] = getattr(self, task_name, None)
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


class Learner:
    """Base class for active learning implementations.
    
    This class provides the foundational functionality for active learning workflows,
    including task registration, execution management, and configuration handling.
    
    Attributes:
        criterion_function: Configuration for criterion/stopping condition functions.
        training_function: Configuration for training functions.
        simulation_function: Configuration for simulation functions.
        active_learn_function: Configuration for active learning functions.
        asyncflow: The workflow engine for managing asynchronous task execution.
        register_and_submit: Whether to automatically register and submit tasks.
        utility_task: Decorator for utility tasks.
        training_task: Decorator for training tasks.
        simulation_task: Decorator for simulation tasks.
        active_learn_task: Decorator for active learning tasks.
    """

    @typeguard.typechecked
    def __init__(self, asyncflow: WorkflowEngine,
                 register_and_submit: bool = True) -> None:
        """Initialize the Learner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.
            register_and_submit: Whether to automatically register and submit decorated tasks.
        """
        self.criterion_function: Dict[str, Any] = {}
        self.training_function: Dict[str, Any] = {}
        self.simulation_function: Dict[str, Any] = {}
        self.active_learn_function: Dict[str, Any] = {}

        self.asyncflow: WorkflowEngine = asyncflow

        self.register_and_submit: bool = register_and_submit

        self.utility_task: Callable = self.register_decorator('utility')
        self.training_task: Callable = self.register_decorator('training')
        self.simulation_task: Callable = self.register_decorator('simulation')
        self.active_learn_task: Callable = self.register_decorator('active_learn')

    def _get_iteration_task_config(self, base_task: Dict[str, Any],
                                   config: Optional[LearnerConfig],
                                   task_key: str, iteration: int) -> Dict[str, Any]:
        """Get task configuration for a specific iteration, merging base config with iteration-specific overrides.
        
        Args:
            base_task: Base task configuration from parent.
            config: Learner-specific configuration.
            task_key: Task type ('simulation', 'training', 'active_learn', 'criterion').
            iteration: Current iteration number.

        Returns:
            Merged task configuration dictionary containing 'func', 'args', and 'kwargs'.
        """
        # Start with base task configuration
        task_config: Dict[str, Any] = base_task.copy() if base_task else {
            "func": None, 
            "args": (), 
            "kwargs": {}
        }

        # Apply iteration-specific overrides if available
        if config:
            iter_config: Optional[TaskConfig] = config.get_task_config(task_key, iteration)
            if iter_config:
                # Use explicit None checks to allow intentional clearing with empty collections
                if iter_config.args is not None:
                    task_config["args"] = iter_config.args
                if iter_config.kwargs is not None:
                    task_config["kwargs"] = iter_config.kwargs
                    
        return task_config

    def create_iteration_schedule(self, task_name: str, schedule: Dict[int, Dict[str, Any]]) -> Dict[int, TaskConfig]:
        """Helper method to create iteration-specific configurations.
        
        Args:
            task_name: Name of the task type.
            schedule: Dictionary mapping iteration numbers to args/kwargs configuration.
            
        Returns:
            Dictionary mapping iterations to TaskConfig objects.
            
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

    def create_adaptive_schedule(self, task_name: str, param_schedule: Callable[[int], Dict[str, Any]]) -> Dict[int, TaskConfig]:
        """Helper method to create adaptive iteration schedules using a function.
        
        Args:
            task_name: Name of the task type.
            param_schedule: Function that takes iteration number and returns config dict.
            
        Returns:
            Dictionary with computed TaskConfig for each iteration.
            
        Example:
            def adaptive_lr(iteration):
                lr = 0.01 * (0.9 ** iteration)
                return {'kwargs': {'learning_rate': lr}}
            
            adaptive_config = learner.create_adaptive_schedule('training', adaptive_lr)
        """
        # For now, we'll pre-compute a reasonable range. In practice, you might
        # want to compute this dynamically or use a lazy evaluation approach.
        max_precompute: int = 100
        return {
            i: TaskConfig(
                args=param_schedule(i).get('args', ()),
                kwargs=param_schedule(i).get('kwargs', {})
            )
            for i in range(max_precompute)
        }

    def register_decorator(self, task_attr_name: str) -> Callable:
        """Generic decorator factory for registering simulation/training/etc. tasks.
        
        Args:
            task_attr_name: Name of the task attribute to set (e.g., 'training', 'simulation').
            
        Returns:
            Decorator function that can be used to register tasks.
        """
        def decorator(func: Callable) -> Callable:
            """Decorator that registers a task function.
            
            Args:
                func: The function to be decorated and registered.
                
            Returns:
                Wrapped function that can be called with runtime arguments.
            """
            # Set the base function reference at decoration time
            setattr(self, f"{task_attr_name}_function", {
                'func': func,
                'args': (),
                'kwargs': {},
            })

            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                """Wrapper function that updates task configuration and optionally submits the task.
                
                Args:
                    *args: Positional arguments for the task.
                    **kwargs: Keyword arguments for the task.
                    
                Returns:
                    Task result if register_and_submit is True, otherwise None.
                """
                task_obj: Dict[str, Any] = {
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
                          operator: str = '') -> Callable:
        """Create a decorator for stop criterion functions.
        
        Args:
            metric_name: Name of the metric to evaluate for stopping condition.
            threshold: Threshold value for comparison.
            operator: Comparison operator (optional for standard metrics).
            
        Returns:
            Decorator function for stop criterion tasks.
        """
        @typeguard.typechecked
        def decorator(func: Callable) -> Callable:
            """Decorator that registers a stop criterion function.
            
            Args:
                func: The criterion function to be decorated.
                
            Returns:
                Wrapped async function that evaluates the stopping condition.
            """
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
            async def wrapper(*args, **kwargs) -> Tuple[bool, float]:
                """Wrapper that evaluates the stopping condition.
                
                Args:
                    *args: Positional arguments for the criterion function.
                    **kwargs: Keyword arguments for the criterion function.
                    
                Returns:
                    Tuple of (should_stop: bool, metric_value: float).
                """
                # Update runtime args/kwargs
                self.criterion_function.update({
                    'args': args,
                    'kwargs': kwargs
                })

                if self.register_and_submit:
                    # await the result to process it
                    res: Any = await self._register_task(self.criterion_function)
                    return self._check_stop_criterion(res)
                    
            return wrapper

        return decorator

    def _register_task(self, task_obj: Dict[str, Any], deps: Optional[Union[Any, Tuple[Any, ...]]] = None) -> Any:
        """Register and submit a task for execution.
        
        Args:
            task_obj: Dictionary containing task configuration with 'func', 'args', and 'kwargs'.
            deps: Optional dependencies. Can be a single dependency or tuple of dependencies.
            
        Returns:
            Task future object for the submitted task.
        """
        func: Callable = task_obj['func']
        args: Tuple[Any, ...] = task_obj['args']

        # Ensure deps is added as a tuple
        if deps:
            if not isinstance(deps, tuple):  # Check if deps is not a tuple
                deps = (deps,)  # Wrap deps in a tuple if it's a single Task
            args += deps

        kwargs: Dict[str, Any] = task_obj['kwargs']

        return self.asyncflow.executable_task(func)(*args, **kwargs)

    def compare_metric(self, metric_name: str, metric_value: float, threshold: float, operator: str = '') -> bool:
        """Compare a metric value against a threshold using a specified operator.
        
        Args:
            metric_name: Name of the metric to compare.
            metric_value: The value of the metric.
            threshold: The threshold to compare against.
            operator: The comparison operator. Supported values:
                - '<': metric_value < threshold
                - '>': metric_value > threshold
                - '==': metric_value == threshold
                - '<=': metric_value <= threshold
                - '>=': metric_value >= threshold
        
        Returns:
            The result of the comparison.
            
        Raises:
            ValueError: If operator is not provided for custom metrics or if
                operator is not recognized.
        """
        # check for custom/user defined metric
        if not metrics.is_supported_metric(metric_name):
            if not operator:
                excp: str = f'Operator value must be provided for custom metric {metric_name}, '
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

    def _start_pre_loop(self) -> Tuple[Any, Any]:
        """Start the initial step for active learning by defining and setting simulation and training tasks.
        
        Returns:
            Tuple containing (simulation_task, training_task) futures.
        """
        sim_task: Any = self._register_task(self.simulation_function)
        train_task: Any = self._register_task(self.training_function, deps=sim_task)
        return sim_task, train_task

    def _check_stop_criterion(self, stop_task_result: Any) -> Tuple[bool, float]:
        """Check if the stopping criterion is met based on task result.
        
        Args:
            stop_task_result: Result from the criterion task, should be convertible to float.
            
        Returns:
            Tuple of (should_stop: bool, metric_value: float).
            
        Raises:
            Exception: If the task result cannot be converted to a numerical value.
            TypeError: If the stop criterion task doesn't produce a numerical value.
        """
        try:
            metric_value: float = float(stop_task_result)
        except Exception as e:
            raise Exception(f"Failed to obtain a numerical value from criterion task: {e}")

        # check if the metric value is a number
        if isinstance(metric_value, (float, int)):
            operator: str = self.criterion_function['operator']
            threshold: float = self.criterion_function['threshold']
            metric_name: str = self.criterion_function['metric_name']

            if self.compare_metric(metric_name, metric_value, threshold, operator):
                print(f'stop criterion metric: {metric_name} is met with value of: {metric_value}'
                      '. Breaking the active learning loop')
                return True, metric_value
            else:
                print(f'stop criterion metric: {metric_name} is not met yet ({metric_value}).')
                return False, metric_value
        else:
            raise TypeError(f'Stop criterion task must produce a numerical value, got {type(metric_value)} instead')

    def teach(self, max_iter: int = 0) -> None:
        """Train the model using active learning.
        
        Args:
            max_iter: Maximum number of iterations to run.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError('This is not supported, please define your teach method and invoke it directly')

    def get_result(self, task_name: str) -> List[Any]:
        """Get the result of a task(s) by its name.
        
        Tasks might have similar names yet different future and task IDs.
        
        Args:
            task_name: Name of the task to retrieve results for.
            
        Returns:
            List of results from tasks with the matching name.
            
        Note:
            This method assumes the existence of a 'tasks' attribute that contains
            task information with 'future' and 'description' fields.
        """
        tasks: List[Any] = [t['future'].result() 
                           for t in self.tasks.values() 
                           if t['description']['name'] == task_name]

        return tasks

    async def shutdown(self, *args, **kwargs) -> Any:
        """Shutdown the asyncflow workflow engine.
        
        Args:
            *args: Positional arguments to pass to asyncflow.shutdown().
            **kwargs: Keyword arguments to pass to asyncflow.shutdown().
            
        Returns:
            Result from asyncflow.shutdown().
        """
        return await self.asyncflow.shutdown(*args, **kwargs)
