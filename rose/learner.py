import typeguard
import itertools

from abc import ABC, abstractmethod

from typing import Callable, Dict, Any, Optional, List, Union
from functools import wraps
from pydantic import BaseModel

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ThreadExecutionBackend
from radical.asyncflow import RadicalExecutionBackend

from .metrics import LearningMetrics as metrics


class TaskConfig(BaseModel):
    """Configuration for a single task."""
    args: tuple = ()
    kwargs: dict = {}
    
    class Config:
        extra = "forbid"
        json_encoders = {
            tuple: list,
        }


class LearnerConfig(BaseModel, ABC):
    """Base configuration class for active learners with per-iteration support."""
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

class Learner:

    @typeguard.typechecked
    def __init__(self, engine: Union[ThreadExecutionBackend,
                                     RadicalExecutionBackend],
                register_and_submit: bool=True) -> None:

        self.engine = engine
        self.criterion_function = {}
        self.training_function = {}
        self.simulation_function = {}
        self.active_learn_function = {}

        self.asyncflow = WorkflowEngine(self.engine)

        self.register_and_submit = register_and_submit

        self.utility_task = self.register_decorator('utility')
        self.training_task = self.register_decorator('training')
        self.simulation_task = self.register_decorator('simulation')
        self.active_learn_task = self.register_decorator('active_learn')

    def _get_iteration_task_config(self, base_task: Dict,
                                   config: Optional[LearnerConfig],
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
                # Use explicit None checks to allow intentional clearing with empty collections
                if iter_config.args is not None:
                    task_config["args"] = iter_config.args
                if iter_config.kwargs is not None:
                    task_config["kwargs"] = iter_config.kwargs
                    
        return task_config


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

        return self.asyncflow.executable_task(func)(*args, **kwargs)

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
    

    async def shutdown(self, *args, **kwargs):
        return await self.asyncflow.shutdown(*args, **kwargs)
