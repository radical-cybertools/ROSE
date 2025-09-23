from functools import wraps
from typing import Any, Callable, Optional, Union

import typeguard
from pydantic import BaseModel
from radical.asyncflow import (
    WorkflowEngine,
)

from .metrics import LearningMetrics as Metrics


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


class LearnerConfig(BaseModel):
    """Base configuration class for active learners with per-iteration support.

    This class provides configuration management for different types of learning tasks
    across multiple iterations. Each task type can have either a single configuration
    applied to all iterations, or iteration-specific configurations.

    Attributes:
        simulation: Configuration for simulation tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
        training: Configuration for training tasks. Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
        prediction: Configuration for prediction tasks. Can be a single TaskConfig
        active_learn: Configuration for active learning tasks. Can be a single
            TaskConfig or a dictionary mapping iteration numbers to TaskConfig
            objects.
        criterion: Configuration for criterion tasks. Can be a single TaskConfig
        uncertainty: Configuration for uncertainty Quantification tasks. 
            Can be a single TaskConfig or a dictionary mapping iteration numbers 
            to TaskConfig objects.
    """

    simulation: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None
    training: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None
    prediction: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None
    active_learn: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None
    criterion: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None
    uncertainty: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None

    class Config:
        """Pydantic configuration for LearnerConfig."""

        extra = "forbid"
        json_encoders = {
            tuple: list,
        }

    def get_task_config(self, task_name: str, iteration: int) -> Optional[TaskConfig]:
        """Get the task configuration for a specific iteration.

        Args:
            task_name: Name of the task ('simulation',
            'training', 'prediction', 'active_learn', 'criterion', 'uncertainty).
            iteration: The iteration number (0-based).

        Returns:
            TaskConfig for the specified iteration, or None if not configured.

        Note:
            If a dictionary of configurations is provided, the method will first
            look for an exact iteration match, then fall back to default configs
            (key -1 or 'default').
        """
        task_config: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = getattr(
            self, task_name, None
        )
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
            return task_config.get(-1) or task_config.get("default")

        return None


class Learner:
    """Base class for active learning implementations.

    This class provides the foundational functionality for active learning workflows,
    including task registration, execution management, and configuration handling.

    Attributes:
        criterion_function: Configuration for criterion/stopping condition functions.
        uncertainty_function: Configuration for Uncertainty Quantification 
                              condition functions.
        training_function: Configuration for training functions.
        prediction_function: Configuration for prediction functions.
        simulation_function: Configuration for simulation functions.
        active_learn_function: Configuration for active learning functions.
        asyncflow: The workflow engine for managing asynchronous task execution.
        register_and_submit: Whether to automatically register and submit tasks.
        utility_task: Decorator for utility tasks.
        training_task: Decorator for training tasks.
        prediction_task: Decorator for prediction tasks.
        simulation_task: Decorator for simulation tasks.
        active_learn_task: Decorator for active learning tasks.
        uncertainty_task: Decorator for uncertainty quantification tasks.
    """

    @typeguard.typechecked
    def __init__(
        self, asyncflow: WorkflowEngine, register_and_submit: bool = True
    ) -> None:
        """Initialize the Learner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.
            register_and_submit: Whether to automatically register
            and submit decorated tasks.
        """
        self.criterion_function: dict[str, Any] = {}
        self.uncertainty_function: dict[str, Any] = {}
        self.training_function: dict[str, Any] = {}
        self.simulation_function: dict[str, Any] = {}
        self.active_learn_function: dict[str, Any] = {}
        self.prediction_function: dict[str, Any] = {}

        self.asyncflow: WorkflowEngine = asyncflow

        self.register_and_submit: bool = register_and_submit

        self.utility_task: Callable = self.register_decorator("utility")
        self.training_task: Callable = self.register_decorator("training")
        self.simulation_task: Callable = self.register_decorator("simulation")
        self.active_learn_task: Callable = self.register_decorator("active_learn")
        self.prediction_task: Callable = self.register_decorator("prediction")
        self.uncertainty_task: Callable = self.register_decorator("uncertainty")
        

        self.iteration: int = 0
        self.metric_values_per_iteration: dict[int, dict[str, float]] = {}
        self.uncertainty_values_per_iteration: dict[int, dict[str, float]] = {}

    def _get_iteration_task_config(
        self,
        base_task: dict[str, Any],
        config: Optional[LearnerConfig],
        task_key: str,
        iteration: int,
    ) -> dict[str, Any]:
        """Get task configuration for a specific iteration,
        merging base config with iteration-specific overrides."""

        # Start with a copy of the base task (or empty dict if None)
        task_config = base_task.copy() if base_task else {}

        # Ensure required keys exist with defaults
        task_config.setdefault("func", None)
        task_config.setdefault("args", ())
        task_config.setdefault("kwargs", {})
        task_config.setdefault("decor_kwargs", {})

        # Make a deep copy of decor_kwargs to avoid shared references
        if "decor_kwargs" in task_config and task_config["decor_kwargs"]:
            task_config["decor_kwargs"] = task_config["decor_kwargs"].copy()

        # Apply iteration-specific overrides if available
        if config:
            iter_config = config.get_task_config(task_key, iteration)
            if iter_config:
                if iter_config.args is not None:
                    task_config["args"] = iter_config.args
                if iter_config.kwargs is not None:
                    task_config["kwargs"] = iter_config.kwargs

        return task_config

    def create_iteration_schedule(
        self, task_name: str, schedule: dict[int, dict[str, Any]]
    ) -> dict[int, TaskConfig]:
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
                args=config.get("args", ()), kwargs=config.get("kwargs", {})
            )
            for iteration, config in schedule.items()
        }

    def create_adaptive_schedule(
        self, task_name: str, param_schedule: Callable[[int], dict[str, Any]]
    ) -> dict[int, TaskConfig]:
        """Helper method to create adaptive iteration schedules using a function.

        Args:
            task_name: Name of the task type.
            param_schedule: Function that takes iteration number
            and returns config dict.

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
                args=param_schedule(i).get("args", ()),
                kwargs=param_schedule(i).get("kwargs", {}),
            )
            for i in range(max_precompute)
        }

    def register_decorator(self, task_attr_name: str) -> Callable:
        """Decorator factory that registers a task function under a given name."""

        def decorator_factory(_func=None, **decor_kwargs) -> Callable:
            """Actual decorator returned to wrap the function."""

            def decorator(func: Callable) -> Callable:
                # Capture immutable values at decoration time
                decoration_as_executable = decor_kwargs.pop("as_executable", True)
                decoration_decor_kwargs = decor_kwargs.copy()

                # Store initial placeholder (so validation passes)
                base_task_obj: dict[str, Any] = {
                    "func": func,
                    "args": (),
                    "kwargs": {},
                    "decor_kwargs": decoration_decor_kwargs,
                    "as_executable": decoration_as_executable,
                }
                setattr(self, f"{task_attr_name}_function", base_task_obj)

                @wraps(func)
                def wrapper(*args, **kwargs) -> Any:
                    # Each call -> update the stored task object
                    task_obj = {
                        "func": func,
                        "args": args,
                        "kwargs": kwargs,
                        "decor_kwargs": decoration_decor_kwargs.copy(),
                        "as_executable": decoration_as_executable,
                    }

                    # overwrite the attribute so external consumers always see "latest"
                    setattr(self, f"{task_attr_name}_function", task_obj)

                    if self.register_and_submit:
                        return self._register_task(task_obj)

                    return func(*args, **kwargs)  # fallback: run locally

                return wrapper

            # Handle both @decorator and @decorator()
            if _func is not None:
                return decorator(_func)
            else:
                return decorator

        return decorator_factory

    @typeguard.typechecked
    def as_stop_criterion(
        self,
        metric_name: str,
        threshold: float,
        operator: str = "",
        as_executable: bool = True,
        **decor_kwargs,
    ) -> Callable:
        """Create a decorator for stop criterion functions."""

        def decorator(func: Callable) -> Callable:
            """Decorator that registers a stop criterion function."""

            # Capture immutable values at decoration time
            final_as_executable = decor_kwargs.pop("as_executable", as_executable)
            clean_decor_kwargs = decor_kwargs.copy()

            # Store initial config immediately (so validation passes)
            base_task_obj = {
                "func": func,
                "args": (),
                "kwargs": {},
                "decor_kwargs": clean_decor_kwargs,
                "as_executable": final_as_executable,
                "operator": operator,
                "threshold": threshold,
                "metric_name": metric_name,
            }
            self.criterion_function = base_task_obj

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> tuple[bool, float]:
                """Async wrapper that evaluates the stopping condition."""
                # Build fresh task object with runtime values
                task_obj = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "decor_kwargs": clean_decor_kwargs.copy(),
                    "as_executable": final_as_executable,
                    "operator": operator,
                    "threshold": threshold,
                    "metric_name": metric_name,
                }

                # Update so external callers always see "latest state"
                self.criterion_function = task_obj

                if self.register_and_submit:
                    # Submit and check the stop criterion
                    result = await self._register_task(task_obj)
                    return self._check_stop_criterion(result)

                # If not submitting immediately, evaluate locally
                metric_value = await func(*args, **kwargs)
                return self._check_stop_criterion(metric_value)

            return async_wrapper

        return decorator

    @typeguard.typechecked
    def uncertainty_quantification(
        self,
        uq_metric_name: str,
        query_size: float,
        threshold: float,
        operator: str = "",
        as_executable: bool = True,
        **decor_kwargs,
    ) -> Callable:
        """Create a decorator for uncertainty quantification functions."""

        def decorator(func: Callable) -> Callable:
            """Decorator that registers an uncertainty quantification function."""

            # Capture immutable values at decoration time
            final_as_executable = decor_kwargs.pop("as_executable", as_executable)
            clean_decor_kwargs = decor_kwargs.copy()

            # Store initial config immediately (so validation passes)
            base_task_obj = {
                "func": func,
                "args": (),
                "kwargs": {},
                "decor_kwargs": clean_decor_kwargs,
                "as_executable": final_as_executable,
                "operator": operator,
                "query_size": query_size,
                "threshold": threshold,
                "uq_metric_name": uq_metric_name,
            }
            self.uncertainty_function = base_task_obj

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> tuple[bool, float]:
                """Async wrapper that evaluates the stopping condition."""
                # Build fresh task object with runtime values
                task_obj = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "decor_kwargs": clean_decor_kwargs.copy(),
                    "as_executable": final_as_executable,
                    "operator": operator,
                    "query_size": query_size,
                    "threshold": threshold,
                    "uq_metric_name": uq_metric_name,
                }

                # Update so external callers always see "latest state"
                self.uncertainty_function = task_obj

                if self.register_and_submit:
                    # Submit and check the uncertainty quantification
                    result = await self._register_task(task_obj)
                    return self._check_uncertainty(result)

                # If not submitting immediately, evaluate locally
                metric_value = await func(*args, **kwargs)
                return self._check_uncertainty(metric_value)

            return async_wrapper

        return decorator
    
    def _register_task(
        self,
        task_obj: dict[str, Any],
        deps: Optional[Union[Any, tuple[Any, ...]]] = None,
    ) -> Any:
        """Register and submit a task for execution.

        Args:
            task_obj: Dictionary containing task configuration with
            'func', 'args', and 'kwargs'.
            deps: Optional dependencies. Can be a single dependency
            or tuple of dependencies.

        Returns:
            Task future object for the submitted task.
        """
        func: Callable = task_obj["func"]
        args: tuple[Any, ...] = task_obj["args"]

        # Ensure deps is added as a tuple
        if deps:
            if not isinstance(deps, tuple):  # Check if deps is not a tuple
                deps = (deps,)  # Wrap deps in a tuple if it's a single Task
            args += deps

        kwargs: dict[str, Any] = task_obj["kwargs"]
        decor_kwargs: dict[Any] = task_obj["decor_kwargs"]
        as_executable: bool = task_obj.get("as_executable", True)

        if as_executable:
            return self.asyncflow.executable_task(func, **decor_kwargs)(*args, **kwargs)
        else:
            return self.asyncflow.function_task(func, **decor_kwargs)(*args, **kwargs)

    def compare_metric(
        self,
        metric_name: str,
        metric_value: float,
        threshold: float,
        operator: str = "",
    ) -> bool:
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
        if not Metrics.is_supported_metric(metric_name):
            if not operator:
                raise ValueError(
                    "Operator value must be provided for "
                    f"custom metric '{metric_name}', "
                    "and must be one of the following: "
                    "LESS_THAN_THRESHOLD, GREATER_THAN_THRESHOLD, "
                    "EQUAL_TO_THRESHOLD, LESS_THAN_OR_EQUAL_TO_THRESHOLD, "
                    "GREATER_THAN_OR_EQUAL_TO_THRESHOLD."
                )

        # standard metric
        else:
            operator = Metrics.get_operator(metric_name)

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

    def _start_pre_loop(self) -> tuple[Any, Any]:
        """Start the initial step for active learning by defining and
           setting simulation and training tasks.

        Returns:
            tuple containing (simulation_task, training_task) futures.
        """
        sim_task: Any = self._register_task(self.simulation_function)
        train_task: Any = self._register_task(self.training_function, deps=sim_task)
        prediction_task: Any = self._register_task(self.prediction_function, 
                                                deps=train_task)
        return sim_task, train_task, prediction_task

    def _check_uncertainty(self, uncertainty_task_result: Any) -> tuple[bool, float]:
        """Check if the uncertainty criterion is met based on task result.

        Args:
            stop_task_result: Result from the criterion task,
            should be convertible to float.

        Returns:
            tuple of (should_stop: bool, metric_value: float).

        Raises:
            Exception: If the task result cannot be converted to a numerical value.
            TypeError: If the stop criterion task doesn't produce a numerical value.
        """
        try:
            uncertainty_value: float = float(uncertainty_task_result)
        except Exception as e:
            raise Exception(
                f"Failed to obtain a numerical value from criterion task: {e}"
            ) from e

        # check if the metric value is a number
        if isinstance(uncertainty_value, (float, int)):
            operator: str = self.uncertainty_function["operator"]
            threshold: float = self.uncertainty_function["threshold"]
            uq_metric_name: str = self.uncertainty_function["uq_metric_name"]

            self.uncertainty_values_per_iteration[self.iteration] = uncertainty_value
            self.iteration += 1

            if self.compare_metric(uq_metric_name, uncertainty_value, threshold, operator):
                print(
                    f"stop uncertainty metric: {uq_metric_name} "
                    f"is met with value of: {uncertainty_value} "
                    ". Breaking the active learning loop"
                )
                return True, uncertainty_value
            else:
                print(
                    f"uncertainty metric: {uq_metric_name} "
                    f"is not met yet ({uncertainty_value})."
                )
                return False, uncertainty_value
        else:
            raise TypeError(
                f"uncertainty task must produce a "
                f"numerical value, got {type(uncertainty_value)} instead"
            )
        
    def _check_stop_criterion(self, stop_task_result: Any) -> tuple[bool, float]:
        """Check if the stopping criterion is met based on task result.

        Args:
            stop_task_result: Result from the criterion task,
            should be convertible to float.

        Returns:
            tuple of (should_stop: bool, metric_value: float).

        Raises:
            Exception: If the task result cannot be converted to a numerical value.
            TypeError: If the stop criterion task doesn't produce a numerical value.
        """
        try:
            metric_value: float = float(stop_task_result)
        except Exception as e:
            raise Exception(
                f"Failed to obtain a numerical value from criterion task: {e}"
            ) from e

        # check if the metric value is a number
        if isinstance(metric_value, (float, int)):
            operator: str = self.criterion_function["operator"]
            threshold: float = self.criterion_function["threshold"]
            metric_name: str = self.criterion_function["metric_name"]

            self.metric_values_per_iteration[self.iteration] = metric_value
            self.iteration += 1

            if self.compare_metric(metric_name, metric_value, threshold, operator):
                print(
                    f"stop criterion metric: {metric_name} "
                    f"is met with value of: {metric_value} "
                    ". Breaking the active learning loop"
                )
                return True, metric_value
            else:
                print(
                    f"stop criterion metric: {metric_name} "
                    f"is not met yet ({metric_value})."
                )
                return False, metric_value
        else:
            raise TypeError(
                f"Stop criterion task must produce a "
                f"numerical value, got {type(metric_value)} instead"
            )

    def teach(self) -> None:
        """Teach method to be implemented by subclasses.
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "This is not supported, please define your "
            "teach method and invoke it directly"
        )

    def get_metric_results(self) -> list[float]:
        """Get the result of a task(s) by its name.

        Tasks might have similar names yet different future and task IDs.

        Args:
            task_name: Name of the task to retrieve results for.

        Returns:
            list of results from tasks with the matching name.

        Note:
            This method assumes the existence of a 'tasks' attribute that contains
            task information with 'future' and 'description' fields.
        """
        return self.metric_values_per_iteration

    def get_uncertainty_results(self) -> list[float]:
        """Get the uncertainty values from the learner.

        Returns:
            list of uncertainty values from the learner.
        """
        return self.uncertainty_values_per_iteration
    
    async def shutdown(self, *args, **kwargs) -> Any:
        """Shutdown the asyncflow workflow engine.

        Args:
            *args: Positional arguments to pass to asyncflow.shutdown().
            **kwargs: Keyword arguments to pass to asyncflow.shutdown().

        Returns:
            Result from asyncflow.shutdown().
        """
        return await self.asyncflow.shutdown(*args, **kwargs)
