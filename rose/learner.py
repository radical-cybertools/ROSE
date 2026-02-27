import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

import typeguard
from pydantic import BaseModel
from radical.asyncflow import (
    WorkflowEngine,
)

from .metrics import LearningMetrics as Metrics


@dataclass
class IterationState:
    """General-purpose state information yielded at each iteration.

    This class provides a standardized view of the learner's state at each
    iteration, enabling external decision makers (LLM agents) to observe
    progress and make decisions. All domain-specific state (e.g.,
    labeled_count, uncertainty) is stored in the `state` dictionary and can
    be accessed via attribute-style access.

    Attributes:
        iteration: Current iteration number (0-based).
        metric_name: Name of the criterion metric being tracked.
        metric_value: Current metric value (from criterion task).
        metric_threshold: Target threshold for stopping criterion.
        metric_history: List of all metric values from previous iterations.
        should_stop: Whether the stopping criterion suggests stopping.
        current_config: Current LearnerConfig being used.
        state: Dictionary containing all registered state from tasks.
            Supports attribute-style access (e.g., state.labeled_count).

    Example:
        Access registered state via attributes::

            async for state in learner.start(max_iter=10):
                # These access the 'state' dict automatically
                print(state.labeled_count)      # -> state.state['labeled_count']
                print(state.mean_uncertainty)   # -> state.state['mean_uncertainty']
                print(state.my_custom_value)    # -> state.state['my_custom_value']
    """

    iteration: int
    metric_name: str | None = None
    metric_value: float | None = None
    metric_threshold: float | None = None
    metric_history: list[float] = field(default_factory=list)
    should_stop: bool = False
    current_config: Optional["LearnerConfig"] = None

    # All domain-specific state goes here
    state: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to state dict.

        Args:
            name: Attribute name to look up.

        Returns:
            Value from state dict if found, None otherwise.
        """
        # Avoid infinite recursion for dataclass fields
        if name == "state":
            raise AttributeError(name)
        state_dict = object.__getattribute__(self, "state")
        if name in state_dict:
            return state_dict[name]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state dict with a default.

        Args:
            key: Key to look up in state.
            default: Default value if key not found.

        Returns:
            Value from state dict or default.
        """
        return self.state.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the iteration state.
        """
        result = {
            "iteration": self.iteration,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_threshold": self.metric_threshold,
            "metric_history": self.metric_history,
            "should_stop": self.should_stop,
        }
        # Merge in all state values
        result.update(self.state)
        return result


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
    """Base configuration class for learners with per-iteration support.

    This class provides configuration management for different types of learning tasks
    across multiple iterations. Each task type can have either a single configuration
    applied to all iterations, or iteration-specific configurations.

    Attributes:
        simulation: Configuration for simulation tasks (Active Learning).
        training: Configuration for training tasks (Active Learning).
        active_learn: Configuration for active learning tasks (Active Learning).
        environment: Configuration for environment tasks (Reinforcement Learning).
        update: Configuration for update tasks (Reinforcement Learning).
        criterion: Configuration for criterion/stopping tasks.
        prediction: Configuration for prediction tasks.
    """

    # Active Learning fields
    simulation: TaskConfig | dict[int, TaskConfig] | None = None
    training: TaskConfig | dict[int, TaskConfig] | None = None
    prediction: TaskConfig | dict[int, TaskConfig] | None = None
    active_learn: TaskConfig | dict[int, TaskConfig] | None = None
    # Reinforcement Learning fields
    environment: TaskConfig | dict[int, TaskConfig] | None = None
    update: TaskConfig | dict[int, TaskConfig] | None = None
    # Common fields
    criterion: TaskConfig | dict[int, TaskConfig] | None = None

    class Config:
        """Pydantic configuration for LearnerConfig."""

        extra = "forbid"
        json_encoders = {
            tuple: list,
        }

    def get_task_config(self, task_name: str, iteration: int) -> TaskConfig | None:
        """Get the task configuration for a specific iteration.

        Args:
            task_name: Name of the task ('simulation', 'training', 'prediction',
                'active_learn', 'environment', 'update', 'criterion').
            iteration: The iteration number (0-based).

        Returns:
            TaskConfig for the specified iteration, or None if not configured.

        Note:
            If a dictionary of configurations is provided, the method will first
            look for an exact iteration match, then fall back to default configs
            (key -1 or 'default').
        """
        task_config: TaskConfig | dict[int, TaskConfig] | None = getattr(self, task_name, None)
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
        Quantification condition functions.
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
    """

    @typeguard.typechecked
    def __init__(self, asyncflow: WorkflowEngine, register_and_submit: bool = True) -> None:
        """Initialize the Learner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.
            register_and_submit: Whether to automatically register
            and submit decorated tasks.
        """
        self.criterion_function: dict[str, Any] = {}

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

        self.iteration: int = 0
        self.metric_values_per_iteration: dict[int, dict[str, float]] = {}

        self._state_registry: dict[str, Any] = {}
        self._state_callbacks: list[Callable[[str, Any], None]] = []

        self._stop_event = asyncio.Event()

    @property
    def is_stopped(self) -> bool:
        """Check if the learner has been requested to stop."""
        return self._stop_event.is_set()

    def stop(self) -> None:
        """Signal the learner to stop execution as soon as possible."""
        self._stop_event.set()

    def _get_iteration_task_config(
        self,
        base_task: dict[str, Any],
        config: LearnerConfig | None,
        task_key: str,
        iteration: int,
    ) -> dict[str, Any]:
        """Get task configuration for a specific iteration, merging base config with iteration-
        specific overrides."""

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
            iteration: TaskConfig(args=config.get("args", ()), kwargs=config.get("kwargs", {}))
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

    def _register_task(
        self,
        task_obj: dict[str, Any],
        deps: Any | tuple[Any, ...] | None = None,
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
        """Start the initial step for active learning by defining and setting simulation and
        training tasks.

        Returns:
            tuple containing (simulation_task, training_task) futures.
        """
        sim_task: Any = self._register_task(self.simulation_function)
        train_task: Any = self._register_task(self.training_function, deps=sim_task)
        prediction_task: Any = self._register_task(
            self.prediction_function_function, deps=train_task
        )
        return sim_task, train_task, prediction_task

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
            raise Exception(f"Failed to obtain a numerical value from criterion task: {e}") from e

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
                print(f"stop criterion metric: {metric_name} is not met yet ({metric_value}).")
                return False, metric_value
        else:
            raise TypeError(
                f"Stop criterion task must produce a "
                f"numerical value, got {type(metric_value)} instead"
            )

    def start(self) -> None:
        """Start method to be implemented by subclasses.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "This is not supported, please define your Start method and invoke it directly"
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

    def register_state(self, key: str, value: Any) -> None:
        """Register a state value to be accessed by external entity.

        Called by user tasks to expose internal state (uncertainty scores,
        data counts, model info, etc.) to external entities or top components
        like agents.

        Args:
            key: State identifier (e.g., 'uncertainty_scores', 'labeled_count').
            value: State value to register.

        Example:
            In your active_learn task::

                @learner.active_learn_task
                async def active_learn(*args):
                    uncertainty = model.predict_uncertainty(X_unlabeled)
                    learner.register_state('uncertainty_scores', uncertainty)
                    learner.register_state(
                        'mean_uncertainty', float(uncertainty.mean())
                    )
                    return 'python active.py'
        """
        self._state_registry[key] = value
        for callback in self._state_callbacks:
            try:
                callback(key, value)
            except Exception:
                pass  # Don't let callback errors break the workflow

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a registered state value.

        Args:
            key: State identifier.
            default: Default value if key not found.

        Returns:
            Registered state value or default.

        Example::

            uncertainty = learner.get_state('uncertainty_scores')
            labeled_count = learner.get_state('labeled_count', default=0)
        """
        return self._state_registry.get(key, default)

    def get_all_state(self) -> dict[str, Any]:
        """Get all registered state values.

        Returns:
            Copy of the state registry dictionary.
        """
        return self._state_registry.copy()

    def clear_state(self) -> None:
        """Clear all registered state.

        Typically called between iterations to reset transient state.
        """
        self._state_registry.clear()

    def _extract_state_from_result(self, result: Any, exclude_keys: set[str] | None = None) -> None:
        """Extract state from task result if it's a dict.

        This method provides a universal way to extract state from task
        results across all learner types (Active Learning, RL, UQ).
        Each key-value pair in the dict is registered as state via
        register_state().

        Args:
            result: Task result. If dict, each key-value pair is registered
                as state. Non-dict results are ignored.
            exclude_keys: Optional set of keys to exclude from extraction.
                Useful for criterion results where certain keys (e.g.,
                'metric_value', 'should_stop') are handled separately.

        Example:
            In a learner implementation::

                train_result = await train_task
                self._extract_state_from_result(train_result)
                # If train_result is {"loss": 0.1, "accuracy": 0.95},
                # both values are now accessible via state.loss, state.accuracy

            With exclusions for criterion results::

                stop_result = await stop_task
                self._extract_state_from_result(
                    stop_result, exclude_keys={"metric_value", "should_stop"}
                )
        """
        if isinstance(result, dict):
            for k, v in result.items():
                if exclude_keys is None or k not in exclude_keys:
                    self.register_state(k, v)

    def on_state_update(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for state updates.

        The callback is invoked each time register_state() is called.

        Args:
            callback: Function called with (key, value) on each state update.

        Example::

            def my_callback(key, value):
                print(f"State updated: {key} = {value}")

            learner.on_state_update(my_callback)
        """
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Remove a previously registered state callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def build_iteration_state(
        self,
        iteration: int,
        metric_value: float | None = None,
        should_stop: bool = False,
        current_config: Optional["LearnerConfig"] = None,
    ) -> IterationState:
        """Build an IterationState from current learner state.

        Combines metric tracking with registered state to create a complete
        snapshot for user consumption.

        Args:
            iteration: Current iteration number.
            metric_value: Current metric value (if available).
            should_stop: Whether stopping criterion is met.
            current_config: Current configuration being used.

        Returns:
            IterationState populated with all available information.
        """
        # Get metric info from criterion function
        metric_name = None
        metric_threshold = None
        if self.criterion_function:
            metric_name = self.criterion_function.get("metric_name")
            metric_threshold = self.criterion_function.get("threshold")

        # Build metric history
        metric_history = list(self.metric_values_per_iteration.values())

        # Copy all registered state
        state = self.get_all_state()

        return IterationState(
            iteration=iteration,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_threshold=metric_threshold,
            metric_history=metric_history,
            should_stop=should_stop,
            current_config=current_config,
            state=state,
        )

    async def shutdown(self, *args, **kwargs) -> Any:
        """Shutdown the asyncflow workflow engine.

        Args:
            *args: Positional arguments to pass to asyncflow.shutdown().
            **kwargs: Keyword arguments to pass to asyncflow.shutdown().

        Returns:
            Result from asyncflow.shutdown().
        """
        return await self.asyncflow.shutdown(*args, **kwargs)
