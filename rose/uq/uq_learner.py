from functools import wraps
from typing import Any, Callable, Optional, Union

import typeguard
from radical.asyncflow import WorkflowEngine

from ..learner import Learner, LearnerConfig, TaskConfig


class UQLearnerConfig(LearnerConfig):
    """
    Attributes:
        uncertainty: Configuration for uncertainty quantification tasks.
            Can be a single TaskConfig
            or a dictionary mapping iteration numbers to TaskConfig objects.
    """
    uncertainty: Optional[Union[TaskConfig, dict[int, TaskConfig]]] = None

class UQLearner(Learner):
    """UQ active learner that runs iterations one after another.
    This class implements a sequential active learning approach based
    on Uncertainty Quantification.
    Each iteration consists of simulation, a set of training and prediction steps,
    and active learning phases executed in sequence.
    The learner can be configured with per-iteration parameters using UQLearnerConfig.

    Attributes:
        learner_name (Optional[str])    :   Identifier for the learner.
                                            Set by ParallelActiveLearner when
                                            used in parallel mode.
        prediction_task                 :   Decorator to register prediction task
        learner_results                 :   Learner training results.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the UQLearner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.

        """
        super().__init__(asyncflow, register_and_submit=False)

        self.uncertainty_function: dict[str, Any] = {}
        self.uncertainty_task: Callable = self.register_decorator("uncertainty")
        self.uncertainty_values_per_iteration: dict[int, dict[str, float]] = {}

        self.learner_name: str = "UQLearner"
        self.learner_results: list[dict[str, Any]] = []


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

            if self.compare_metric(
                uq_metric_name, uncertainty_value, threshold, operator
            ):
                print(
                    f"Stop uncertainty metric: {uq_metric_name} "
                    f"is met with value of: {uncertainty_value} "
                    ". Breaking the active learning loop"
                )
                return True, uncertainty_value
            else:
                print(
                    f"Uncertainty metric: {uq_metric_name} "
                    f"is not met yet ({uncertainty_value})."
                )
                return False, uncertainty_value
        else:
            raise TypeError(
                f"Uncertainty task must produce a "
                f"numerical value, got {type(uncertainty_value)} instead"
            )

    def get_uncertainty_results(self) -> list[float]:
        """Get the uncertainty values from the learner.

        Returns:
            list of uncertainty values from the learner.
        """
        return self.uncertainty_values_per_iteration
