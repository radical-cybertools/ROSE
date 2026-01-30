import asyncio
import itertools
import logging
from collections.abc import Iterator
from typing import Any, Callable, Optional, Union

from radical.asyncflow import WorkflowEngine

from ..learner import Learner, LearnerConfig, TaskConfig
from .active_learner import SequentialActiveLearner

logger = logging.getLogger(__name__)


class AlgorithmSelector(Learner):
    """AlgorithmSelector runs multiple active learning algorithms in parallel.

    This class runs multiple active learning algorithms in parallel, each as a
    separate sequential active learning loop, and selects the best performing
    algorithm based on iterations and final metric results.

    Now utilizes ParallelActiveLearner for better parallel execution management.

    Attributes:
        active_learn_functions: Dictionary mapping algorithm names to their functions.
        algorithm_results: Dictionary storing stats for each active learning pipeline.
        best_pipeline_name: Name of the best performing algorithm.
        best_pipeline_stats: Statistics of the best performing algorithm.
        active_learn_task: Reference to the algorithm active learn task decorator.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the AlgorithmSelector.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.
        """
        super().__init__(asyncflow, register_and_submit=False)

        # Override the parent's active_learn decorator with our named version
        self.active_learn_functions: dict[str, dict[str, Any]] = {}

        # A dictionary to store stats for each active learning pipeline
        # e.g. self.algorithm_results['algo_1'] = \
        # {'iterations': 5, 'last_result': 0.01}
        self.algorithm_results: dict[str, dict[str, Any]] = {}
        self.best_pipeline_name: Optional[str] = None
        self.best_pipeline_stats: Optional[dict[str, Any]] = None

        self.active_learn_task: Callable[[str], Callable] = (
            self._algorithm_active_learn_task
        )

    def _algorithm_active_learn_task(
        self, **decor_kwargs
    ) -> Callable[[Callable], Callable]:
        """Create a decorator for registering active learning algorithms.

        Args:
            **decor_kwargs: Decorator keyword arguments including
            'name' and others like 'as_executable'

        Returns:
            A decorator function that registers the algorithm function.
        """

        def decorator(func: Callable) -> Callable:
            """Decorator that registers an active learning function.

            Args:
                func: The active learning function to register.

            Returns:
                The original function unchanged.
            """
            # Extract name from decor_kwargs
            name = decor_kwargs.pop("name", func.__name__)

            # Register the function with decorator kwargs
            self.active_learn_functions[name] = {
                "func": func,
                "args": (),
                "kwargs": {},
                "decor_kwargs": decor_kwargs,
            }
            return func

        return decorator

    def _create_algorithm_learner(
        self,
        algorithm_name: str,
        algorithm_func: Callable,
        config: Optional[LearnerConfig],
    ) -> "SequentialActiveLearner":
        """Create a SequentialActiveLearner instance for a specific algorithm.

        Args:
            algorithm_name: Name of the algorithm.
            algorithm_func: The active learning function for this algorithm.
            config: Configuration for this specific algorithm.

        Returns:
            Configured SequentialActiveLearner instance.
        """
        # Import here to avoid circular imports
        from rose.al import SequentialActiveLearner

        # Create a new sequential learner with the same asyncflow
        sequential_learner: SequentialActiveLearner = SequentialActiveLearner(
            self.asyncflow
        )

        # Copy the base functions from the parent learner
        sequential_learner.simulation_function = self.simulation_function
        sequential_learner.training_function = self.training_function
        sequential_learner.active_learn_function = algorithm_func
        sequential_learner.criterion_function = self.criterion_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_id = algorithm_name

        return sequential_learner

    async def start(
        self,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        algorithm_configs: Optional[dict[str, LearnerConfig]] = None,
    ) -> dict[str, Any]:
        """Run multiple active learning algorithms in parallel and select the best.

        Runs multiple active learning algorithms in parallel, each using a different
        AL algorithm, for multiple iterations. After completion, selects the best
        active learning algorithm based on performance metrics.

        Args:
            max_iter: Maximum number of iterations for each pipeline. 0 for infinite.
            skip_pre_loop: Whether to skip pre-loop simulation and training.
            algorithm_configs: Dictionary mapping algorithm names to
            their configurations.

        Returns:
            Dictionary containing results from all algorithms and best algorithm info.
            Contains keys: 'algorithm_results', 'best_algorithm', 'best_stats'.

        Raises:
            Exception: If required functions are not set or no algorithms are
            registered.
            ValueError: If no algorithms complete successfully.
        """
        # Validate required functions
        if (
            not self.simulation_function
            or not self.training_function
            or not self.active_learn_functions
        ):
            raise Exception(
                "Simulation, Training, and at least one  AL function must be set!"
            )

        if not max_iter and not self.criterion_function:
            raise Exception(
                "Either max_iter or stop_criterion_function must be provided."
            )

        if not self.active_learn_functions:
            raise Exception(
                "No active learning algorithms registered! "
                "Use @active_learn_task decorator."
            )

        # Initialize algorithm configs if not provided
        algorithm_configs = algorithm_configs or {}

        logger.info(
            f"Starting algorithm selection with "
            f"{len(self.active_learn_functions)} algorithms: "
            f"{list(self.active_learn_functions.keys())}"
        )

        @self.asyncflow.block
        async def _run_algorithm_pipeline(
            al_name: str, al_task: dict
        ) -> dict[str, Any]:
            """Run a single algorithm pipeline.

            Args:
                algorithm_name: Name of the algorithm to run.
                algorithm_func: The active learning function for this algorithm.

            Returns:
                Dictionary containing the algorithm's results including
                iterations and final result.

            Raises:
                Exception: If the algorithm pipeline fails during execution.
            """
            try:
                algorithm_name = al_name
                algorithm_func = al_task["func"]

                # Create and configure the sequential learner for this algorithm
                algorithm_config: Optional[LearnerConfig] = algorithm_configs.get(
                    algorithm_name, None
                )
                sequential_learner: SequentialActiveLearner = (
                    self._create_algorithm_learner(
                        algorithm_name, algorithm_func, algorithm_config
                    )
                )

                logger.info(f"[Algorithm-{algorithm_name}] Starting pipeline")

                # Track iterations and results for this algorithm
                iteration_count: int = 0
                final_result: float = float("inf")

                # Initialize tasks for pre-loop
                sim_task: tuple = ()
                train_task: tuple = ()

                if not skip_pre_loop:
                    # Pre-loop: use iteration 0 configuration
                    sim_config: TaskConfig = (
                        sequential_learner._get_iteration_task_config(
                            sequential_learner.simulation_function,
                            algorithm_config,
                            "simulation",
                            0,
                        )
                    )
                    train_config: TaskConfig = (
                        sequential_learner._get_iteration_task_config(
                            sequential_learner.training_function,
                            algorithm_config,
                            "training",
                            0,
                        )
                    )

                    sim_task = sequential_learner._register_task(sim_config)
                    train_task = sequential_learner._register_task(
                        train_config, deps=sim_task
                    )

                # Determine iteration range
                iteration_range: Union[Iterator[int], range]
                if not max_iter:
                    iteration_range = itertools.count()
                else:
                    iteration_range = range(max_iter)

                # Main learning loop
                for i in iteration_range:
                    logger.info(f"[Algorithm-{algorithm_name}] Starting Iteration-{i}")

                    # Get iteration-specific configurations
                    acl_config: TaskConfig = (
                        sequential_learner._get_iteration_task_config(
                            al_task, algorithm_config, "active_learn", i
                        )
                    )

                    acl_task = sequential_learner._register_task(
                        acl_config, deps=(sim_task, train_task)
                    )

                    # Check stop criterion if configured
                    if sequential_learner.criterion_function:
                        criterion_config: TaskConfig = (
                            sequential_learner._get_iteration_task_config(
                                sequential_learner.criterion_function,
                                algorithm_config,
                                "criterion",
                                i,
                            )
                        )
                        stop_task = sequential_learner._register_task(
                            criterion_config, deps=acl_task
                        )
                        stop: Any = await stop_task

                        should_stop: bool
                        stop_value: float
                        should_stop, stop_value = (
                            sequential_learner._check_stop_criterion(stop)
                        )
                        final_result = stop_value
                        iteration_count = i + 1

                        if should_stop:
                            logger.info(
                                f"[Algorithm-{algorithm_name}] Stop criterion met "
                                f"with value of: {final_result}. "
                                "Breaking the active learning loop."
                            )
                            break

                    # Prepare next iteration tasks
                    next_sim_config: TaskConfig = (
                        sequential_learner._get_iteration_task_config(
                            sequential_learner.simulation_function,
                            algorithm_config,
                            "simulation",
                            i + 1,
                        )
                    )
                    next_train_config: TaskConfig = (
                        sequential_learner._get_iteration_task_config(
                            sequential_learner.training_function,
                            algorithm_config,
                            "training",
                            i + 1,
                        )
                    )

                    sim_task = sequential_learner._register_task(
                        next_sim_config, deps=acl_task
                    )
                    train_task = sequential_learner._register_task(
                        next_train_config, deps=sim_task
                    )

                    # Wait for training to complete
                    await train_task

                    iteration_count = i + 1

                # Store results for this algorithm
                result_dict: dict[str, Any] = {
                    "iterations": iteration_count,
                    "last_result": final_result,
                }
                self.algorithm_results[algorithm_name] = result_dict

                logger.info(
                    f"[Algorithm-{algorithm_name}] Completed "
                    f" with {iteration_count} iterations, final result: {final_result}"
                )

                return self.algorithm_results[algorithm_name]

            except Exception as e:
                logger.error(f"[Algorithm-{algorithm_name}] Failed with error: {e}")
                # Store failure information
                error_dict: dict[str, Any] = {
                    "iterations": 0,
                    "last_result": float("inf"),
                    "error": str(e),
                }
                self.algorithm_results[algorithm_name] = error_dict
                raise

        # Submit all algorithm pipelines asynchronously
        logger.debug(self.active_learn_functions)
        futures: list[Any] = [
            _run_algorithm_pipeline(al_name, al_task)
            for al_name, al_task in self.active_learn_functions.items()
        ]

        # Wait for all algorithms to complete
        try:
            results: list[Any] = await asyncio.gather(*futures, return_exceptions=False)

            # Process results and handle any exceptions
            for _, (algorithm_name, result) in enumerate(
                zip(self.active_learn_functions.keys(), results)
            ):
                if isinstance(result, Exception):
                    logger.error(f"[Algorithm-{algorithm_name}] Failed: {result}")
                    self.algorithm_results[algorithm_name] = {
                        "iterations": 0,
                        "last_result": float("inf"),
                        "error": str(result),
                    }

            # Select the best algorithm
            self._select_best_algorithm()

            return {
                "algorithm_results": self.algorithm_results,
                "best_algorithm": self.best_pipeline_name,
                "best_stats": self.best_pipeline_stats,
            }

        except Exception as e:
            logger.error(f"Error during algorithm selection: {e}")
            raise

    def _select_best_algorithm(self) -> None:
        """Select the best algorithm based on iterations and final result.

        Lower iterations and lower final results are considered better.
        Updates the best_pipeline_name and best_pipeline_stats attributes.

        Raises:
            ValueError: If no algorithm results are found or
            no algorithms completed successfully.
        """
        if not self.algorithm_results:
            raise ValueError(
                "No algorithm results found! "
                "Please make sure that at least one active "
                "learning algorithm ran successfully!"
            )

        # Filter out failed algorithms (those with errors)
        successful_algorithms: dict[str, dict[str, Any]] = {
            name: stats
            for name, stats in self.algorithm_results.items()
            if "error" not in stats and stats["iterations"] > 0
        }

        if not successful_algorithms:
            raise ValueError("No algorithms completed successfully!")

        # Sort by (iterations, last_result) - both ascending (lower is better)
        sorted_algorithms: list[tuple[str, dict[str, Any]]] = sorted(
            successful_algorithms.items(),
            key=lambda kv: (kv[1]["iterations"], kv[1]["last_result"]),
        )

        self.best_pipeline_name, self.best_pipeline_stats = sorted_algorithms[0]

        logger.info(
            f"Best algorithm is '{self.best_pipeline_name}' "
            f"with {self.best_pipeline_stats['iterations']} iteration(s) "
            f"and final metric result {self.best_pipeline_stats['last_result']}"
        )

    def get_best_algorithm(self) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Get the best algorithm name and its statistics.

        Returns:
            tuple containing the best algorithm name and its statistics dictionary.
            Returns (None, None) if no best algorithm has been selected yet.
        """
        return self.best_pipeline_name, self.best_pipeline_stats

    def get_all_results(self) -> dict[str, dict[str, Any]]:
        """Get results from all algorithms.

        Returns:
            Dictionary mapping algorithm names to their results. This is a copy
            of the internal results to prevent external modification.
        """
        return self.algorithm_results.copy()
