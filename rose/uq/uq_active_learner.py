import asyncio
import copy
import itertools
import warnings
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, Optional, Union

from radical.asyncflow import WorkflowEngine

from rose.learner import IterationState, TaskConfig
from rose.uq.uq_learner import UQLearner, UQLearnerConfig


class SeqUQLearner(UQLearner):
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
        """Initialize the SeqUQLearner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.

        """
        super().__init__(asyncflow)

    async def start(
        self,
        model_names: list,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learning_config: Optional[UQLearnerConfig] = None,
    ) -> AsyncIterator[IterationState]:
        """Start the learner and yield state at each iteration.

        This is the main entry point for running the learner. It returns an
        async iterator that yields IterationState at each iteration, giving
        the caller full control over the learning loop.

        Args:
            model_names: List of model names used for training.
            num_predictions: Number of predictions per training model.
            max_iter: Maximum number of iterations to run. If 0, runs until
                stop criterion is met (requires criterion_function to be set).
            skip_pre_loop: If True, skips the initial simulation and training
                phases before the main learning loop.
            learning_config: Configuration object containing per-iteration
                parameters.

        Yields:
            IterationState containing current iteration info, metrics, and
            all registered state from tasks.

        Raises:
            Exception: If required functions are not set or if neither
                max_iter nor criterion_function is provided.
        """
        # Validate required functions
        if (
            not self.simulation_function
            or not self.training_function
            or not self.prediction_function
            or not self.active_learn_function
        ):
            raise Exception(
                "Simulation, Training, prediction, and at least"
                " one AL function must be set!"
            )
        # Validate exit criteria
        if not max_iter and not self.criterion_function:
            raise Exception(
                "Either max_iter or stop_criterion_function must be provided."
            )

        self._max_iter = max_iter if max_iter > 0 else None
        learner_config = learning_config

        print(f"[Learner {self.learner_name}] Starting execution...")
        if len(model_names) > 1:
            prefix = (
                f"[Learner {self.learner_name}] starting training for "
                "Ensemble of Models: "
            )
        else:
            prefix = (
                f"[Learner {self.learner_name}] starting training for Single Model: "
            )
        print(f"{prefix} {model_names}")

        async def _training_stage(
            learning_config: Optional[UQLearnerConfig],
            model_name: str,
            iteration_count: int,
        ) -> list[Any]:
            """Run a simulation, train of a single model, and generate a set of
                predictions for that model.
            Args:
                learning_config:
                    Configuration object for retrieving task configurations.
                model_name:
                    Name of mode used for training.
                iteration_count:
                    Active learning Iteration count
            Returns:
                List of results from task (prediction) execution.
            Raises:
                Exception: If the training fails during execution.
            """

            sim_config: TaskConfig = self._get_iteration_task_config(
                self.simulation_function,
                learning_config,
                "simulation",
                iteration_count,
            )

            training_config: dict[str, Any] = copy.deepcopy(
                self._get_iteration_task_config(
                    self.training_function,
                    learning_config,
                    "training",
                    iteration_count,
                )
            )
            training_config["kwargs"]["--model_name"] = model_name

            sim_task = self._register_task(sim_config)
            training_task = self._register_task(training_config, deps=sim_task)

            # Wait for simulation (extract state later)
            await sim_task
            if self.is_stopped:
                return []

            # Wait for training (extract state later)
            await training_task
            if self.is_stopped:
                return []

            prediction_tasks = []
            for i in range(num_predictions):
                prediction_config: dict[str, Any] = copy.deepcopy(
                    self._get_iteration_task_config(
                        self.prediction_function,
                        learning_config,
                        "prediction",
                        iteration_count,
                    )
                )
                prediction_config["kwargs"]["--model_name"] = model_name
                prediction_config["kwargs"]["--iteration"] = i
                prediction_task = self._register_task(
                    prediction_config, deps=training_task
                )
                prediction_tasks.append(prediction_task)

            print(
                f"[{self.learner_name}-{model_name}] Completed training "
                f"for {iteration_count + 1} iteration(s) "
            )
            return await asyncio.gather(*prediction_tasks)

        # Track iterations and results for this pipeline
        stop_training = {name: False for name in model_names}

        # Initialize tasks for pre-loop
        training_tasks: tuple = ()
        train_results: list[Any] = []

        if not skip_pre_loop:
            futures: list[Coroutine] = [
                _training_stage(learner_config, model_name, 0)
                for model_name in model_names
            ]

            train_results = await asyncio.gather(*futures)
            if self.is_stopped:
                return

        # Determine iteration range
        iteration_range: Union[Iterator[int], range]
        if not max_iter:
            iteration_range = itertools.count()
        else:
            iteration_range = range(max_iter)

        # Main learning loop
        for i in iteration_range:
            if self.is_stopped:
                print(f"[Learner {self.learner_name}] Stop requested, exiting loop.")
                break

            # Clear transient state
            self.clear_state()

            # Extract state from train/prediction results
            for res_list in train_results:
                if isinstance(res_list, list):
                    for res in res_list:
                        self._extract_state_from_result(res)

            print(f"[Learner {self.learner_name}] Starting Iteration-{i}")
            uq_task: Any = None
            if self.uncertainty_function:
                uq_config: dict[str, Any] = self._get_iteration_task_config(
                    self.uncertainty_function, learner_config, "uncertainty", i
                )
                uq_task = self._register_task(uq_config, deps=training_tasks)
                uq_value = await uq_task
                if self.is_stopped:
                    break
                print(f"[Learner {self.learner_name}] {uq_value}")

                uq_model_stop, uq_stop_value = self._check_uncertainty(uq_value)
                if uq_model_stop:
                    print(
                        f"[Learner {self.learner_name}] UQ value reached "
                        f"its threshold - Stopping training for all models"
                        f" at iteration {i} with value: "
                        f"{uq_stop_value}"
                    )
                    # Build and yield final state
                    yield self.build_iteration_state(
                        iteration=i,
                        metric_value=uq_stop_value,
                        should_stop=True,
                        current_config=learner_config,
                    )
                    break

            # Get iteration-specific configurations
            acl_config: dict[str, Any] = self._get_iteration_task_config(
                self.active_learn_function, learner_config, "active_learn", i
            )
            acl_task = self._register_task(
                acl_config,
                deps=(uq_task if uq_task else training_tasks),
            )
            al_results = await acl_task
            if self.is_stopped:
                break
            self._extract_state_from_result(al_results)

            print(f"[Learner {self.learner_name}] {al_results}")

            # Check stop criterion if configured
            metric_value = None
            should_stop_all = False
            if self.criterion_function:
                stop_tasks = {}
                for model_name in model_names:
                    if stop_training[model_name]:
                        continue
                    criterion_function_cfg = copy.deepcopy(self.criterion_function)
                    criterion_function_cfg["kwargs"]["--model_name"] = model_name
                    stop_task = self._register_task(
                        criterion_function_cfg, deps=acl_task
                    )
                    stop_tasks[model_name] = stop_task

                results = await asyncio.gather(
                    *stop_tasks.values(), return_exceptions=True
                )
                if self.is_stopped:
                    break

                # Filter out exceptions
                valid_results = {}
                for model_name, res in zip(stop_tasks.keys(), results):
                    if not isinstance(res, Exception):
                        valid_results[model_name] = res

                should_stop_count = 0
                final_results = []
                for model_name, stop in valid_results.items():
                    model_stop, stop_val = self._check_stop_criterion(stop)
                    final_results.append(stop_val)
                    if model_stop:
                        stop_training[model_name] = True
                        should_stop_count += 1
                    metric_value = stop_val  # Last one or could be mean

                if should_stop_count == len(stop_tasks):
                    should_stop_all = True
                    print(
                        f"[Learner {self.learner_name}] Stopping "
                        f"criterion met for all models at iteration {i}"
                    )

            # Build and yield state
            yield self.build_iteration_state(
                iteration=i,
                metric_value=metric_value,
                should_stop=should_stop_all,
                current_config=learner_config,
            )

            if should_stop_all:
                break

            # Prepare next iteration
            next_futures: list[Coroutine] = [
                _training_stage(learner_config, model_name, i + 1)
                for model_name in model_names
                if not stop_training[model_name]
            ]
            train_results = await asyncio.gather(*next_futures)
            if self.is_stopped:
                break

    async def teach(
        self,
        model_names: list,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learning_config: Optional[UQLearnerConfig] = None,
    ) -> list[dict[str, Any]]:
        """Run sequential active learning loop to completion.
        .. deprecated::
            Use :meth:`start` instead.
        """
        warnings.warn(
            "teach() is deprecated and will be removed in a future version. "
            "Use start() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        results = []
        async for state in self.start(
            model_names=model_names,
            num_predictions=num_predictions,
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            learning_config=learning_config,
        ):
            results.append(state.to_dict())
        return results
        return self.learner_results


class ParallelUQLearner(SeqUQLearner):
    """
    Parallel active learner that runs multiple SeqUQLearners concurrently.
    This class orchestrates multiple SeqUQLearner instances to run in parallel,
    allowing for concurrent exploration of the learning space. Each learner can be
    configured independently through per-learner UQLearnerConfig objects.
    The parallel learner manages the lifecycle of all sequential learners and collects
    their results when all have completed their learning processes.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the Parallel Active Learner.
        Args:
            asyncflow:      The workflow engine instance used to manage async tasks
                            across all parallel learners.
        """
        super().__init__(asyncflow)

    def _create_sequential_learner(self, learner_name: str) -> SeqUQLearner:
        """Create a SeqUQLearner instance for a parallel learner.
        Creates and configures a new SeqUQLearner with the same base
        functions as the parent parallel learner, but with a unique identifier
        for logging and debugging purposes.

        Args:
            learner_name: Unique identifier for the learner.
        Returns:
            A fully configured SeqUQLearner instance ready to run
            independently in the parallel learning environment.
        """
        # Create a new sequential learner with the same asyncflow
        sequential_learner: SeqUQLearner = SeqUQLearner(self.asyncflow)

        # Copy the base functions from the parent learner
        sequential_learner.simulation_function = self.simulation_function
        sequential_learner.training_function = self.training_function
        sequential_learner.prediction_function = self.prediction_function
        sequential_learner.active_learn_function = self.active_learn_function
        sequential_learner.criterion_function = self.criterion_function
        sequential_learner.uncertainty_function = self.uncertainty_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_name = learner_name
        return sequential_learner

    def _convert_to_sequential_config(
        self, parallel_config: Optional[UQLearnerConfig]
    ) -> Optional[UQLearnerConfig]:
        """Convert a UQLearnerConfig to a UQLearnerConfig.
        Note: This method currently performs a direct copy as both parallel and
        sequential learners use the same UQLearnerConfig type. This method exists
        to provide a clear interface for potential future differences in
        configuration handling.

        Args:
            parallel_config: Configuration object designed for parallel learner
                usage. Contains simulation, training, active_learn, and criterion
                parameters.

        Returns:
            Equivalent UQLearnerConfig suitable for use with SeqUQLearner,
            or None if input was None.
        """
        if parallel_config is None:
            return None

        # Create UQLearnerConfig with same parameters
        return UQLearnerConfig(
            simulation=parallel_config.simulation,
            training=parallel_config.training,
            active_learn=parallel_config.active_learn,
            criterion=parallel_config.criterion,
            prediction=parallel_config.prediction,
            uncertainty=parallel_config.uncertainty,
        )

    async def start(
        self,
        learner_names: list,
        model_names: list,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[dict[str, Optional[UQLearnerConfig]]] = None,
    ) -> list[Any]:
        """Run parallel active learning by launching multiple SeqUQLearner.
        Orchestrates multiple SeqUQLearner instances to run concurrently,
        each with potentially different configurations. All learners run
        independently and their results are collected when all have completed.

        Args:
            learner_names: list of learner names to run concurrently.
            max_iter: Maximum number of iterations for each learner. If 0,
                learners run until their individual stop criteria are met.
            skip_pre_loop: If True, all learners skip their initial simulation
                and training phases.
            learner_configs: A list of configuration objects, one for each learner.
                            If None, all learners will use the default configuration.
                            If provided, the length must match the number
                            of elements in learner_names.

        Returns:
            list containing the results from each learner, in the same order
            as the learners were launched. Result types depend on the specific
            implementation of the learning functions.

        Raises:
            Exception: If required base functions are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            ValueError: If learner_configs length doesn't match parallel_learners.
        """
        # Validate base functions are set
        if (
            not self.simulation_function
            or not self.training_function
            or not self.active_learn_function
        ):
            raise Exception(
                "Simulation, Training, and Active Learning functions must be set!"
            )

        if not max_iter and not self.criterion_function:
            raise Exception(
                "Either max_iter or stop_criterion_function must be provided."
            )

        # Prepare learner configurations
        learner_configs = learner_configs or {name: None for name in learner_names}
        if len(learner_configs) != len(learner_names):
            raise ValueError("learner_configs length must match learner_names")

        async def _run_sequential_learner(learner_id: int) -> Any:
            """Run a single SeqUQLearner.
            Internal async function that manages the lifecycle of a single
            SeqUQLearner within the parallel learning context.

            Args:
                learner_name: Unique identifier for this learner instance.
            Returns:
                The result from the sequential learner's teach method.
            Raises:
                Exception: Re-raises any exception from the sequential learner
                    with additional context about which learner failed.
            """
            learner_name = learner_names[learner_id]
            try:
                # Create and configure the sequential learner
                sequential_learner: SeqUQLearner = self._create_sequential_learner(
                    learner_name
                )

                # Convert parallel config to sequential config
                sequential_config: Optional[UQLearnerConfig] = (
                    self._convert_to_sequential_config(learner_configs[learner_name])
                )
                print(f"[Parallel-Learner-{learner_name}] Starting sequential learning")

                # Run the sequential learner
                final_state = None
                async for state in sequential_learner.start(
                    model_names=model_names,
                    num_predictions=num_predictions,
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learning_config=sequential_config,
                ):
                    final_state = state
                    if self.is_stopped:
                        sequential_learner.stop()
                        # Child will detect is_stopped in its loop

                self.metric_values_per_iteration[f"learner-{learner_name}"] = (
                    sequential_learner.metric_values_per_iteration
                )
                self.uncertainty_values_per_iteration[f"learner-{learner_name}"] = (
                    sequential_learner.uncertainty_values_per_iteration
                )
                return final_state

            except Exception as e:
                print(f"[Parallel-Learner-{learner_name}] Failed with error: {e}")
                raise

        print(f"Starting Parallel Active Learning with {len(learner_names)} learners")

        # Submit all learners asynchronously
        futures: list[Coroutine] = [
            _run_sequential_learner(i) for i in range(len(learner_names))
        ]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*futures)

    async def teach(
        self,
        learner_names: list,
        model_names: list,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[dict[str, Optional[UQLearnerConfig]]] = None,
    ) -> list[Any]:
        """Run parallel active learning loop to completion.
        .. deprecated::
            Use :meth:`start` instead.
        """
        warnings.warn(
            "teach() is deprecated and will be removed in a future version. "
            "Use start() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.start(
            learner_names=learner_names,
            model_names=model_names,
            num_predictions=num_predictions,
            max_iter=max_iter,
            skip_pre_loop=skip_pre_loop,
            learner_configs=learner_configs,
        )
