import asyncio
import copy
import itertools
from collections.abc import Iterator
from typing import Any, Optional, Union

from radical.asyncflow import WorkflowEngine

from rose.uq.uq_learner import UQLearner, UQLearnerConfig

from ..learner import TaskConfig


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

    async def teach(
        self,
        model_names: list,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learning_config: Optional[dict[str, UQLearnerConfig]] = None,
    ) -> list[dict[str, Any]]:
        """Run sequential active learning with optional per-iteration configuration.
        Executes the active learning loop sequentially, with each iteration containing
        simulation, a set of training and prediction steps, and active learning phases.
        Supports configurable
        stopping criteria and per-iteration parameter customization.

        Args:
            num_predictions : Number of predictions per training model –
                            if only a single model is used for training,
                            the UQ metrics must be calculated based on
                            multiple predictions.
            max_iter        : Maximum number of iterations to run. If 0, runs until
                            stop criterion is met
                            (requires criterion_function to be set).
            skip_pre_loop   : If True, skips the initial simulation and training
                            phases before the main learning loop.
            learner_config  : Configuration object containing per-iteration
                            parameters for simulation, training, prediction,
                            active learning, and
                            criterion functions.
        Returns:
            The result of the learning process, including the mean value of uncertainty
            quantification (UQ) metrics
            and a list of model validation criteria (stop_value).
        Raises:
            Exception: If required functions (simulation_function, training_functions,
            prediction_function, active_learn_function) are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            Exception: If a training task has no matching prediction task.
            Exception: If any task of the learner fails during execution.
        """

        # # Validate required functions
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

        # Initialize learner configs if not provided
        learning_config = learning_config or {}

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

        async def _traininig_stage(
            learning_config: TaskConfig, model_name: str, iteration_count: int
        ) -> dict[str, Any]:
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
                The result from last task (prediction) execution.
            Raises:
                Exception: If the training fails during execution.
            """
            try:
                sim_config: TaskConfig = self._get_iteration_task_config(
                    self.simulation_function,
                    learning_config,
                    "simulation",
                    iteration_count,
                )

                training_config: TaskConfig = copy.deepcopy(
                    self._get_iteration_task_config(
                        self.training_function,
                        learning_config,
                        "training",
                        iteration_count,
                    )
                )
                training_config["kwargs"]["--model_name"] = model_name

                sim_task = self._register_task(sim_config)
                training_task = await self._register_task(
                    training_config, deps=sim_task
                )

                prediction_tasks = []
                for i in range(num_predictions):
                    prediction_config: TaskConfig = self._get_iteration_task_config(
                        self.prediction_function,
                        learning_config,
                        "prediction",
                        iteration_count,
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

            except Exception as e:
                print(
                    f"[{self.learner_name}-{model_name}] "
                    f"Failed train/prediction with error: {e}"
                )
                raise

        async def _run_pipeline() -> dict[str, Any]:
            """Run a single pipeline.
            Raises:
                Exception: If any task in pipeline fails during execution.
            """
            try:
                # Track iterations and results for this pipeline
                iteration_count: int = 0
                stop_training = {name: False for name in model_names}

                # Initialize tasks for pre-loop
                training_tasks: tuple = ()

                if not skip_pre_loop:
                    futures: list[Any] = [
                        _traininig_stage(learning_config, model_name, 0)
                        for model_name in model_names
                    ]

                    training_tasks = await asyncio.gather(*futures)

                # Determine iteration range
                iteration_range: Union[Iterator[int], range]
                if not max_iter:
                    iteration_range = itertools.count()
                else:
                    iteration_range = range(max_iter)

                # Main learning loop
                for i in iteration_range:
                    if self.criterion_function and self.uncertainty_function:
                        # Store results for current iteration
                        result_dict: dict[str, Any] = {
                            "iterations": iteration_count,
                        }

                    print(f"[Learner {self.learner_name}] Starting Iteration-{i}")
                    uq_task: tuple = ()
                    if self.uncertainty_function:
                        # Get iteration-specific configurations
                        uq_config: TaskConfig = self._get_iteration_task_config(
                            self.uncertainty_function, learning_config, "uncertainty", i
                        )
                        uq_task = self._register_task(uq_config, deps=training_tasks)
                        uq_value = await uq_task
                        print(f"[Learner {self.learner_name}] {uq_value}")

                        uq_model_stop, uq_stop_value = self._check_uncertainty(uq_value)
                        result_dict["uq"] = uq_stop_value
                        if uq_model_stop:
                            self.learner_results.append(result_dict)
                            print(
                                f"[Learner {self.learner_name}] UQ value reached "
                                f"its threshold - Stopping training for all models"
                                f" at iteration {i} with value: "
                                f"{uq_stop_value}"
                            )
                            break

                    # Get iteration-specific configurations
                    acl_config: TaskConfig = self._get_iteration_task_config(
                        self.active_learn_function, learning_config, "active_learn", i
                    )
                    acl_task = self._register_task(
                        acl_config,
                        deps=(uq_task if uq_task else training_tasks),
                    )
                    al_results = await acl_task

                    print(f"[Learner {self.learner_name}] {al_results}")

                    # Check stop criterion if configured
                    if self.criterion_function:
                        stop_tasks = {}
                        # Run validation for each model in learner
                        for model_name in model_names:
                            if stop_training[model_name]:
                                continue
                            criterion_function = self.criterion_function
                            criterion_function["kwargs"]["--model_name"] = model_name
                            stop_task = self._register_task(
                                criterion_function, deps=acl_task
                            )
                            stop_tasks[model_name] = stop_task

                        results = await asyncio.gather(*stop_tasks.values())
                        stops = dict(zip(stop_tasks.keys(), results))

                        model_stop: bool
                        stop_value: float
                        # The pipeline will stop once all models meet the exit criteria.
                        should_stop: int = 0
                        final_results = []
                        for model_name, stop in stops.items():
                            model_stop, stop_value = self._check_stop_criterion(stop)
                            final_results.append(stop_value)
                            if model_stop:
                                stop_training[model_name] = True
                                should_stop += 1
                                print(
                                    f"[Learner {self.learner_name}] Model "
                                    f"{model_name} will stop training"
                                    f" as stop criterion is met at iteration {i}"
                                )

                        # Store results for current iteration
                        result_dict["criterion"] = final_results

                        self.learner_results.append(result_dict)

                        if should_stop == len(stops):
                            print(
                                f"[Learner {self.learner_name}] Stopping "
                                f"criterion met for all models at iteration {i} "
                                f"with value: {stop_value}"
                            )
                            break

                    iteration_count = i + 1
                    futures: list[Any] = [
                        _traininig_stage(learning_config, model_name, iteration_count)
                        for model_name in model_names
                        if not stop_training[model_name]
                    ]
                    await asyncio.gather(*futures)

                    print(
                        f"[Learner {self.learner_name}] Completed "
                        f"{iteration_count + 1} iteration(s)"
                    )

            except Exception as e:
                print(f"[Learner {self.learner_name}] Failed with error: {e}")
                return e

        try:
            result = await _run_pipeline()
            if isinstance(result, Exception):
                print(f"[Learner {self.learner_name}] Failed: {result}")
            else:
                print(f"[Learner {self.learner_name}] Completed successfully")

        except Exception as e:
            print(f"Learner {self.learner_name}] Failed with error: {e}")
            return e
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

    async def teach(
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

        # # Validate base functions are set
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
        learner_configs = learner_configs or [None] * len(learner_names)
        if len(learner_configs) != len(learner_names):
            raise ValueError("learner_configs length must match learner_names")

        print(f"Starting Parallel Active Learning with {len(learner_names)} learners")

        async def _run_sequential_learner(learner_name: int) -> Any:
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
                learner_result = await sequential_learner.teach(
                    model_names=model_names,
                    num_predictions=num_predictions,
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learning_config=sequential_config,
                )

                self.metric_values_per_iteration[f"learner-{learner_name}"] = (
                    sequential_learner.metric_values_per_iteration
                )
                self.uncertainty_values_per_iteration[f"learner-{learner_name}"] = (
                    sequential_learner.uncertainty_values_per_iteration
                )
                return learner_result

            except Exception as e:
                print(f"[Parallel-Learner-{learner_name}] Failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures: list[Any] = [
            _run_sequential_learner(learner_name) for learner_name in learner_names
        ]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*[f for f in futures])
