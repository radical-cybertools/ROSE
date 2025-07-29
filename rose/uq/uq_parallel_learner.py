import asyncio
import itertools
import typeguard
import copy
from typing import Callable, Dict, Any, Optional, List, Union, Tuple, Iterator
from functools import wraps
from ..learner import Learner
from ..learner import TaskConfig
from ..learner import LearnerConfig
from radical.asyncflow import WorkflowEngine


class UQLearner(Learner):
    """UQ active learner that runs iterations one after another.
    
    This class implements a sequential active learning approach based on Uncertainty Quantification. 
    Each iteration consists of simulation, a set of training and prediction steps, and active learning phases executed in sequence. 
    The learner can be configured with per-iteration parameters using LearnerConfig.
    
    Attributes:
        learner_name (Optional[int]):   Identifier for the learner. 
                                        Set by ParallelActiveLearner when used in parallel mode.
        training_functions          :   Dictionary used to store individual training tasks, with each model name as the key.
        training_task               :   Decorator to register training task
        predict_functions           :   Dictionary used to store individual prediction tasks, with each model name as the key.
        predict_task                :   Decorator to register prediction task
        learner_results             :   Learner raining results.
    """

    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the UQLearner.

        Args:
            asyncflow: The workflow engine instance for managing async tasks.

        """
        super().__init__(asyncflow, register_and_submit=False)

        self.learner_name: str = 'UQLearner'
        
        self.training_functions: Dict[str, Dict[str, Any]] = {}
        # Override the parent's training_function decorator
        self.training_task: Callable[[str], Callable] = self._ensemble_training_task
        
        self.predict_functions: Dict[str, Dict[str, Any]] = {}
        # Override the parent's training_function decorator
        self.predict_task: Callable[[str], Callable] = self._ensemble_predict_task

        # List storing training results for each iteration.
        self.learner_results: List[str, Dict[str, Any]] = []
    
    def _ensemble_training_task(self, name: str) -> Callable[[Callable], Callable]:
        """Create a decorator for registering training task.
        Args:
            name: The model's unique identifier."
        Returns:
            A decorator function that registers the training function.
        """
        def decorator(func: Callable) -> Callable:
            """Decorator that registers an training function.
            Args:
                func: The training function to register.
            Returns:
                The original function unchanged.
            """
            # immediately register the function itself
            self.training_functions[name] = {
                'func': func,
                'args': (),
                'kwargs': {'--model_name': name}}
            return func  # just return the original
        return decorator

    @typeguard.typechecked
    def _ensemble_predict_task(self, name: str) -> Callable[[Callable], Callable]:
        """Create a decorator for registering prediction.
        Args:
            name: The model's unique identifier."
        Returns:
            A decorator function that registers the prediction function.
        """
        def decorator(func: Callable) -> Callable:
            """Decorator that registers an prediction function.
            Args:
                func: The prediction function to register.
            Returns:
                The original function unchanged.
            """
            # immediately register the function itself
            self.predict_functions[name] = {
                'func': func,
                'args': (),
                'kwargs': {'--model_name': name}}
            return func  # just return the original
        return decorator
    
    async def teach(self, 
                    num_predictions: int = 1,
                    max_iter: int = 0,
                    skip_pre_loop: bool = False,
                    learning_config: Optional[Dict[str, LearnerConfig]] = None) -> Dict[str, Any]:
        """Run sequential active learning with optional per-iteration configuration.
        
        Executes the active learning loop sequentially, with each iteration containing
        simulation, a set of training and prediction steps, and active learning phases. Supports configurable
        stopping criteria and per-iteration parameter customization.
        
        Args:
            num_predictions : Number of predictions per training model – 
                            if only a single model is used for training, the UQ metrics must be calculated based on multiple predictions.
            max_iter        : Maximum number of iterations to run. If 0, runs until
                            stop criterion is met (requires criterion_function to be set).
            skip_pre_loop   : If True, skips the initial simulation and training
                            phases before the main learning loop.
            learner_config  : Configuration object containing per-iteration
                            parameters for simulation, training, prediction, active learning, and
                            criterion functions.    
        Returns:
            The result of the learning process, including the mean value of uncertainty quantification (UQ) metrics 
            and a list of model validation criteria (stop_value).
        Raises:
            Exception: If required functions (simulation_function, training_functions, prediction_functions,
                active_learn_function) are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            Exception: If a training task has no matching prediction task.
            Exception: If any task of the learner fails during execution.
        """

        # Validate required functions
        if not self.simulation_function or not self.training_functions or not self.predict_functions or not self.active_learn_function:
            raise Exception("Simulation, Training, and at least one AL function must be set!")
        # Validate exit criteria
        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")
        # Validate that each training model has respective prediction function
        for name in self.training_functions.keys():
            if name not in self.predict_functions.keys():
                raise Exception(f"Predict function for model '{name}' is not registered ({self.predict_functions.keys()}).")
    
        # Initialize learner configs if not provided
        learning_config = learning_config or {}

        print(f"[Learner {self.learner_name}] Starting execution...")
        if len(self.training_functions) > 1:
            prefix = f"[Learner {self.learner_name}] starting training for Ensemble of Models: "
        else:
            prefix = f"[Learner {self.learner_name}] starting training for Single Model: "  
        print(f"{prefix} {list(self.training_functions.keys())}")

        @self.asyncflow.block
        async def _traininig_block(learning_config: TaskConfig, 
                                    model_name: str, 
                                    iteration_count: int) -> Dict[str, Any]:
            """Run a simulation, train of a single model, and generate a set of predictions for that model.
            Args:
                learning_config:    Configuration object for retrieving task configurations.
                model_name:         Name of mode used for training.
                iteration_count:    Active learning Iteration count
            Returns:
                The result from last task (prediction) execution.  
            Raises:
                Exception: If the training block fails during execution.
            """
            try:
                
                predict_func = self.predict_functions[model_name]
                predict_func['kwargs']['--iteration'] = iteration_count

                sim_config: TaskConfig = self._get_iteration_task_config(
                        self.simulation_function, learning_config, 'simulation', iteration_count
                    )
                train_config: TaskConfig = self._get_iteration_task_config(
                    self.training_functions[model_name], learning_config, 'training', iteration_count
                )

                train_config['kwargs'] = train_config['kwargs'] | self.training_functions[model_name]['kwargs']

                sim_task = self._register_task(sim_config)
                train_task =  await self._register_task(train_config, deps=sim_task)

                predict_tasks = []
                for i in range(num_predictions):
                    predict_func['kwargs']['--iteration'] = i
                    predict_config: TaskConfig = self._get_iteration_task_config(
                        predict_func, learning_config, 'predict', iteration_count
                    )
                    predict_task =  self._register_task(predict_config, deps=train_task)
                    predict_tasks.append(predict_task)

                print(f"[Model-{model_name}] Completed with {iteration_count} iterations ")
                return await asyncio.gather(*predict_tasks)

            except Exception as e:
                print(f"[Model-{model_name}] Failed train/prediction with error: {e}")
                raise

        @self.asyncflow.block
        async def _run_pipeline() -> Dict[str, Any]:
            """Run a single pipeline.
            Raises:
                Exception: If any task in pipeline fails during execution.
            """
            try:
                # Track iterations and results for this pipeline
                iteration_count: int = 0

                if not skip_pre_loop:
                    futures: List[Any] = [
                                    _traininig_block(learning_config, model_name, 0) 
                                    for model_name in self.training_functions.keys()
                                ]
                    train_tasks = await asyncio.gather(*futures)

                # Determine iteration range
                iteration_range: Union[Iterator[int], range]
                if not max_iter:
                    iteration_range = itertools.count()
                else:
                    iteration_range = range(max_iter)

                # Main learning loop
                for i in iteration_range:
                    print(f'[Learner {self.learner_name}] Starting Iteration-{i}')

                    # Get iteration-specific configurations
                    acl_config: TaskConfig = self._get_iteration_task_config(
                        self.active_learn_function, learning_config, 'active_learn', i
                    )

                    acl_task = self._register_task(acl_config, deps=train_tasks)
                    uq: Any = await acl_task
                    print(f'[Learner {self.learner_name}] UQ metrics {uq}')

                    # Check stop criterion if configured
                    if self.criterion_function:
                        stop_tasks = []
                        # Run validation for each model in learner
                        for model_name in self.training_functions.keys():
                            criterion_function = self.criterion_function
                            criterion_function['kwargs']['--model_name'] = model_name
                            stop_task =  self._register_task(criterion_function, deps=acl_task)
                            stop_tasks.append(stop_task)
                        stops = await asyncio.gather(*stop_tasks)

                        model_stop: bool
                        stop_value: float
                        should_stop: int = 0  #The pipeline will stop once all models meet the exit criteria.
                        final_results = []
                        for stop in stops:
                            model_stop, stop_value = self._check_stop_criterion(stop)
                            final_results.append(stop_value)
                            if model_stop:
                                should_stop +=1

                        # Store results for current iteration
                        result_dict: Dict[str, Any] = {
                            'iterations': iteration_count,
                            'criterion': final_results,
                            'uq': str(uq).strip()
                        }
                        self.learner_results.append(result_dict)

                        if should_stop == len(stops):
                            print(f"[Learner {self.learner_name}] Stopping criterion met for all models at iteration {i} with value: {stop_value}")
                            break
                    
                    iteration_count = i + 1
                    futures: List[Any] = [
                                    _traininig_block(learning_config, name, iteration_count) 
                                    for name in self.training_functions.keys()
                                ]
                    await asyncio.gather(*futures)

                print(f"[Learner {self.learner_name}] Completed with {iteration_count} iterations")

            except Exception as e:
                print(f"[Learner {self.learner_name}] Failed with error: {e}")
                raise

        try:
            result = await _run_pipeline()
            if isinstance(result, Exception):
                print(f"[Learner {self.learner_name}] Failed: {result}")
            else:
                print(f"[Learner {self.learner_name}] Completed successfully")

        except Exception as e:
            print(f"[Learner {self.learner_name}] Failed with error: {e}")
            raise
        return self.learner_results
    

class ParallelUQLearner(UQLearner):
    """Parallel active learner that runs multiple UQLearners concurrently.
    
    This class orchestrates multiple UQLearner instances to run in parallel,
    allowing for concurrent exploration of the learning space. Each learner can be
    configured independently through per-learner LearnerConfig objects.
    
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

    def _create_sequential_learner(self, 
        learner_name: str
    ) -> UQLearner:
        """Create a UQLearner instance for a parallel learner.
        
        Creates and configures a new UQLearner with the same base
        functions as the parent parallel learner, but with a unique identifier
        for logging and debugging purposes.

        Args:
            learner_name: Unique identifier for the learner.
        Returns:
            A fully configured UQLearner instance ready to run
            independently in the parallel learning environment.
        """
        # Create a new sequential learner with the same asyncflow
        sequential_learner: UQLearner = UQLearner(self.asyncflow)

        # Copy the base functions from the parent learner
        sequential_learner.simulation_function = self.simulation_function
        sequential_learner.training_functions = self.training_functions
        sequential_learner.predict_functions = copy.deepcopy(self.predict_functions)
        for predict_func in sequential_learner.predict_functions.values():
            predict_func['kwargs']['--prediction_dir'] = f'{learner_name}_prediction'
            predict_func['kwargs']['--learner_name'] = learner_name

        sequential_learner.active_learn_function = self.active_learn_function
        sequential_learner.criterion_function = self.criterion_function

        # Set learner-specific identifier for logging
        sequential_learner.learner_name = learner_name
        return sequential_learner

    def _convert_to_sequential_config(self, 
        parallel_config: Optional[LearnerConfig]
    ) -> Optional[LearnerConfig]:
        """Convert a LearnerConfig to a LearnerConfig.
        
        Note: This method currently performs a direct copy as both parallel and
        sequential learners use the same LearnerConfig type. This method exists
        to provide a clear interface for potential future differences in
        configuration handling.

        Args:
            parallel_config: Configuration object designed for parallel learner
                usage. Contains simulation, training, active_learn, and criterion
                parameters.

        Returns:
            Equivalent LearnerConfig suitable for use with UQLearner,
            or None if input was None.
        """
        if parallel_config is None:
            return None
        # Create LearnerConfig with same parameters
        return LearnerConfig(
            simulation=parallel_config.simulation,
            training=parallel_config.training,
            active_learn=parallel_config.active_learn,
            criterion=parallel_config.criterion
        )

    async def teach(self,
        learner_names: List,
        num_predictions: int = 1,
        max_iter: int = 0,
        skip_pre_loop: bool = False,
        learner_configs: Optional[Dict[str, Optional[LearnerConfig]]] = None,
    ) -> List[Any]:
        """Run parallel active learning by launching multiple UQLearners.
        
        Orchestrates multiple UQLearner instances to run concurrently,
        each with potentially different configurations. All learners run
        independently and their results are collected when all have completed.

        Args:
            learner_names: List of learner names to run concurrently.
            max_iter: Maximum number of iterations for each learner. If 0,
                learners run until their individual stop criteria are met.
            skip_pre_loop: If True, all learners skip their initial simulation
                and training phases.
            learner_configs: A list of configuration objects, one for each learner. 
                            If None, all learners will use the default configuration. 
                            If provided, the length must match the number of elements in learner_names.

        Returns:
            List containing the results from each learner, in the same order
            as the learners were launched. Result types depend on the specific
            implementation of the learning functions.

        Raises:
            Exception: If required base functions are not set.
            Exception: If neither max_iter nor criterion_function is provided.
            ValueError: If learner_configs length doesn't match parallel_learners.
        """

        parallel_learners = len(learner_names)
        # Validate base functions are set
        if not self.simulation_function or not self.training_functions or not self.active_learn_function:
            raise Exception("Simulation, Training, and Active Learning functions must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        # Prepare learner configurations
        learner_configs = learner_configs or [None] * len(learner_names)
        if len(learner_configs) != parallel_learners:
            raise ValueError("learner_configs length must match parallel_learners")

        print(f"Starting Parallel Active Learning with {parallel_learners} learners")

        @self.asyncflow.block
        async def _run_sequential_learner(learner_name: int) -> Any:
            """Run a single UQLearner.
            
            Internal async function that manages the lifecycle of a single
            UQLearner within the parallel learning context.
            
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
                sequential_learner: UQLearner = self._create_sequential_learner(
                    learner_name)
                
                # Convert parallel config to sequential config
                sequential_config: Optional[LearnerConfig] = self._convert_to_sequential_config(
                    learner_configs[learner_name]
                )
                print(f"[Parallel-Learner-{learner_name}] Starting sequential learning")

                # Run the sequential learner
                return {learner_name: await sequential_learner.teach(
                    num_predictions = num_predictions,
                    max_iter=max_iter,
                    skip_pre_loop=skip_pre_loop,
                    learning_config=sequential_config
                )}
            except Exception as e:
                print(f"[Parallel-Learner-{learner_name}] Failed with error: {e}")
                raise

        # Submit all learners asynchronously
        futures: List[Any] = [_run_sequential_learner(learner_name) for learner_name in learner_names]

        # Wait for all learners to complete and collect results
        return await asyncio.gather(*[f for f in futures])
