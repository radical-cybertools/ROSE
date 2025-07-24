import asyncio
import typeguard
import itertools
from typing import Callable, Dict, List, Any, Optional
from functools import wraps

from rose.learner import Learner

from radical.asyncflow import WorkflowEngine


class ReinforcementLearner(Learner):
    """Base class for reinforcement learning implementations.
    
    This class provides the foundation for implementing reinforcement learning
    algorithms with asynchronous workflow management. It maintains dictionaries
    for test, update, and environment functions that can be registered and
    executed as tasks.
    
    Attributes:
        test_function (Dict): Dictionary storing test function configurations.
        update_function (Dict): Dictionary storing update function configurations.
        environment_function (Dict): Dictionary storing environment function configurations.
        update_task (Callable): Registered update task callable.
        environment_task (Callable): Registered environment task decorator.
    """

    @typeguard.typechecked
    def __init__(self, asyncflow: WorkflowEngine, register_and_submit: bool = True) -> None:
        """Initialize the ReinforcementLearner.
        
        Args:
            asyncflow (WorkflowEngine): The workflow engine for managing asynchronous tasks.
            register_and_submit (bool, optional): Whether to automatically register and
                submit tasks. Defaults to True.
        """

        super().__init__(asyncflow, register_and_submit)

        self.test_function = {}
        self.update_function = {}
        self.environment_function = {}

        self.update_task: Callable = self.register_decorator('update')
        self.environment_task: Callable = self.register_decorator('environment')

    @typeguard.typechecked
    def as_stop_criterion(self, metric_name: str,
                                threshold: float,
                                operator: str = ''):
        # This is the outer function that takes arguments like metric_name and threshold
        @typeguard.typechecked
        def decorator(func: Callable):  # This is the actual decorator function that takes func
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Store the relevant information in self.criterion_function
                self.test_function = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                    'operator': operator,
                    'threshold': threshold,
                    'metric_name': metric_name}

                self.criterion_function = self.test_function

                if self.register_and_submit:
                    res = await self._register_task(self.test_function)
                    return self._check_stop_criterion(res)
            return wrapper
        return decorator


class SequentialReinforcementLearner(ReinforcementLearner):
    """Sequential reinforcement learning implementation.
    
    This class implements a sequential reinforcement learning loop where the learner
    interacts with the environment in a series of steps, updating its policy based
    on the rewards received from the environment. It's useful for implementing both
    on-policy (PPO, A2C) and off-policy (DQN) learning algorithms.
    
    The learning process follows this sequential pattern:
    
        Iteration 1:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration 2:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration 3:
        [Env] -> [Update] -> [Test]
                    |
                    v
        Iteration N:
        [Env] -> [Update] -> [Test]
    
    Each iteration consists of three sequential steps:
    1. Environment interaction to collect experiences
    2. Policy update based on collected experiences  
    3. Testing/evaluation of the updated policy
    """
    
    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the SequentialReinforcementLearner.
        
        Args:
            asyncflow (WorkflowEngine): The workflow engine for managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=True)

    async def learn(self, max_iter: int = 0) -> None:
        """Run the sequential reinforcement learning loop.

        Executes the reinforcement learning algorithm for a specified number of
        iterations. Each iteration performs environment interaction, policy update,
        and testing in sequence. The loop can be terminated early if stopping
        criteria are met.
        
        Args:
            max_iter (int, optional): The maximum number of iterations for the
                reinforcement learning loop. If 0 or not provided, runs indefinitely.
                Defaults to 0.
                
        Raises:
            Exception: If environment, update, or test functions are not set.
        """
        # Validate that required functions are set
        if not self.environment_function or not self.update_function:
            raise Exception("Environment and Update function must be set!")

        if not self.test_function:
            raise Exception("Test function must be set!")

        # Setup iteration counter
        if not max_iter:
            max_iter = itertools.count()
        else:
            max_iter = range(max_iter)

        # Execute the RL loop
        for i in max_iter:
            print(f'Starting Iteration-{i}')
            
            # Register and execute tasks in sequence
            env_task = self._register_task(self.environment_function)
            update_task = self._register_task(self.update_function, deps=env_task)
            test_task = self._register_task(self.test_function, deps=update_task)

            # Wait for test completion and check stopping criteria
            test_result = await test_task

            should_stop, _ = self._check_stop_criterion(test_result)

            if should_stop:
                break


class ParallelExperience(ReinforcementLearner):
    """Parallel experience collection reinforcement learning implementation.
    
    This class implements a parallel reinforcement learning loop where multiple
    environments run in parallel to collect experiences. After all environments
    complete their data collection, the experiences are merged and a single
    update step is performed on the aggregated data.
    
    The learning process follows this parallel pattern:
    
        Environment 1       Environment 2     Environment 3
            |                   |                 |
          [Collect]         [Collect]         [Collect]
            |                   |                 |
                --->   [Merge Experiences]   <---
                                |
                                v
                        [Update Policy]
                                |
                                v
                        [Test Policy]
    
    This approach is particularly useful for:
    - Collecting diverse experiences from multiple environment instances
    - Improving sample efficiency through parallel data collection
    - Implementing distributed reinforcement learning algorithms
    
    Attributes:
        environment_functions (Dict[str, Dict]): Dictionary mapping environment names
            to their function configurations.
        work_dir (str): Working directory for saving and loading experience banks.
    """
    
    def __init__(self, asyncflow: WorkflowEngine) -> None:
        """Initialize the ParallelExperience learner.
        
        Args:
            asyncflow (WorkflowEngine): The workflow engine for managing asynchronous tasks.
        """
        super().__init__(asyncflow, register_and_submit=False)
        self.environment_functions: Dict[str, Dict] = {}
        self.work_dir = '.'
        
    def environment_task(self, name: str) -> Callable:
        """Decorator to register an environment task under a given name.
        
        This decorator allows registering multiple environment functions that will
        be executed in parallel during the learning process. Each environment
        should collect experiences independently.
        
        Args:
            name (str): Unique name identifier for the environment task.
            
        Returns:
            Callable: Decorator function that wraps the environment function.
            
        Example:
            @par_exp.environment_task(name='env_1')
            def environment_1(*args):
                # Environment 1 logic here
                pass
            
            @par_exp.environment_task(name='env_2')
            def environment_2(*args):
                # Environment 2 logic here
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.environment_functions[name] = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs
                }
                if self.register_and_submit:
                    return self._register_task(self.environment_functions[name])
            return wrapper
        return decorator

    def merge_banks(self) -> None:
        """Merge all experience banks from parallel environments.
        
        This method searches for experience bank files in the working directory,
        loads them, merges them into a single consolidated experience bank, and
        then removes the individual bank files to clean up the workspace.
        
        The merged experience bank is saved as "experience_bank.pkl" in the
        working directory.
        
        Note:
            Experience bank files are expected to follow the naming pattern
            "experience_bank_*.pkl" where * can be any string identifier.
        """
        import os
        from .experience import ExperienceBank

        # Find all experience bank files
        bank_files = []
        for filename in os.listdir(self.work_dir):
            if filename.startswith("experience_bank_") and filename.endswith(".pkl"):
                bank_files.append(os.path.join(self.work_dir, filename))
        
        if not bank_files:
            print("No experience banks found!")
            return

        print(f"Found {len(bank_files)} experience banks")
        
        # Create merged bank and load all files
        merged = ExperienceBank()
        total = 0
        
        for bank_file in bank_files:
            try:
                bank = ExperienceBank.load(bank_file)
                merged.merge_inplace(bank)
                total += len(bank)
                print(f"  Merged {len(bank)} from {os.path.basename(bank_file)}")
            except Exception as e:
                print(f"  Failed to load {bank_file}: {e}")
    
        # Clean up individual bank files
        for bank_file in bank_files:
            try:
                os.remove(bank_file)
            except Exception as e:
                print(f"  Failed to delete {bank_file}: {e}")

        # Save merged bank
        merged.save(self.work_dir, "experience_bank.pkl")
        print(f"  Saved merged bank with {total} total experiences")

    async def learn(self, max_iter: int = 0) -> None:
        """Run the parallel reinforcement learning loop.
        
        Executes the reinforcement learning algorithm with parallel experience
        collection for a specified number of iterations. Each iteration runs
        multiple environments in parallel, merges their experiences, updates
        the policy, and tests performance.
        
        Args:
            max_iter (int, optional): The maximum number of iterations for the
                reinforcement learning loop. If 0 or not provided, runs indefinitely.
                Defaults to 0.
                
        Raises:
            Exception: If environment functions, update function, or test function
                are not properly configured.
        """
        # Validate that required functions are set
        if not self.environment_functions or \
                not self.update_function or \
                not self.test_function:
            raise Exception("Environment, Update, and Test functions must be set!")

        # Setup iteration counter
        if not max_iter:
            max_iter = itertools.count()
        else:
            max_iter = range(max_iter)

        # Execute the parallel RL loop
        for i in max_iter:
            print(f'Starting Iteration-{i}')

            # Collect experiences from parallel environments
            env_tasks = []
            for _, env_func in self.environment_functions.items():
                env_task = self._register_task(env_func)
                env_tasks.append(env_task)

            # Wait for all environment tasks to complete
            await asyncio.gather(*env_tasks, return_exceptions=True)

            # Merge all collected experiences
            self.merge_banks()

            # Update policy and test performance
            update_task = self._register_task(self.update_function)
            test_task = self._register_task(self.test_function, deps=update_task)
            test_result = await test_task

            # Check if we should stop learning
            should_stop, _ = self._check_stop_criterion(test_result)

            print(f'Finished Iteration-{i}')

            if should_stop:
                break