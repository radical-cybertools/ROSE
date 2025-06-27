import typeguard
import itertools
from typing import Callable, Dict
from functools import wraps

from rose.engine import ResourceEngine
from rose.engine import WorkflowEngine
from rose.metrics import ActiveLearningMetrics as metrics

class ReinforcementLearner(WorkflowEngine):
    
    @typeguard.typechecked
    def __init__(self, engine: ResourceEngine, register_and_submit: bool=True) -> None:

        self.test_function = {}
        self.update_function = {}
        self.environment_function = {}

        super().__init__(engine)

        self.register_and_submit = register_and_submit


    def environment_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.environment_function = {'func':func,
                                        'args':args,
                                        'kwargs':kwargs}

            if self.register_and_submit:
                return self._register_task(self.environment_function)
        return wrapper

    def update_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.update_function = {'func':func,
                                          'args':args,
                                          'kwargs':kwargs}
            if self.register_and_submit:
                return self._register_task(self.update_function)
        return wrapper

    def utility_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            utility_function = {'func':func,
                                'args':args,
                                'kwargs':kwargs}

            if self.register_and_submit:
                return self._register_task(utility_function)
        return wrapper


    @typeguard.typechecked
    def as_stop_criterion(self, metric_name: str,
                                threshold: float,
                                operator: str = ''):
        # This is the outer function that takes arguments like metric_name and threshold
        @typeguard.typechecked
        def decorator(func: Callable):  # This is the actual decorator function that takes func
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Store the relevant information in self.criterion_function
                self.test_function = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                    'operator': operator,
                    'threshold': threshold,
                    'metric_name': metric_name}

                if self.register_and_submit:
                    res = self._register_task(self.test_function).result()
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

        return super().__call__(func)(*args, **kwargs)

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

    def _check_stop_criterion(self, stop_task_result):

        try:
            metric_value = eval(stop_task_result)
        except Exception as e:
            raise Exception(f"Failed to obtain a numerical value from criterion task: {e}")

        # check if the metric value is a number
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            operator = self.test_function['operator']
            threshold = self.test_function['threshold']
            metric_name = self.test_function['metric_name']

            if self.compare_metric(metric_name, metric_value, threshold, operator):
                print(f'stop criterion metric: {metric_name} is met with value of: {metric_value}'\
                      '. Breaking the reinforcement learning loop')
                return True, metric_value
            else:
                print(f'stop criterion metric: {metric_name} is not met yet ({metric_value}).')
                return False, metric_value
        else:
            raise TypeError(f'Stop criterion task must produce a numerical value, got {type(metric_value)} instead')

    def learn(self, max_iter:int = 0):
        raise NotImplementedError('This is not supported, please define your learn method and invoke it directly')


    def get_result(self, task_name: str):
        '''
        Get the result of a task(s) by its name, tasks might have
        similar name yet different future and task IDs.
        '''
        tasks = [t['future'].result()
                 for t in self.tasks.values()
                 if t['description']['name'] == task_name]

        return tasks

class SequentialReinforcementLearner(ReinforcementLearner):
    """
    SequentialReinforcementLearner is a subclass of ReinforcementLearner that implements
    a sequential reinforcement learning loop, where the learner interacts with the environment
    in a series of steps, updating its policy based on the rewards received from the environment.
    Useful for implementing on-policy(PPO, A2C) and off-policy(DQN) learning algorithms.

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
    """
    def __init__(self, engine: ResourceEngine) -> None:
        super().__init__(engine, register_and_submit=False)

    def learn(self, max_iter:int = 0):
        '''
        Run the reinforcement learning loop for a specified number of iterations.
        Args:
            max_iter (int, optional): The maximum number of iterations for the
            reinforcement learning loop. If not provided, the value set during initialization   
            will be used. Defaults to 0.
        '''
        # start the initial step for RL by defining and setting environment and update tasks
        if not self.environment_function or \
              not self.update_function:
            raise Exception("Environment and Update function must be set!")
        
        if not self.test_function:
            raise Exception("Test function must be set!")

        if not max_iter:
            max_iter = itertools.count()
        else:
            max_iter = range(max_iter)

        # form the RL loop and workflow
        for i in max_iter:
            print(f'Starting Iteration-{i}')
            env_task = self._register_task(self.environment_function)
            update_task = self._register_task(self.update_function, deps=env_task)

            test_task = self._register_task(self.test_function, deps=update_task)

            test_result = test_task.result()
            should_stop, _ = self._check_stop_criterion(test_result)
            if should_stop:
                break

class ParallelExperience(ReinforcementLearner):
    """
    ParallelExperience is a subclass of ReinforcementLearner that implements
    a parallel reinforcement learning loop, where multiple environments are run in parallel
    to collect experiences, and then a single update step is performed on the collected experiences.

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
    """
    def __init__(self, engine: ResourceEngine) -> None:
        super().__init__(engine, register_and_submit=False)
        self.environment_functions: Dict[str, Dict] = {}
        self.work_dir = '.'
        
    def environment_task(self, name: str):
        """
        A decorator that registers an environment task under the given name.
        
        Usage:
        
        @par_exp.environment_task(name='env_1')
        def environment_1(*args):
            ...
        
        @par_exp.environment_task(name='env_2')
        def environment_2(*args):
            ...
        """
        def decorator(func: Callable):
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

    def merge_banks(self):
        import os
        from .experience import Experience, ExperienceBank, create_experience
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
            except:
                print(f"  Failed to load {bank_file}")
    
        for bank_file in bank_files:
            try:
                os.remove(bank_file)
            except Exception as e:
                print(f"  Failed to delete {bank_file}: {e}")
        
        # Save merged bank
        merged.save(self.work_dir, "experience_bank.pkl")

    def learn(self, max_iter:int = 0):
        '''
        Run the parallel reinforcement learning loop for a specified number of iterations.
        Args:
            max_iter (int, optional): The maximum number of iterations for the
            reinforcement learning loop. If not provided, the value set during initialization   
            will be used. Defaults to 0.
        '''
        
        if not self.environment_functions or \
                not self.update_function or \
                    not self.test_function:
                raise Exception("Environment, Update, and Test functions must be set!")

        if not max_iter:
            max_iter = itertools.count()
        else:
            max_iter = range(max_iter)

        # form the RL loop and workflow
        for i in max_iter:
            print(f'Starting Iteration-{i}')

            # Collect experiences from parallel environments
            env_tasks = []
            for name, env_func in self.environment_functions.items():
                env_task = self._register_task(env_func)
                env_tasks.append(env_task)
            
            # Wait for all environment tasks to complete
            env_results = [env.result() for env in env_tasks]
            
            self.merge_banks()
            
            update_task = self._register_task(self.update_function)
            test_task = self._register_task(self.test_function, deps=update_task)
            test_result = test_task.result()

            should_stop, _ = self._check_stop_criterion(test_result)

            if should_stop:
                break
