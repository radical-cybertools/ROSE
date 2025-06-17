import typeguard
import itertools
from typing import Callable, Dict
from functools import wraps

from .engine import ResourceEngine
from .engine import WorkflowEngine
from .metrics import ActiveLearningMetrics as metrics

class ActiveLearner(WorkflowEngine):

    @typeguard.typechecked
    def __init__(self, engine: ResourceEngine, register_and_submit: bool=True) -> None:

        self.criterion_function = {}
        self.training_function = {}
        self.simulation_function = {}
        self.active_learn_function = {}

        super().__init__(engine)

        self.register_and_submit = register_and_submit


    def simulation_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.simulation_function = {'func':func,
                                        'args':args,
                                        'kwargs':kwargs}
            
            if self.register_and_submit:
                return self._register_task(self.simulation_function)
        return wrapper

    def training_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.training_function = {'func':func,
                                      'args':args,
                                      'kwargs':kwargs}
            if self.register_and_submit:
                return self._register_task(self.training_function)
        return wrapper

    def active_learn_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.active_learn_function = {'func':func,
                                          'args':args,
                                          'kwargs':kwargs}
            if self.register_and_submit:
                return self._register_task(self.active_learn_function)
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
                self.criterion_function = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                    'operator': operator,
                    'threshold': threshold,
                    'metric_name': metric_name}
                
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

class SequentialActiveLearner(ActiveLearner):
    '''
    SequentialActiveLearner is a subclass of ActiveLearner that implements
    a sequential active learning loop.

           Iteration 1:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration 2:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration 3:
    [Sim] -> [Active Learn] -> [Train]

                |
                v
           Iteration N
    '''
    def __init__(self, engine: ResourceEngine) -> None:
        '''
        Initialize the SequentialActiveLearner object.

        Args:
            engine: The ResourceEngine object that manages the resources and
            tasks submission to HPC resources during the active learning loop.
        '''
        super().__init__(engine, register_and_submit=False)

    def teach(self, max_iter:int = 0, skip_pre_loop:bool = False):
        '''
        Run the active learning loop for a specified number of iterations.

        Args:
            max_iter (int, optional): The maximum number of iterations for the
            active learning loop. If not provided, the value set during initialization
            will be used. Defaults to 0.
        '''
        # start the initial step for active learning by
        # defining and setting simulation and training tasks
        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_function:
            raise Exception("Simulation and Training function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        sim_task, train_task = (), ()

        if not skip_pre_loop:
            # step-1 invoke the pre_step only once
            sim_task, train_task = self._start_pre_loop()

        # if no max_iter is provided, run the loop indefinitely
        # and until the stop criterion is met
        if not max_iter:
            max_iter = itertools.count()

        else:
            max_iter = range(max_iter)

        # step-2 form the ACL loop and workflow
        for i in max_iter:
            print(f'Starting Iteration-{i}')
            acl_task = self._register_task(self.active_learn_function, deps=(sim_task, train_task))

            if self.criterion_function:
                stop_task = self._register_task(self.criterion_function, deps=acl_task)
                stop = stop_task.result()

                should_stop, _ = self._check_stop_criterion(stop)
                if should_stop:
                    break

            sim_task = self._register_task(self.simulation_function, deps=acl_task)
            train_task = self._register_task(self.training_function, deps=sim_task)

            # block/wait for each workflow until it finishes
            train_task.result()


class ParallelActiveLearner(SequentialActiveLearner):
    '''
    ParallelActiveLearner is a subclass of ActiveLearner that implements
    a parallel active learning loop.

    Parallel Learner 1        Parallel Learner 2        Parallel Learner 3
            |                         |                         |
          [Sim]                     [Sim]                     [Sim]
            |                         |                         |
         [Train]                   [Train]                   [Train]
            |                         |                         |
        [Active Learn]          [Active Learn]            [Active Learn]
            |                         |                         |
            v                         v                         v
    -----------------------------------------------------------------------
                       Parallel Execution of All Learners
                                      |
                                      v
                         (Next Parallel Learner N)
                                      |
                                    [Sim]
                                      |
                                   [Train]
                                      |
                               [Active Learn]
    '''
    def __init__(self, engine: ResourceEngine) -> None:
        '''
        Initialize the ParallelActiveLearner object.

        Args:
            engine: The ResourceEngine object that manages the resources and
            tasks submission to HPC resources during the active learning loop.
        '''
        super().__init__(engine)

    def teach(self, parallel_learners:int = 2, skip_pre_loop:bool = False):
        '''
        Run the active learning loop for a specified number of iterations.

        Args:
            skip_pre_loop: (bool, optional): Whether to skip the pre-loop step.
            If True, the pre-loop step will be skipped. Defaults to False.

            parallel_learners: (int, optional): The maximum number of active learner workflows
            to be submitted in parallel. If not provided, the value set during initialization
            will be used. Defaults to 1.
        '''
        if parallel_learners < 2:
            excp = 'parallel_learners must be greater than 1. '
            excp += 'Otherwise use SequentialActiveLearner'
            raise ValueError(excp)

        def _parallel_active_learn():
            super(ParallelActiveLearner, self).teach(max_iter=1,
                                                     skip_pre_loop=skip_pre_loop)

        submitted_learners = []
        for learner in range(parallel_learners):
            async_teach = self.as_async(_parallel_active_learn)
            submitted_learners.append(async_teach())
            print(f'Learner-{learner} is submitted for execution')

        # block/wait for each workflow until it finishes
        [learner.result() for learner in submitted_learners]

class AlgorithmSelector(ActiveLearner):
    """
    AlgorithmSelector is a subclass of ActiveLearner that implements 
    multiple active learning pipelines in parallel, each pipeline is a 
    sequential active learning loop, and uses the same simulation and 
    training tasks, but distinct active learning task.
    """
    def __init__(self, engine: ResourceEngine) -> None:
        ''' 
        Initialize the AlgorithmSelector object.

        Args:
            engine: The ResourceEngine object that manages the resources and
            tasks submission to HPC resources during the active learning loop.
        '''
        super().__init__(engine, register_and_submit=False)
        self.active_learn_functions: Dict[str, Dict] = {}

        # A dictionary to store stats for each active learning pipeline
        # e.g. self.algorithm_results['algo_1'] = {'iterations': 5, 'last_result': 0.01}
        self.algorithm_results: Dict[str, Dict] = {}
        self.best_pipeline_name = None
        self.best_pipeline_stats = None

    def active_learn_task(self, name: str):
        """
        A decorator that registers an active learning task under the given name.
        
        Usage:
        
        @algo_selector.active_learn_task(name='algo_1')
        def active_learn_1(*args):
            ...
        
        @algo_selector.active_learn_task(name='algo_2')
        def active_learn_2(*args):
            ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.active_learn_functions[name] = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs
                }
                if self.register_and_submit:
                    return self._register_task(self.active_learn_functions[name])
            return wrapper
        return decorator

    def teach_and_select(self, max_iter:int = 0, skip_pre_loop:bool = False):
        """
        Run the active learning pipelines in parallel, each using a different AL algorithm,
        for multiple iterations similar to SequentialActiveLearner.
        After that, select the best active learning algorithm

        Args:
            max_iter (int, optional): The maximum number of iterations for each pipeline.
                                      If 0 and a criterion function is provided, it will run
                                      until the criterion is met.
            skip_pre_loop (bool, optional): If True, skip the initial pre-loop step
                                            (simulation+training setup).
        """
        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_functions:
            raise Exception("Simulation, Training, and at least one AL function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or a stop_criterion_function must be provided.")

        def _parallel_active_learn(al_task, name):
            if not skip_pre_loop:
                sim_task, train_task = self._start_pre_loop()
            else:
                sim_task, train_task = (), ()

            if not max_iter:
                iteration_range = itertools.count()
            else:
                iteration_range = range(max_iter)

            stop_value = float('inf')
            num_iterations = 0

            for i in iteration_range:
                print(f'[Pipeline: {al_task["func"].__name__}] Starting Iteration-{i}')
                acl_task = self._register_task(al_task, deps=(sim_task, train_task))

                if self.criterion_function:
                    stop_task = self._register_task(self.criterion_function, deps=acl_task)
                    stop = stop_task.result()

                    should_stop, stop_value = self._check_stop_criterion(stop)
                    if should_stop:
                        num_iterations = i + 1
                        break

                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)
                # Wait for the training to complete before next iteration
                train_task.result()
                num_iterations = i + 1
            
            self.algorithm_results[name] = {
                'iterations': num_iterations,
                'last_result': stop_value
            }

        submitted_learners = []
        for name, al_task in self.active_learn_functions.items():
            async_teach = self.as_async(_parallel_active_learn)
            submitted_learners.append(async_teach(al_task, name))
            print(f'Pipeline-{name} is submitted for execution')

        # block/wait for each pipeline until it finishes
        [learner.result() for learner in submitted_learners]

        if self.algorithm_results:
            # Sort by (iterations, last_result)
            sorted_pipelines = sorted(
                self.algorithm_results.items(),
                key=lambda kv: (kv[1]['iterations'], kv[1]['last_result'])
            )
            self.best_pipeline_name, self.best_pipeline_stats = sorted_pipelines[0]
            print(f"Best algorithm is '{self.best_pipeline_name}' "
                  f"with {self.best_pipeline_stats['iterations']} iteration(s) "
                  f"and final metric result {self.best_pipeline_stats['last_result']}")
        else:
            excp = "No pipeline stats found! Please make sure that at least one active learning algorithm "
            excp += "is used, and the status of each active learning pipeline to make sure that at least "
            excp += "one of them is running successfully!"
            raise ValueError(excp)


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
        from rose.experience import Experience, ExperienceBank, create_experience
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
