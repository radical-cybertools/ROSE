import typeguard
import itertools

from typing import Callable
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
                print(f'stop criterion metric: {metric_name} is met, breaking the active learning loop')
                return True
            else:
                return False
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
        # start the initlial step for active learning by
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

        if not max_iter:
            max_iter = itertools.count()

            if not self.criterion_function:
                excp = 'Stop criterion function must be provided if max_iter is not specified'
                raise ValueError(excp)
        else:
            max_iter = range(max_iter)

        # step-2 form the ACL loop and workflow
        for i in max_iter:
            print(f'Starting Iteration-{i}')
            acl_task = self._register_task(self.active_learn_function, deps=(sim_task, train_task))

            if self.criterion_function:
                stop_task = self._register_task(self.criterion_function, deps=acl_task)
                stop = stop_task.result()

                if self._check_stop_criterion(stop):
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
        |                          |                         |
      [Sim]                      [Sim]                     [Sim]
        |                          |                         |
    [Active Learn]           [Active Learn]            [Active Learn]
        |                          |                         |
     [Train]                    [Train]                   [Train]
        |                          |                         |
        v                          v                         v
    -------------------------------------------------------------
                    Parallel Execution of All Learners

                                   |
                                   v
                      (Next Parallel Learner N)
                                   |
                                 [Sim]
                                   |
                             [Active Learn]
                                   |
                                [Train]
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

        for learner in range(parallel_learners):
            print(f'Submitting Learner-{learner} for execution')
            async_teach = self.as_async(_parallel_active_learn)
            async_teach()
