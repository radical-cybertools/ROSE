from rose import ResourceEngine
from rose import WorkflowEngine
from typing import Callable
from functools import wraps


class ActiveLearner(WorkflowEngine):
    def __init__(self, engine: ResourceEngine) -> None:

        self.criterion_function = {}
        self.training_function = {}
        self.simulation_function = {}
        self.active_learn_function = {}

        super().__init__(engine)


    def simulation_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.simulation_function = {'func':func,
                                        'args':args,
                                        'kwargs':kwargs}
            return func
        return wrapper


    def training_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.training_function = {'func':func,
                                       'args':args,
                                       'kwargs':kwargs}
            return func
        return wrapper

    def active_learn_task(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.active_learn_function = {'func':func,
                                          'args':args,
                                          'kwargs':kwargs}
            return func
        return wrapper


    def as_stop_criterion(self, func:Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.criterion_function = {'func':func,
                                       'args':args,
                                       'kwargs':kwargs}
            return func
        return wrapper


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


    def _invok_stop_criterion(self):
        func = self.criterion_function['func']
        args = self.criterion_function['args'] + args
        kwargs = self.criterion_function['kwargs']
        return func(*args, **kwargs)


    def _start_pre_step(self):
        
        # start the initlial step for active learning by 
        # defining and setting simulation and training tasks
        sim_task = self._register_task(self.simulation_function)
        train_task = self._register_task(self.training_function,
                                         deps=sim_task)
        return sim_task, train_task


    def _start_main_step(self):

        acl_task = self._register_task(self.active_learn_function)
        sim_task = self._register_task(self.simulation_function)
        train_task = self._register_task(self.training_function)

        return acl_task, sim_task, train_task


    def teach(self, max_iter: int = 0):

        if not self.simulation_function or \
              not self.training_function or \
                not self.active_learn_function:
            raise Exception("Simulation and Training function must be set!")

        if not max_iter and not self.criterion_function:
            raise Exception("Either max_iter or stop_criterion_function must be provided.")

        # step-1 invoke the pre_step only once
        sim_task, train_task = self._start_pre_step()

        # step-2 form the ACL loop and workflow
        if max_iter and not self.criterion_function:
            print('No stop condition was specified running for max_iter times')
            # Run for a fixed number of iterations
            for i in range(max_iter):
                print(f'Iteration-{i}')
                acl_task = self._register_task(self.active_learn_function, deps=train_task)
                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)

                train_task.result()

        elif self.criterion_function and not max_iter:
            print('No max_iter was specified running until stop condition is met!')
            # Run indefinitely until stop criterion is met
            while True:
                acl_task = self._register_task(self.active_learn_function, deps=train_task)
                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)

                train_task.result()

                result = self._invok_stop_criterion()
                if result:
                    break

        elif max_iter and self.criterion_function:
            print('Both stop condition and max_iter were specified, running until they are satisified')
            # Run up to max_iter or until stop criterion is met
            for _ in range(max_iter):
                print(f'Iteration-{i}\n')
                acl_task = self._register_task(self.active_learn_function, deps=train_task)
                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)

                train_task.result()

                result = self._invok_stop_criterion()
                if result:
                    break
