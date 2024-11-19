import typeguard

from typing import Callable
from functools import wraps

from .engine import ResourceEngine
from .engine import WorkflowEngine

METRIC_MAP = {
    "model_accuracy": ">=",                  # Accuracy should be greater than or equal to a threshold
    "query_efficiency": ">",                 # Efficiency should be greater than a threshold
    "labeling_cost": "<",                    # Cost should be less than a threshold
    "diversity_of_selected_samples": ">",    # Diversity should be greater than a threshold
    "uncertainty": "<",                      # Uncertainty should be less than a threshold
    "query_reduction": ">",                  # Query reduction should be greater than a threshold
    "area_under_learning_curve_auc": ">",    # AUC should be greater than a threshold
    "expected_model_improvement_emi": ">",   # Improvement should be greater than a threshold
    "sampling_bias": "<",                    # Bias should be less than a threshold
    "prediction_confidence": ">",            # Confidence should be greater than a threshold
    "class_distribution_shift": "<",         # Distribution shift should be less than a threshold
    "cumulative_error_reduction": ">",       # Error reduction should be greater than a threshold
    "query_selection_strategy_performance": ">",  # Strategy performance should be greater than a threshold
    "number_of_iterations": "<",             # Iterations should be less than a threshold
    "f1_score": ">",                         # F1 Score should be greater than a threshold
    "precision": ">",                        # Precision should be greater than a threshold
    "recall": ">",                           # Recall should be greater than a threshold
    "information_gain": ">",                 # Information gain should be greater than a threshold
    "time_to_convergence": "<",              # Time to convergence should be less than a threshold
    "model_complexity": "<",                 # Model complexity should be minimized (less complex models preferred)
    "data_labeling_redundancy": "<",         # Redundancy in labeled data should be minimized
    "generalization_error": "<",             # Generalization error should be minimized (closer to zero)
    "learning_rate": "<",                    # Learning rate should be minimized for faster convergence
    "training_time": "<",                    # Training time should be minimized
    "confidence_interval_width": "<",        # Confidence intervals should be narrow for better certainty
    "overfitting": "<",                      # Overfitting should be minimized (avoid excessively complex models)
    "active_learning_efficiency": ">",       # Efficiency of the active learning loop (should increase over time)
    "exploration_vs_exploitation": ">",      # Balance exploration and exploitation (higher value preferred)
    "mean_squared_error_mse": "<",           # MSE should be minimized (closer to zero)
}


class ActiveLearner(WorkflowEngine):
    
    @typeguard.typechecked
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

    def as_stop_criterion(self, metric_name: str,
                                threshold: float):
        # This is the outer function that takes arguments like metric_name and threshold
        def decorator(func: Callable):  # This is the actual decorator function that takes func
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Store the relevant information in self.criterion_function
                self.criterion_function = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                    'threshold': threshold,
                    'metric_name': metric_name}
                return func
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

    def compare_metric(self, metric_name, metric_value, threshold):
        operator = METRIC_MAP.get(metric_name)
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


    def _start_pre_step(self):
        
        # start the initlial step for active learning by 
        # defining and setting simulation and training tasks
        sim_task = self._register_task(self.simulation_function)
        train_task = self._register_task(self.training_function,
                                         deps=sim_task)
        return sim_task, train_task
    

    def _check_stop_criterion(self, stop_task_result):

        metric_value = eval(stop_task_result)
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            threshold = self.criterion_function['threshold']
            metric_name = self.criterion_function['metric_name']

            if self.compare_metric(metric_name, metric_value, threshold):
                print(f'stop criterion metric {metric_name} is met, breaking the active learning loop')
                return True
            else:
                return False
        else:
            raise TypeError(f'Stop criterion script must return a numerical got {type(metric_value)} instead')

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

                # block/wait for each workflow until it finishes
                train_task.result()

        elif self.criterion_function and not max_iter:
            print('No max_iter was specified running until stop condition is met!')
            # Run indefinitely until stop criterion is met
            while True:
                acl_task = self._register_task(self.active_learn_function, deps=train_task)
                
                stop_task = self._register_task(self.criterion_function, deps=acl_task)
                stop = stop_task.result()

                if self._check_stop_criterion(stop):
                    break

                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)

        elif max_iter and self.criterion_function:
            print('Both stop condition and max_iter were specified, running until they are satisified')
            # Run up to max_iter or until stop criterion is met
            for i in range(max_iter):
                print(f'Iteration-{i}\n')
                acl_task = self._register_task(self.active_learn_function, deps=(sim_task,
                                                                                 train_task))

                stop_task = self._register_task(self.criterion_function, deps=acl_task)
                stop = stop_task.result()

                if self._check_stop_criterion(stop):
                    break

                sim_task = self._register_task(self.simulation_function, deps=acl_task)
                train_task = self._register_task(self.training_function, deps=sim_task)
