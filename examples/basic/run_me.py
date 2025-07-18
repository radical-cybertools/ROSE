import os
import sys

from rose import TaskConfig
from rose import LearnerConfig

from rose.al import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.asyncflow import Task, ThreadExecutionBackend


engine = ThreadExecutionBackend({})
acl = SequentialActiveLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the simulation task
@acl.simulation_task
def simulation(*args, **kwargs):
    return f'{code_path}/sim.py'

# Define and register the training task
@acl.training_task
def training(*args, **kwargs):
    return f'{code_path}/train.py'

# Define and register the active learning task
@acl.active_learn_task
def active_learn(*args, **kwargs):
    return f'{code_path}/active.py'

# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
def check_mse(*args):
    return f'{code_path}/check_mse.py'

# Create configurations for different parallel learners
configs = LearnerConfig(
        simulation=TaskConfig(args=(10,), kwargs={'batch_size': 128}),
        # Per-iteration training configuration
        training={
            0: TaskConfig(kwargs={'lr': 0.01, 'optimizer': 'adam'}),
            5: TaskConfig(kwargs={'lr': 0.005, 'optimizer': 'adam'}),
            10: TaskConfig(kwargs={'lr': 0.001, 'optimizer': 'adam'}),
            -1: TaskConfig(kwargs={'lr': 0.0001, 'optimizer': 'adam'})  # default
        },
        active_learn=TaskConfig(kwargs={'strategy': 'random'})
    )

# Start the teaching process
acl.teach(learner_config=configs)
engine.shutdown()
