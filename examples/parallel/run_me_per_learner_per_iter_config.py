import sys, os

from rose.learner import TaskConfig
from rose.learner import ParallelActiveLearner
from rose.learner import ParallelLearnerConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ParallelActiveLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the simulation task
@acl.simulation_task
def simulation(*args, **kwargs):
    return Task(executable=f'{code_path}/sim.py')

# Define and register the training task
@acl.training_task
def training(*args, **kwargs):
    return Task(executable=f'{code_path}/train.py')

# Define and register the active learning task
@acl.active_learn_task
def active_learn(*args, **kwargs):
    return Task(executable=f'{code_path}/active.py')

# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
def check_mse(*args, **kwargs):
    return Task(executable=f'{code_path}/check_mse.py')

# Run with custom configs
results = acl.teach(
    parallel_learners=3,
    learner_configs=[
        # Learner 0: Same config for all iterations (your current pattern)
        ParallelLearnerConfig(simulation=TaskConfig(kwargs={"--n_labeled": "200",
                                                            "--n_features": 2})),
        
        # Learner 1: Different configs per iteration
        ParallelLearnerConfig(
            simulation={
                0: TaskConfig(kwargs={"--n_labeled": "100", "--n_features": 2}),
                5: TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2}),
                10: TaskConfig(kwargs={"--n_labeled": "400", "--n_features": 2}),
                -1: TaskConfig(kwargs={"--n_labeled": "500", "--n_features": 2})  # default
            }
        ),
        
        # Learner 2: No custom config (uses base functions)
        None
    ]
)

engine.shutdown()
