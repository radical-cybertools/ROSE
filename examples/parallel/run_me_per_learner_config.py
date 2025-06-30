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

adaptive_sim = acl.create_adaptive_schedule('simulation', 
    lambda i: {
        'kwargs': {
            '--n_labeled': str(100 + i * 50),  # Increase labeled data each iteration
            '--n_features': 2
        }
    })

results = acl.teach(
    parallel_learners=2,
    learner_configs=[
        ParallelLearnerConfig(simulation=adaptive_sim),
        ParallelLearnerConfig(simulation=TaskConfig(kwargs={"--n_labeled": "300",
                                                            "--n_features": 4}))
    ]
)

engine.shutdown()