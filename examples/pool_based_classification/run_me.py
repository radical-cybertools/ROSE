import os
import sys

from rose.metrics import MODEL_ACCURACY
from rose.engine import Task, ResourceEngine
from rose.learner import SequentialActiveLearner


engine = ResourceEngine({'cores': 64, 'runtime': 60, 'resource': 'ncsa.delta', 'access_schema': 'interactive'})

learner = SequentialActiveLearner(engine)
code_path = f'{sys.executable} /u/alsaadi1/RADICAL/ROSE/examples/pool_based_classification'

# Define and register the simulation task
@learner.simulation_task
def simulation(*args):
    return Task(cores_per_rank=4,
                executable=f'{code_path}/simulation.py')

# Define and register the training task
@learner.training_task
def training(*args):
    return Task(cores_per_rank=12,
                executable=f'{code_path}/training.py')

# Define and register the active learning task
@learner.active_learn_task
def active_learn(*args):
    return Task(cores_per_rank=12,
                executable=f'{code_path}/active_learn.py')

@learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=75)
def check_acc(*args):
    return Task(cores_per_rank=12,
                executable=f'{code_path}/check_metric.py')

# Now, call the tasks and teach
simul = simulation()
train = training()
active = active_learn()
check = check_acc()

# Start the teaching process
learner.teach(max_iter=10)

res = learner.get_result(task_name='check_acc')
print(res)

engine.shutdown()
