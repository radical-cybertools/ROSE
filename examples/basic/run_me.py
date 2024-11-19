import os

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine
from rose.metrics import MEAN_SQUARED_ERROR_MSE

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ActiveLearner(engine)
code_path = f'python3 {os.getcwd()}'

# Define and register the simulation task
@acl.simulation_task
def simulation(*args):
    return Task(executable=f'{code_path}/sim.py')

# Define and register the training task
@acl.training_task
def training(*args):
    return Task(executable=f'{code_path}/train.py')

# Define and register the active learning task
@acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'{code_path}/active.py')

# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
def check_mse(*args):
    return Task(executable=f'{code_path}/check_mse.py')

# Now, call the tasks and teach
simul = simulation()
train = training()
active = active_learn()
stop_cond = check_mse()

# Start the teaching process
acl.teach(max_iter=10)
engine.shutdown()
