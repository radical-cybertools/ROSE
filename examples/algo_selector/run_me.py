import os
import sys

from rose.learner import ParallelActiveLearningAlgoSelector
from rose.engine import Task, ResourceEngine
from rose.metrics import MEAN_SQUARED_ERROR_MSE

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
algo_selector = ParallelActiveLearningAlgoSelector(engine)
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the simulation task
@algo_selector.simulation_task
def simulation(*args):
    return Task(executable=f'{code_path}/sim.py')

# Define and register the training task
@algo_selector.training_task
def training(*args):
    return Task(executable=f'{code_path}/train.py')

# Define and register Multiple AL tasks
@algo_selector.active_learn_task(name='algo_1')
def active_learn_1(*args):
    return Task(executable=f'{code_path}/active_1.py')

@algo_selector.active_learn_task(name='algo_2')
def active_learn_2(*args):
    return Task(executable=f'{code_path}/active_2.py')

# Defining the stop criterion with a metric (MSE in this case)
@algo_selector.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.01)
def check_mse(*args):
    return Task(executable=f'{code_path}/check_mse.py')

# Now, call the tasks and teach
simul = simulation()
train = training()
active_1 = active_learn_1()
active_2 = active_learn_2()
stop_cond = check_mse()

# Start the teaching process
algo_selector.teach(max_iter=4)
engine.shutdown()
