import os
import sys
from rose.learner import ActiveLearner
from rose.metrics import MODEL_ACCURACY
from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})

learner = ActiveLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the simulation task
@learner.simulation_task
def simulation(*args):
    return Task(executable=f'{code_path}/simulation.py')

# Define and register the training task
@learner.training_task
def training(*args):
    return Task(executable=f'{code_path}/training.py')

# Define and register the active learning task
@learner.active_learn_task
def active_learn(*args):
    return Task(executable=f'{code_path}/active_learn.py')

# Defining the stop criterion with a metric (MSE in this case)
@learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=0.99)
def check_accuracy(*args):
    return Task(executable=f'{code_path}/check_accuracy.py')

def teach():
    # 10 iterations of active learn
    for acl_iter in range(10):
        print(f'Starting Iteration-{acl_iter}')
        simul = simulation()
        train = training(simul)
        active = active_learn(simul, train)

        if check_accuracy(active):
            print('Accuracy met the threshold')
            break

teach()
engine.shutdown()
