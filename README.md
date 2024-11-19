# ROSE
What is ROSE:

RADICAL Optimal & Smart-Surrogate Explorer (ROSE) toolkit is a framework to support the concurrent and adaptive executions of simulation and surrogate training and selection tasks on High Performance Computing (HPC) resources.
ROSE is a Python package that provides a set of tools to facilitate the development of active learning workflows for scientific simulations. It allows users to define simulation and surrogate training tasks, and to automatically manage their execution on HPC resources. 

ROSE also provides a set of tools to facilitate the selection of the best surrogate model for a given simulation, based on a set of performance metrics.

ROSE is built on top of the RADICAL-Cybertools framework, which provides a set of tools to facilitate the development of scientific workflows on HPC resources. 

RADICAL-Cybertools provides a set of tools to manage the execution of tasks on HPC resources, to transfer data between resources, and to monitor the execution of tasks.

How to install:
`pip install .`

How to use:

1- Import the necessary pacakges
```python
import os

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine
```

2- Define the resource engine and Active learner
```python
engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ActiveLearner(engine)
```

3- Define the simulation task
```python
@acl.simulation_task
def simulation(*args):
    return Task(executable=f'python3 sim.py')
```

4- Define the surrogate training task
```python
@acl.training_task
def training(*args):
    return Task(executable=f'python3 train.py')
```

5- Define and register the active learning task
```python
@acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'python3 active.py')
```

6- Optionally you can define a stop criterion or run for MAX ITERATIONS
```python
# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name='mean_squared_error_mse', threshold=0.1)
def check_mse(*args):
    return Task(executable=f'python3 check_mse.py 0.25')
```

7- Finally invoke the tasks and register them with the active learner as a workflow
```python
simul = simulation()
train = training()
active = active_learn()
stop_cond = check_mse()

# Start the teaching loop and break if max_iter = 10 or stop condition is met
acl.teach(max_iter=10)
engine.shutdown()
```