# ROSE
What is ROSE:

The RADICAL Optimal & Smart-Surrogate Explorer (ROSE) toolkit is a framework for supporting the concurrent and adaptive execution of simulation and surrogate training and selection tasks on High-Performance Computing (HPC) resources.
ROSE is a Python package that provides tools to facilitate the development of active learning methods for scientific applications. It allows users to define simulation and surrogate training tasks and automatically manage their execution on HPC resources. 

ROSE also provides tools to facilitate the selection of the best surrogate model for a given simulation based on performance metrics.

ROSE uses RADICAL-Cybertools -- middleware building blocks to facilitate the development of sophisticated scientific workflows on HPC resources. 

### How to install:

`pip install .`


### Documentation:

The complete documentation is available [here](https://radical-cybertools.github.io/ROSE/).

For tutorials and walkthrough notebooks please check [here](examples)



### Basic usage

```python
import os

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ActiveLearner(engine)

@acl.simulation_task
def simulation(*args):
    return Task(executable=f'python3 sim.py')

@acl.training_task
def training(*args):
    return Task(executable=f'python3 train.py')

@acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'python3 active.py')

simulation = simulation()
training = training()
active_learn = active_learn()

# Start the teaching loop and break if max_iter = 10
acl.teach(max_iter=10)
engine.shutdown()
```
