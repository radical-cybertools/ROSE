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
import asyncio

from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.al.active_learner import SequentialActiveLearner

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

async def main():
    execution_engine = await RadicalExecutionBackend(
        {'runtime': 30,
        'resource': 'local.localhost'}
        )

    asyncflow = await WorkflowEngine.create(execution_engine)
    acl = SequentialActiveLearner(asyncflow)

    @acl.simulation_task
    async def simulation(*args):
        return f'python3 sim.py'

    @acl.training_task
    async def training(*args):
        return f'python3 train.py'

    @acl.active_learn_task
    async def active_learn(*args):
        return f'python3 active.py'

    # Start the teaching loop and break if max_iter = 10
    await acl.teach(max_iter=10)
    await asyncflow.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```
