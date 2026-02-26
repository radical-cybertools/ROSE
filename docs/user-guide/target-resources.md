# Target Machines for Executing AL Workflows
ROSE enables the orchestration of ML Surrogate building workflows on diverse computing resources using [radical.asyncflow](https://github.com/radical-cybertools/radical.asyncflow). Below, we will show how you can specify your `local computer` and `remote HPC machine` as target resources using the `RadicalExecutionBackend` from [RHAPSODY](https://github.com/radical-cybertools/rhapsody).

## Local Computer
For local execution, user can use their desktops, laptops, and their own small clusters to execute their AL workflows as follows:
```python
import os

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend

from rose.al.active_learner import SequentialActiveLearner

engine = await RadicalExecutionBackend(
    {'runtime': 30,
    'resource': 'local.localhost'})

asyncflow = await WorkflowEngine.create(engine)

acl = SequentialActiveLearner(asyncflow)
```

## HPC Resources
To execute AL workflows on HPC machines, users must have an active allocation on the target machine and specify their resource requirements, as well as the time needed to execute their workflows. Remember, ROSE uses `RadicalExecutionBackend` from [RHAPSODY](https://github.com/radical-cybertools/rhapsody) (`rhapsody-py`) which is an interface for RADICAL-Pilot runtime system. For more information on how to access, set up, and execute workflows on HPC machines, refer to the following link [RADICAL-Pilot Job Submission](https://radicalpilot.readthedocs.io/en/stable/tutorials/submission.html):

```python
import os

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend

from rose.al.active_learner import SequentialActiveLearner


hpc_engine = await RadicalExecutionBackend(
    {'runtime': 30, 'cores': 4096,
     'gpus' : 4, 'resource': 'tacc.frontera'})

asyncflow = await WorkflowEngine.create(hpc_engine)

acl = SequentialActiveLearner(asyncflow)
```