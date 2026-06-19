# Target Machines for Executing AL Workflows
ROSE enables the orchestration of ML Surrogate building workflows on diverse computing resources using [radical.asyncflow](https://github.com/radical-cybertools/radical.asyncflow). Below, we will show how you can specify your `local computer` and `remote HPC machine` as target resources using the `DragonExecutionBackendV3` from [RHAPSODY](https://radical-cybertools.github.io/rhapsody/getting-started/advanced-usage/?h=hpc#multiple-execution-backends).

## Local Computer
For local execution, user can use their desktops, laptops, and their own small clusters to execute their AL workflows as follows:
```python
import os

from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner

engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())

asyncflow = await WorkflowEngine.create(engine)

acl = SequentialActiveLearner(asyncflow)
```

## HPC Resources

To execute AL workflows on HPC machines, users must have an active allocation on the target machine and specify their resource requirements to execute their workflows. Remember, ROSE uses `DragonExecutionBackendV3` from [RHAPSODY](https://github.com/radical-cybertools/rhapsody) (`rhapsody-py`) which is an interface for multip execution and AI runtime system. For more information on how to access, set up, and execute workflows on HPC machines, refer to the following link [ROSE with RHAPSODY on HPC](https://radical-cybertools.github.io/rhapsody/getting-started/advanced-usage/?h=hpc#hpc-workloads-with-dragon):


!!! note
    For any ROSE script that uses `DragonExecutionBackendV3`, user must run the
    rose_script.py with `dragon` binary instead of `python`. Please refer to the following link for more information: [use ROSE with RHAPSODY-Dragon on HPC](https://radical-cybertools.github.io/rhapsody/hpc-machines/#backend-compatibility)

```python
import os

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackendV3

from rose.al.active_learner import SequentialActiveLearner


hpc_engine = await DragonExecutionBackendV3()

asyncflow = await WorkflowEngine.create(hpc_engine)

acl = SequentialActiveLearner(asyncflow)
```
