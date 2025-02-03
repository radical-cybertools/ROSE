# Target Machines for Executing ACL Workflows
ROSE enables the execution of ACL workflows on diverse computing resources. Below, we will show how you can specify your `local computer` and `remote HPC machine` as target resources.

## Local Computer
For local execution, user can use their desktops, laptops, and their own small clusters to execute their ACL workflows as follows:
```python
import os

from rose.engine import ResourceEngine
from rose.learner import ActiveLearner

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ActiveLearner(engine)
```

## HPC Resources
To execute ACL workflows on HPC machines, users must have an active allocation on the target machine and specify their resource requirements, as well as the time needed to execute their workflows. Remember, ROSE is based on RADICAL-Pilot. For more information on how to access, set up, and execute workflows on HPC machines, refer to the following link [RADICAL-Pilot Job Submission](https://radicalpilot.readthedocs.io/en/stable/tutorials/submission.html):

```python
import os

from rose.engine import ResourceEngine
from rose.learner import ActiveLearner


hpc_engine = ResourceEngine({'runtime': 30,
                             'cores': 4096,
                             'gpus' : 4,
                             'resource': 'tacc.frontera'})
acl = ActiveLearner(engine)
```