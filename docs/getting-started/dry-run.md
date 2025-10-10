## 🧪 Dry Run with ROSE
ROSE supports dry runs to help you validate your machine learning surrogate workflows without actually executing simulations or launching jobs on HPC systems. This is useful for debugging workflow structure, verifying task generation, and ensuring that everything is connected properly before committing compute resources.


## Example:

Running a Dry Run with `SequentialActiveLearner`
Below is a minimal example showing how to set up and perform a dry run using the `SequentialActiveLearner` in ROSE.

```python
import os
import sys
import asyncio

from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.al.active_learner import SequentialActiveLearner

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend


async def rose_al():
    # Enable dry run in the workflow engine
    asyncflow = await WorkflowEngine.create(engine, dry_run=True)

    # Create an active learner with the workflow engine
    acl = SequentialActiveLearner(asyncflow)

    # Path to your training script or code
    code_path = f'{sys.executable} {os.getcwd()}'

    # Now use `acl` to define and simulate the workflow...
    # (e.g., acl.run(...), acl.sample(...), etc.)
    # During dry run, tasks will be logged but not executed.
```

Run the async function
```python
asyncio.run(rose_al())
```

## ✅ What Happens in a Dry Run?
* Tasks are created and scheduled, but not actually run.

* ROSE logs the task definitions, dependencies, and flow structure.

* Useful for catching configuration errors or invalid paths before real execution.