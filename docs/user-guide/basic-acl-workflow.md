# Define your target machine to run on

Import ROSE main modules:
```python
from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.al.active_learner import SequentialActiveLearner

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend
```


Define your resource engine, as we described in our previous [Target Resources](target-resources.md) step:

```python
engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
asyncflow = await WorkflowEngine.create(engine)

acl = SequentialActiveLearner(asyncflow)
```

Now our resource engine is defined, lets define our main AL workflow components:
!!! note
    The Task object is based on the [Radical.Pilot.TaskDescription](https://radicalpilot.readthedocs.io/en/stable/apidoc.html#radical.pilot.TaskDescription), meaning that users can pass any `args` and `kwargs` that the `Radical.Pilot.TaskDescription` can accept to the Task object.

```python
@acl.simulation_task
async def simulation(*args):
    return 'python3 sim.py'

@acl.training_task
async def training(*args):
    return f'python3 train.py'

@acl.active_learn_task
async def active_learn(*args):
    return f'python3 active.py'
```

!!! tip
ROSE supports defining tasks with python code instead of executables (i.e., python scripts, shell scripts, etc.). To do that, the user have to
pass the `as_executable=False` argument to the decorator as follows:

```python
@acl.simulation_task(as_executable=False)
async def run_simulation(*args) -> dict:
    """Simulate a process and return dummy simulation results."""
    await asyncio.sleep(1)  # Simulate async workload
    results = {
        "input": args,
        "output": [random.random() for _ in range(5)]
    }
    return results

@acl.training_task(as_executable=False)
async def run_training(simulation_results: dict) -> dict:
    """Train a dummy model using simulation results."""
    await asyncio.sleep(1)  # Simulate training time
    model = {
        "weights": [sum(simulation_results["output"]) * 0.1],
        "trained_on": simulation_results["input"]
    }
    return model

@acl.active_learn_task(as_executable=False)
async def run_active_learning(model: dict) -> dict:
    """Perform a dummy active learning step with the trained model."""
    await asyncio.sleep(1)  # Simulate active learning
    selected_samples = [random.randint(0, 100) for _ in range(3)]
    return {
        "model_weights": model["weights"],
        "new_samples": selected_samples
    }
```

Optionally, you can specify a metric to monitor and act as a condition to terminate once your results reach the specified value:

!!! tip

    Specifying both `@acl.as_stop_criterion` and `max_iter` will cause ROSE to follow whichever constraint is satisfied first.
    Specifying neither will cause an error and eventually a failure to your workflow.


!!! note

    ROSE  supports custom/user-defined metrics in addition to a wide range of standard metrics. For a list of standard metrics and how to define a custom metrics, please refer to the following link: [Standard Metrics]().

```python
# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
async def check_mse(*args):
    return f'python3 check_mse.py'
```

!!! Warning
    For any metric function like `@acl.as_stop_criterion` the invoked script like `check_mse.py` must return a numerical value.


Finally invoke the tasks and register them with the active learner as a workflow.
!!! note

    In the Sequential Learner, the invocation order of the tasks is predefined order of tasks as follows: `simulation` --> `training` --> `active_learn`.

```python
# Start the learning loop and break if max_iter = 10 or stop condition is met
await acl.start(max_iter=10)
await asyncflow.shutdown()
```
