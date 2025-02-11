# Define your target machine to run on

Import ROSE main modules:
```python
from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine
```


Define your resource engine, as we described in our previous [Target Resources](target-resources.md) step:
```python
engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ActiveLearner(engine)
```

Now our resource engine is defined, lets define our main AL workflow components:
!!! note
    The Task object is based on the [Radical.Pilot.TaskDescription](https://radicalpilot.readthedocs.io/en/stable/apidoc.html#radical.pilot.TaskDescription), meaning that users can pass any `args` and `kwargs` that the `Radical.Pilot.TaskDescription` can accept to the Task object.

```python
@acl.simulation_task
def simulation(*args):
    return Task(executable=f'python3 sim.py')

@acl.training_task
def training(*args):
    return Task(executable=f'python3 train.py')

@acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'python3 active.py')
```

Optionally, you can specify a metric to monitor and act as a condition to terminate once your results reach the specified value:
!!! tip
    
    Specifying both `@acl.as_stop_criterion` and `max_iter` will cause ROSE to follow whichever constraint is satisfied first.
    Specifying neither will cause an error and eventually a failure to your workflow.


!!! note
    
    ROSE  supports custom/user-defined metrics in addition to a wide range of standard metrics. For a list of standard metrics
    and how to define a custom metrics, please refer to the following link: [Standard Metrics]().

```python
# Defining the stop criterion with a metric (MSE in this case)
@acl.as_stop_criterion(metric_name='mean_squared_error_mse', threshold=0.1)
def check_mse(*args):
    return Task(executable=f'python3 check_mse.py')
```

!!! Warning
    For any metric function like `@acl.as_stop_criterion` the invoked script like `check_mse.py` must return a numerical value.


Finally invoke the tasks and register them with the active learner as a workflow.
!!! note

    In the Sequential Learner, the invocation order of the tasks does not matter as ROSE,
    sequential learner has a predefined order of tasks as follows: `simulation` --> `training` --> `active_learn`.

```python
simul = simulation()
train = training()
active = active_learn()
stop_cond = check_mse()

# Start the teaching loop and break if max_iter = 10 or stop condition is met
acl.teach(max_iter=10)
engine.shutdown()
```