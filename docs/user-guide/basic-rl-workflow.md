## Define your target machine to run on

Import ROSE main modules:
```python
from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

from rose.metrics import GREATER_THAN_THRESHOLD
from rose.rl.reinforcement_learner import SequentialReinforcementLearner
```

Define your resource engine, as we described in our previous [Target Resources](target-resources.md) step:
```python
engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
asyncflow = await WorkflowEngine.create(engine)

rl = SequentialReinforcementLearner(asyncflow)
```

Now our resource engine is defined, lets define our main RL workflow components:

```python
@rl.environment_task
async def environment(*args):
    return f'python3 environment.py'

@rl.update_task
async def update(*args):
    return f'python3 update.py'

@rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=200, operator=GREATER_THAN_THRESHOLD)
async def check_reward(*args):
    return 'python3 check_reward.py'
```

!!! Warning
    For any metric function like `@rl.as_stop_criterion` the invoked script like `check_reward.py` must return a numerical value.

Finally invoke the tasks and register them with the reinforcement learner as a workflow:

```python
# Start the RL training loop and break when stop condition is met
await rl.learn()

# You can also specify maximum iterations
await rl.learn(max_iter=10)
await asyncflow.shutdown()
```
