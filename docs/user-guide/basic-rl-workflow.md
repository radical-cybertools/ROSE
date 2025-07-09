## Define your target machine to run on

Import ROSE main modules:
```python
from rose.rl.learner import SequentialReinforcementLearner
from rose.engine import Task, ResourceEngine
```

Define your resource engine, as we described in our previous [Target Resources](target-resources.md) step:
```python
engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
rl = SequentialReinforcementLearner(engine)
```

Now our resource engine is defined, lets define our main RL workflow components:

```python
@rl.environment_task
def environment(*args):
    return Task(executable=f'python3 environment.py')

@rl.update_task
def update(*args):
    return Task(executable=f'python3 update.py')

@rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=200, operator=GREATER_THAN_THRESHOLD)
def check_reward(*args):
    return Task(executable='python3 check_reward.py')
```

!!! Warning
    For any metric function like `@rl.as_stop_criterion` the invoked script like `check_reward.py` must return a numerical value.

Finally invoke the tasks and register them with the reinforcement learner as a workflow:

```python
env = environment()
upd = update()
stop_cond = check_reward()

# Start the RL training loop and break when stop condition is met
rl.learn()
# You can also specify maximum iterations
rl.learn(max_iter=10)
engine.shutdown()
```
