# Learners with Parameterization Tutorial

This tutorial demonstrates how to configure and run multiple learning pipelines concurrently using `ParallelActiveLearner`. You'll learn how to:

- Set up parallel workflows
- Configure each learner independently
- Use per-iteration and adaptive configurations
- Stream per-learner states in real time as each iteration completes

---

!!! note

This approach can be applied for both Active and Reinforcement learners (Sequential and Parallel).

## Example Overview

**This example includes:**

- **Learner 0**: Adaptive config — increasing labeled data, decreasing noise & learning rate
- **Learner 1**: Per-iteration config — specific checkpoints for tuning
- **Learner 2**: Static config — constant settings throughout
- All learners run **concurrently and independently**
- States from all learners are **streamed in real time** via `async for`

---

## How the API Works

`ParallelActiveLearner.start()` returns an **async iterator** that yields an `IterationState`
each time any parallel learner completes an iteration. States arrive in completion order — not
grouped by learner — so you react to results as they happen.

Each `IterationState` carries a `learner_id` (integer index) identifying which learner produced it:

```python
async for state in acl.start(parallel_learners=3, max_iter=10):
    print(f"Learner {state.learner_id} | iter {state.iteration} | MSE {state.metric_value:.4f}")
```

This is the same interface used by `SequentialActiveLearner`, so code that consumes
`IterationState` works identically for both sequential and parallel learners.

---

## Configuration Modes

### Adaptive Configuration

- Receives iteration number `i`
- Labeled data: `100 + i*50`
- Noise: `0.1 * (0.95^i)`
- Learning rate: `0.01 * (0.9^i)`
- Batch size increases gradually, capped at 64

### Per-Iteration Configuration

- Iteration keys (e.g., `0`, `5`, `10`) set exact checkpoints
- `-1` is the fallback/default config
- Ideal for curriculum learning or scheduled tuning

---

## Setup

!!! warning

The entire API of ROSE must be within an `async` context.

### 1. Imports & Engine

```python
import os
import sys

from rose import TaskConfig
from rose import LearnerConfig

from rose.al import ParallelActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend

engine = await RadicalExecutionBackend(
    {'runtime': 30,
     'resource': 'local.localhost'
     }
     )
asyncflow = await WorkflowEngine.create(engine)
acl = ParallelActiveLearner(asyncflow)
code_path = f'{sys.executable} {os.getcwd()}'
```

### 2. Define Workflow Tasks
```python
@acl.simulation_task
async def simulation(*args, **kwargs):
    n_labeled = kwargs.get("--n_labeled", 100)
    n_features = kwargs.get("--n_features", 2)

    return f"{code_path}/sim.py --n_labeled {n_labeled} --n_features {n_features}"

@acl.training_task
async def training(*args, **kwargs):
    learning_rate = kwargs.get("--learning_rate", 0.1)
    return f'{code_path}/train.py --learning_rate {learning_rate}'

@acl.active_learn_task
async def active_learn(*args, **kwargs):
    return f'{code_path}/active.py'

@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
async def check_mse(*args, **kwargs):
    return f'{code_path}/check_mse.py'
```

## Configuration Approaches

### Approach 1: Static Configuration
```python
async for state in acl.start(
    parallel_learners=2,
    max_iter=10,
    learner_configs=[
        LearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2}),
            training=TaskConfig(kwargs={"--learning_rate": "0.01"})
        ),
        LearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4}),
            training=TaskConfig(kwargs={"--learning_rate": "0.005"})
        )
    ]
):
    print(f"[Learner {state.learner_id}] iter={state.iteration} | MSE={state.metric_value}")
```

### Approach 2: Per-Iteration Configuration
```python
async for state in acl.start(
    parallel_learners=3,
    max_iter=15,
    learner_configs=[
        LearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2})
        ),
        LearnerConfig(
            simulation={
                0: TaskConfig(kwargs={"--n_labeled": "100"}),
                5: TaskConfig(kwargs={"--n_labeled": "200"}),
                10: TaskConfig(kwargs={"--n_labeled": "400"}),
                -1: TaskConfig(kwargs={"--n_labeled": "500"})
            },
            training={
                0: TaskConfig(kwargs={"--learning_rate": "0.01"}),
                5: TaskConfig(kwargs={"--learning_rate": "0.005"}),
                -1: TaskConfig(kwargs={"--learning_rate": "0.001"})
            }
        ),
        None  # Default to base task behavior
    ]
):
    print(f"[Learner {state.learner_id}] iter={state.iteration} | MSE={state.metric_value}")
```

!!! tip "Per-Iteration Config Keys"
Use numeric keys for specific iterations and -1 as a fallback.


### Approach 3: Adaptive Configuration
```python
adaptive_sim = acl.create_adaptive_schedule('simulation',
    lambda i: {
        'kwargs': {
            '--n_labeled': str(100 + i * 50),
            '--n_features': 2,
            '--noise_level': str(0.1 * (0.95 ** i))
        }
    })

adaptive_train = acl.create_adaptive_schedule('training',
    lambda i: {
        'kwargs': {
            '--learning_rate': str(0.01 * (0.9 ** i)),
            '--batch_size': str(min(64, 32 + i * 4))
        }
    })

async for state in acl.start(
    parallel_learners=2,
    max_iter=20,
    learner_configs=[
        LearnerConfig(simulation=adaptive_sim, training=adaptive_train),
        LearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4}),
            training=TaskConfig(kwargs={"--learning_rate": "0.005"})
        )
    ]
):
    print(f"[Learner {state.learner_id}] iter={state.iteration} | MSE={state.metric_value}")
```

### Full Example: All Approaches Combined

```python
adaptive_sim = acl.create_adaptive_schedule('simulation',
    lambda i: {
        'kwargs': {
            '--n_labeled': str(100 + i * 50),
            '--n_features': 2
        }
    })

# Collect the final state per learner if needed
final_states = {}

async for state in acl.start(
    parallel_learners=3,
    max_iter=15,
    learner_configs=[
        LearnerConfig(simulation=adaptive_sim),  # Adaptive
        LearnerConfig(                           # Per-iteration
            simulation={
                0: TaskConfig(kwargs={"--n_labeled": "150", "--n_features": 3}),
                7: TaskConfig(kwargs={"--n_labeled": "250", "--n_features": 3}),
                -1: TaskConfig(kwargs={"--n_labeled": "400", "--n_features": 3})
            }
        ),
        LearnerConfig(                           # Static
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4})
        )
    ]
):
    print(f"[Learner {state.learner_id}] iter={state.iteration} | MSE={state.metric_value}")
    final_states[state.learner_id] = state  # keep last state per learner

await acl.shutdown()
```

### Execution Details

!!! note "Concurrent Execution"
All learners run in parallel and independently. States are yielded in arrival order — whichever learner finishes an iteration first yields next. The loop completes when all learners either reach `max_iter` or meet their stop criterion.

!!! note "Identifying the Source Learner"
Each `IterationState` has a `learner_id` field (integer index, 0-based) so you can distinguish
which learner produced each state inside the loop.

!!! warning "Stop Criteria"
Each learner evaluates its own stop condition. One learner stopping does not affect others.

!!! tip "Performance Tip"
Match the number of parallel learners to available resources. Overloading can slow down execution.


### Quick Reference
`create_iteration_schedule(task_name, schedule)`

```python
schedule = {
    0: {'kwargs': {'--learning_rate': '0.01'}},
    5: {'kwargs': {'--learning_rate': '0.005'}},
    -1: {'kwargs': {'--learning_rate': '0.001'}}
}
config = acl.create_iteration_schedule('training', schedule)
```


`create_adaptive_schedule(task_name, fn)`

```python
def lr_decay(i):
    return {'kwargs': {'--learning_rate': str(0.01 * (0.95 ** i))}}

adaptive_config = acl.create_adaptive_schedule('training', lr_decay)
```


## Next Steps

- Try different active learning algorithms per learner

- Use per-iteration configs to design curriculum learning

- Run parameter sweeps across acquisition functions or model architectures

- Scale learners to match compute resources
