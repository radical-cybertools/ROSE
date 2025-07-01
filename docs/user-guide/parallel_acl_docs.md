**This example demonstrates:**
- **Learner 0**: Uses adaptive configuration with growing labeled data
- **Learner 1**: Uses per-iteration configuration with specific checkpoints
- **Learner 2**: Uses simple static configuration
- All learners run concurrently and independently**How adaptive configurations work:**
- The lambda function receives the iteration number `i` as input
- Labeled data increases linearly: 100, 150, 200, 250... 
- Noise level decreases exponentially: 0.1, 0.095, 0.09...
- Learning rate decays exponentially: 0.01, 0.009, 0.0081...
- Batch size increases gradually but caps at 64**Understanding the per-iteration syntax:**
- Iteration numbers (0, 5, 10) specify exact iterations
- `-1` serves as a fallback for all other iterations
- Learner 0 uses static configuration (same as Approach 1)
- Learner 1 gradually increases labeled data and decreases learning rate
- Learner 2 uses the base function configurations (no custom overrides)**What's happening here?**
- Learner 0 uses 200 labeled samples and 2 features with a learning rate of 0.01
- Learner 1 uses 300 labeled samples and 4 features with a learning rate of 0.005
- Both learners run for up to 10 iterations or until the MSE threshold is met# Parallel Active Learning Tutorial

Welcome to the Parallel Active Learning tutorial! In this guide, you'll learn how to run multiple active learning pipelines concurrently using the `ParallelActiveLearner`. This powerful feature allows you to experiment with different parameters, compare approaches, or scale your active learning workloads across multiple parallel streams.

By the end of this tutorial, you'll understand how to:
- Set up parallel active learning workflows
- Configure different parameters for each learner
- Use per-iteration configurations for fine-grained control
- Create adaptive configurations that change over time

## Step 1: Setting Up Your Environment

First, let's import the necessary ROSE modules for parallel active learning:
```python
import sys, os
from rose.learner import TaskConfig
from rose.learner import ParallelActiveLearner
from rose.learner import ParallelLearnerConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.engine import Task, ResourceEngine
```

Next, create your resource engine and parallel active learner instance:
```python
engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
acl = ParallelActiveLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'
```

## Step 2: Define Your Workflow Components

Now let's define the core components of your active learning workflow. These are the same as in the sequential learner, but will be used across multiple parallel streams:

!!! note
    The Task object is based on the [Radical.Pilot.TaskDescription](https://radicalpilot.readthedocs.io/en/stable/apidoc.html#radical.pilot.TaskDescription), meaning that users can pass any `args` and `kwargs` that the `Radical.Pilot.TaskDescription` can accept to the Task object.

```python
# Define and register the simulation task
@acl.simulation_task
def simulation(*args, **kwargs):
    return Task(executable=f'{code_path}/sim.py')

# Define and register the training task
@acl.training_task
def training(*args, **kwargs):
    return Task(executable=f'{code_path}/train.py')

# Define and register the active learning task
@acl.active_learn_task
def active_learn(*args, **kwargs):
    return Task(executable=f'{code_path}/active.py')

# Optional: Define stop criterion
@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
def check_mse(*args, **kwargs):
    return Task(executable=f'{code_path}/check_mse.py')
```

## Step 3: Choose Your Configuration Approach

The `ParallelActiveLearner` offers three powerful approaches for configuring your parallel learners. Let's explore each one:

### Approach 1: Simple Per-Learner Configuration (Recommended for Beginners)

This is the most straightforward approach. Each learner uses the same configuration throughout all iterations, but different learners can have different configurations:

```python
# Run with custom configs - each learner uses same config for all iterations
results = acl.teach(
    parallel_learners=2,
    max_iter=10,
    learner_configs=[
        ParallelLearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2}),
            training=TaskConfig(kwargs={"--learning_rate": "0.01"})
        ),
        ParallelLearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4}),
            training=TaskConfig(kwargs={"--learning_rate": "0.005"})
        )
    ]
)
```

### Approach 2: Per-Iteration Configuration (Advanced)

For more sophisticated experiments, you can specify different configurations for specific iterations. This is perfect for implementing curriculum learning or parameter scheduling:

```python
# Mix simple and per-iteration configs
results = acl.teach(
    parallel_learners=3,
    max_iter=15,
    learner_configs=[
        # Learner 0: Static config (same as Option 1)
        ParallelLearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2})
        ),
        
        # Learner 1: Per-iteration configuration
        ParallelLearnerConfig(
            simulation={
                0: TaskConfig(kwargs={"--n_labeled": "100", "--n_features": 2}),
                5: TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2}),
                10: TaskConfig(kwargs={"--n_labeled": "400", "--n_features": 2}),
                -1: TaskConfig(kwargs={"--n_labeled": "500", "--n_features": 2})  # default for remaining iterations
            },
            training={
                0: TaskConfig(kwargs={"--learning_rate": "0.01"}),
                5: TaskConfig(kwargs={"--learning_rate": "0.005"}),
                -1: TaskConfig(kwargs={"--learning_rate": "0.001"})  # default
            }
        ),
        
        # Learner 2: Use base function configurations (no custom config)
        None
    ]
)
```

!!! tip "Per-Iteration Configuration Keys"
    - Use specific iteration numbers (0, 1, 2, ...) for exact iteration matches
    - Use `-1` or `'default'` as fallback configuration for unspecified iterations
    - Configurations are applied in order: exact match → default → base function

### Approach 3: Adaptive/Dynamic Configuration (Expert Level)

The most powerful approach uses functions to compute configurations dynamically based on the iteration number. This enables complex adaptive behaviors:

```python
# Create adaptive simulation config that changes each iteration
adaptive_sim = acl.create_adaptive_schedule('simulation', 
    lambda i: {
        'kwargs': {
            '--n_labeled': str(100 + i * 50),  # Increase labeled data each iteration
            '--n_features': 2,
            '--noise_level': str(0.1 * (0.95 ** i))  # Decrease noise over time
        }
    })

# Create adaptive training config with learning rate decay
adaptive_train = acl.create_adaptive_schedule('training',
    lambda i: {
        'kwargs': {
            '--learning_rate': str(0.01 * (0.9 ** i)),  # Exponential decay
            '--batch_size': str(min(64, 32 + i * 4))    # Increasing batch size
        }
    })

results = acl.teach(
    parallel_learners=2,
    max_iter=20,
    learner_configs=[
        ParallelLearnerConfig(
            simulation=adaptive_sim,
            training=adaptive_train
        ),
        ParallelLearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4}),
            training=TaskConfig(kwargs={"--learning_rate": "0.005"})
        )
    ]
)
```

## Step 4: Putting It All Together

Let's create a comprehensive example that demonstrates all three approaches in a single workflow:

```python
import sys, os
from rose.learner import TaskConfig
from rose.learner import ParallelActiveLearner
from rose.learner import ParallelLearnerConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.engine import Task, ResourceEngine

# Setup
engine = ResourceEngine({'runtime': 30, 'resource': 'local.localhost'})
acl = ParallelActiveLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'

# Define workflow tasks
@acl.simulation_task
def simulation(*args, **kwargs):
    return Task(executable=f'{code_path}/sim.py')

@acl.training_task
def training(*args, **kwargs):
    return Task(executable=f'{code_path}/train.py')

@acl.active_learn_task
def active_learn(*args, **kwargs):
    return Task(executable=f'{code_path}/active.py')

@acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
def check_mse(*args, **kwargs):
    return Task(executable=f'{code_path}/check_mse.py')

# Create adaptive configuration
adaptive_sim = acl.create_adaptive_schedule('simulation', 
    lambda i: {
        'kwargs': {
            '--n_labeled': str(100 + i * 50),
            '--n_features': 2
        }
    })

# Run parallel learning with mixed configuration approaches
results = acl.teach(
    parallel_learners=3,
    max_iter=15,
    learner_configs=[
        # Learner 0: Adaptive configuration
        ParallelLearnerConfig(simulation=adaptive_sim),
        
        # Learner 1: Per-iteration configuration
        ParallelLearnerConfig(
            simulation={
                0: TaskConfig(kwargs={"--n_labeled": "150", "--n_features": 3}),
                7: TaskConfig(kwargs={"--n_labeled": "250", "--n_features": 3}),
                -1: TaskConfig(kwargs={"--n_labeled": "400", "--n_features": 3})
            }
        ),
        
        # Learner 2: Simple static configuration
        ParallelLearnerConfig(
            simulation=TaskConfig(kwargs={"--n_labeled": "300", "--n_features": 4})
        )
    ]
)

engine.shutdown()
```

## Understanding the Benefits

!!! tip "Configuration Flexibility"
    - **Backward Compatible**: Existing `TaskConfig` usage continues to work unchanged
    - **Mix and Match**: Different learners can use different configuration approaches
    - **Iteration-Specific**: Fine-grained control over parameters at each iteration
    - **Dynamic Adaptation**: Use functions to compute configurations based on iteration number

!!! note "Execution Model"
    Each learner runs independently and concurrently. The `teach()` method returns when all learners complete their workflows (either by reaching `max_iter` or meeting the stop criterion).

!!! warning "Stop Criterion"
    Like the sequential learner, each parallel learner evaluates the stop criterion independently. A learner stops when its own criterion is met, not when other learners stop.

## Important Notes to Remember

!!! note "Execution Model"
    Each learner runs independently and concurrently. The `teach()` method returns when all learners complete their workflows (either by reaching `max_iter` or meeting the stop criterion).

!!! warning "Stop Criterion Behavior"
    Each parallel learner evaluates the stop criterion independently. A learner stops when its own criterion is met, not when other learners stop. This means learners may finish at different times.

!!! tip "Performance Considerations"
    The number of parallel learners should match your available computational resources. More learners aren't always better if they exceed your system's capacity.

## Quick Reference: Helper Methods

### `create_iteration_schedule(task_name, schedule)`
Perfect for creating discrete parameter changes at specific iterations:

```python
schedule = {
    0: {'kwargs': {'--learning_rate': '0.01'}},      # Start with high LR
    5: {'kwargs': {'--learning_rate': '0.005'}},     # Reduce at iteration 5
    10: {'kwargs': {'--learning_rate': '0.001'}},    # Further reduce at iteration 10
    -1: {'kwargs': {'--learning_rate': '0.0001'}}    # Default for remaining iterations
}
config = acl.create_iteration_schedule('training', schedule)
```

### `create_adaptive_schedule(task_name, param_schedule)`
Ideal for smooth parameter transitions based on mathematical functions:

```python
# Exponential decay learning rate
def lr_decay(iteration):
    return {'kwargs': {'--learning_rate': str(0.01 * (0.95 ** iteration))}}

adaptive_config = acl.create_adaptive_schedule('training', lr_decay)
```

## Next Steps

Now that you understand parallel active learning, try these exercises:

1. **Compare Algorithms**: Use different active learning algorithms in parallel learners
2. **Parameter Sweeps**: Test different hyperparameters across learners
3. **Curriculum Learning**: Implement progressive difficulty using per-iteration configs
4. **Resource Scaling**: Experiment with different numbers of parallel learners

Happy learning! 🚀