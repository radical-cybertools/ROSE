To accelerate the development of UQ-driven active learning methods, ROSE provides a flexible approach for composing and executing complex AL workflows.

In this example, we illustrate how to define a UQ–AL workflow that supports multiple levels of parallelism.

In many cases, an AL cycle requires evaluating **multiple candidate models or simulations concurrently** to quantify predictive uncertainty and guide sample selection. Beyond that, it may also require running **several independent AL workflows in parallel** to exploring multiple uncertainty metrics.

This introduces **two levels of parallelism**:

* **Task-level parallelism** for concurrent model training, inference, or simulation.
* **Workflow-level parallelism** for executing multiple UQ–AL loops side by side.

Both levels can be naturally expressed and efficiently executed using ROSE’s **custom AL policy**, enabling scalable and adaptive uncertainty-aware learning.
```sh            
                             (N AL WFs in Parallel)
          +-------------------+               +-------------------+
          |      UQ WF 1      |               |      UQ WF 2      |   
          +-------------------+               +-------------------+
                   │                                    │
  +----------------+-----------------+  +----------------+-----------------+
  |       (N tasks Parallel)         |  |       (N AL tasks Parallel)      |
  +---------------+  +---------------+  +---------------+  +---------------+
  | Simulation 1,2,..............n   |  | Simulation 1,2,..............n   |
  +---------------+  +---------------+  +---------------+  +---------------+
          |                |                    |                 |
  +---------------+  +---------------+  +---------------+  +---------------+
  |  Train Model  1,2,...........m   |  |  Train Model  1,2,...........m   |
  +---------------+  +---------------+  +---------------+  +---------------+
          |                |                    |                 |
    +-----------------------------+        +-----------------------------+ 
    |      AL based on UQ WF 1    |        |    AL based on UQ WF 1      |   
    +-----------------------------+        +-----------------------------+ 
```
 

### UQ-Driven Active Learning with Parallel Workflows

The **UQ Active Learning (UQ-AL) workflow** extends the traditional active learning loop by running multiple models in parallel. Instead of relying on a single model to guide the selection of new samples, the workflow trains and evaluates an **ensemble of models simultaneously**. Predictions from all models are then aggregated to compute **uncertainty metrics**, which are used to identify the most uncertain examples to the next AL iteration.

#### Key Component: `ParallelUQLearner`

* Trains multiple models in parallel.
* Collects predictions from all models on candidate data points.
* Computes UQ metrics (e.g., entropy, variance, disagreement) across the ensemble.
* Selects the most uncertain samples for labeling or simulation.

#### New Task Types

To support this approach, two new task types have been introduced:

1. **`prediction_task`**

   * Runs inference for each model in the ensemble.
   * Produces prediction outputs that will be used in the UQ calculation.

```python
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the prediction task
@learner.prediction_task()
async def prediction(*args):
    return f'{code_path}/predict.py'

```

2. **`uncertainty_quantification`**

   * Aggregates predictions from all models.
   * Computes uncertainty metrics.
   * Returns the top-k uncertain samples for the next active learning cycle.

```python

# Defining the uncertainty quantification with a metric (PREDICTIVE_ENTROPY in this case)
@learner.uncertainty_quantification(uq_metric_name=PREDICTIVE_ENTROPY, 
                                    threshold=1.0, 
                                    query_size=10)
async def check_uq(*args):
    return f'{code_path}/check_uq.py'xs

#### Workflow Summary

1. Launch multiple **prediction tasks** in parallel.
2. Collect outputs and pass them to the **uncertainty quantification task**.
3. Identify the most uncertain samples.
4. Add these samples to the training set for the next AL iteration.

This design allows **parallel training and uncertainty-aware sampling** within AL workflows, making it easy to scale across many models or candidate datasets.


#### Getting Started with ParallelUQLearner in ROSE

Import and Initialize the UQ Learner

```python
from rose.uq.uq_active_learner import ParallelUQLearner
learner = ParallelUQLearner(asyncflow)
```

Run the Teaching (Active Learning Loop)

```python
PIPELINES = ['UQ_learner1', 'UQ_learner2']

results = await learner.teach(
    learner_names=PIPELINES,
    model_names=MODELS,
    learner_configs=learner_configs,
    max_iter=ITERATIONS, 
    num_predictions=NUM_PREDICTION
)
```

Save Results to File and Shutdown the Learner

```python

print('Teaching is done with Final Results:')
print(results)

with open(Path(os.getcwd(), 'UQ_training_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

await learner.shutdown()
```

#### Defining costom UQ metric

```
from rose.uq import UQScorer, register_uq, UQ_REGISTRY

@register_uq("custom_uq")
def confidence_score(self, mc_preds):
    """
    Custom classification metric: 1 - max predicted probability.
    Lower max prob = higher uncertainty.
    """
    mc_preds, _ = self._validate_inputs(mc_preds)
    mean_probs = np.mean(mc_preds, axis=0)      # [n_instances, n_classes]
    max_prob = np.max(mean_probs, axis=1)
    return 1.0 - max_prob


scorer = UQScorer(task_type="classification")
print("Available metrics:", list(UQ_REGISTRY.keys()))

UQ_METRIC_NAME='custom_uq'
```