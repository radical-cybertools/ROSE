# MLflow Integration with ROSE

This guide shows how to combine ROSE's workflow orchestration with MLflow's experiment tracking for active learning workflows.

## Why Use Both?

ROSE and MLflow solve different problems and work well together:

```
┌─────────────────────────────────────────────────────────────┐
│                    ROSE (Orchestration)                      │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌───────┐ │
│  │Simulation│ → │ Training │ → │Active Learn │ → │ Check │ │
│  └──────────┘   └──────────┘   └─────────────┘   └───────┘ │
│        ↓              ↓               ↓              ↓      │
└────────┼──────────────┼───────────────┼──────────────┼──────┘
         │              │               │              │
         ▼              ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLflow (Tracking)                          │
│  • Log parameters    • Log metrics     • Store models        │
│  • Track iterations  • Learning curves • Model registry      │
└─────────────────────────────────────────────────────────────┘
```

| Tool | Role | What It Does |
|------|------|--------------|
| **ROSE** | Orchestration | Manages task execution order, dependencies, HPC resources, iteration loops |
| **MLflow** | Tracking | Records parameters, metrics, models, and artifacts for analysis |

**ROSE answers:** *What runs? When? Where? In what order?*

**MLflow answers:** *What happened? How well did it perform? Can I reproduce it?*

## Installation

```bash
# Install MLflow
pip install mlflow

# Optional: for visualization
pip install matplotlib
```

## Quick Start

```bash
# Run the example
python mlflow_rose.py

# View results in MLflow UI
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser.

## Basic Integration Pattern

Here's the minimal pattern to integrate MLflow with a ROSE active learning workflow:

```python
import mlflow
from rose.al import SequentialActiveLearner

async def main():
    # 1. Set up MLflow experiment
    mlflow.set_experiment("my_active_learning_experiment")

    with mlflow.start_run():
        # 2. Log your configuration
        mlflow.log_params({
            "max_iterations": 10,
            "mse_threshold": 0.01,
            "n_initial_samples": 10,
        })

        # 3. Set up ROSE learner (as usual)
        learner = SequentialActiveLearner(asyncflow)
        learner.simulation_task(as_executable=False)(simulation)
        learner.training_task(as_executable=False)(training)
        learner.active_learn_task(as_executable=False)(active_learn)
        learner.as_stop_criterion(...)(check_mse)

        # 4. Run ROSE loop and log metrics at each iteration
        async for state in learner.start(max_iter=10):
            # ROSE yields state, MLflow records it
            mlflow.log_metric("mse", state.metric_value, step=state.iteration)
            mlflow.log_metric("labeled_count", state.labeled_count, step=state.iteration)
            mlflow.log_metric("uncertainty", state.mean_uncertainty, step=state.iteration)

        # 5. Log final model
        mlflow.sklearn.log_model(model, "surrogate_model")
```

## What Gets Tracked

### Parameters (logged once at start)

```python
mlflow.log_params({
    "max_iterations": 15,
    "mse_threshold": 0.02,
    "n_initial_samples": 10,
    "n_pool_samples": 200,
    "n_select_per_iteration": 5,
    "orchestrator": "ROSE",
    "learner_type": "SequentialActiveLearner",
})
```

### Metrics (logged at each iteration)

| Metric | Description |
|--------|-------------|
| `mse` | Mean squared error on validation set |
| `train_mse` | Training set MSE |
| `labeled_count` | Number of labeled samples |
| `unlabeled_count` | Remaining pool size |
| `mean_uncertainty` | Average model uncertainty |
| `max_uncertainty` | Maximum uncertainty in pool |

```python
# Log with step number for iteration tracking
mlflow.log_metric("mse", state.metric_value, step=state.iteration)
```

### Final Metrics (logged at end)

| Metric | Description |
|--------|-------------|
| `final_mse` | Final validation MSE |
| `final_mae` | Final mean absolute error |
| `final_r2` | Final R-squared score |
| `total_iterations` | Number of iterations completed |

### Artifacts

- **Models**: Trained surrogate model saved to MLflow Model Registry
- **Plots**: Learning curves (MSE vs iteration, MSE vs sample size)

```python
# Log model with signature for deployment
mlflow.sklearn.log_model(
    model,
    artifact_path="surrogate_model",
    signature=signature,
    registered_model_name="MyModel",
)

# Log plots
mlflow.log_artifact("learning_curve.png", artifact_path="plots")
```

### Tags

```python
mlflow.set_tags({
    "framework": "ROSE+MLflow",
    "task_type": "active_learning",
    "model_type": "GaussianProcessRegressor",
    "status": "success",  # or "failed"
})
```

## Helper Class: MLflowROSETracker

The example includes a helper class that wraps common MLflow operations:

```python
class MLflowROSETracker:
    def start_experiment(self, config: dict):
        """Initialize MLflow run and log parameters."""

    def log_iteration(self, state: IterationState):
        """Log metrics from ROSE iteration state."""

    def log_model(self, model, X_sample, y_sample):
        """Log model to MLflow registry."""

    def log_final_evaluation(self, model):
        """Compute and log final metrics."""

    def end_experiment(self, success: bool):
        """Finalize the MLflow run."""
```

Usage:

```python
tracker = MLflowROSETracker("my_experiment")
tracker.start_experiment({"max_iterations": 10})

async for state in learner.start(max_iter=10):
    tracker.log_iteration(state)

tracker.log_model(model, X_sample, y_sample)
tracker.end_experiment(success=True)
```

## Viewing Results

After running, start the MLflow UI:

```bash
mlflow ui --port 5000
```

In the UI you can:

- **Compare runs**: See how different configurations perform
- **View metrics**: Interactive charts of MSE, uncertainty over iterations
- **Download models**: Get trained models from the registry
- **Check artifacts**: View learning curve plots

## Example Output

```
============================================================
ROSE + MLflow Integration Example
============================================================

ROSE: Orchestrates the active learning workflow
MLflow: Tracks experiments, metrics, and models
============================================================
MLflow Run ID: a1b2c3d4e5f6
MLflow Experiment: ROSE_Active_Learning

[ROSE] Starting active learning loop...
------------------------------------------------------------

[Iteration 0]
  MSE: 0.156234 (threshold: 0.02)
  Labeled: 15, Pool: 195
  Uncertainty - mean: 0.2341, max: 0.4521

[Iteration 1]
  MSE: 0.089123 (threshold: 0.02)
  Labeled: 20, Pool: 190
  Uncertainty - mean: 0.1892, max: 0.3891

...

------------------------------------------------------------
[ROSE] Active learning completed

[MLflow] Logging final model...
[MLflow] Computing final evaluation metrics...

============================================================
Final Results
============================================================
  Total iterations: 8
  Final MSE: 0.018234
  Final MAE: 0.102341
  Final R2:  0.9812
  Final labeled samples: 50

MLflow run completed: a1b2c3d4e5f6

============================================================
To view results, run:
  mlflow ui --port 5000
Then open http://localhost:5000
============================================================
```

## Files

| File | Description |
|------|-------------|
| `mlflow_rose.py` | Complete integration example with all features |
| `README.md` | This documentation |

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ROSE Documentation](https://radical-cybertools.github.io/ROSE/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
