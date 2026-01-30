# MLflow Integration

This guide demonstrates how to combine **ROSE's** workflow orchestration with **MLflow's** experiment tracking to create a robust and observable active learning system.

## Overview

ROSE and MLflow provide a complementary relationship in a research or production pipeline:

*   **ROSE (Orchestration):** Manages task execution order, dependencies, high-performance computing (HPC) resources, and the iterative loop.
*   **MLflow (Tracking):** Records hyperparameters, performance metrics, trained models, and diagnostic plots for analysis and reproducibility.

| Tool | Role | Focus |
|------|------|-------|
| **ROSE** | Orchestrator | *What* runs? *When*? *Where*? In what order? |
| **MLflow** | Tracker | *What happened*? How well did it perform? Can I reproduce it? |

---

## Installation

To use this integration, you need both `mlflow` and `ROSE` installed in your environment.

```bash
# Install MLflow
pip install mlflow

# Optional: for visualization logic in the example
pip install matplotlib scikit-learn
```

---

## Quick Start

You can find a complete integration example in the codebase at `examples/integrations/mlflow/mlflow_rose.py`.

```bash
# Run the integration example
python examples/integrations/mlflow/mlflow_rose.py

# Launch the MLflow UI to view results
mlflow ui --port 5000
```

Once the UI is running, open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Integration Pattern

The standard pattern for integrating MLflow into a ROSE `SequentialActiveLearner` loop involves wrapping the learner's `start()` iterator:

```python
import mlflow
from rose.al import SequentialActiveLearner

async def main():
    # 1. Initialize MLflow Run
    mlflow.set_experiment("ROSE_AL_Experiment")
    
    with mlflow.start_run():
        # 2. Log Configuration
        mlflow.log_params({
            "max_iterations": 10,
            "mse_threshold": 0.01,
        })

        # 3. Setup ROSE Learner
        learner = SequentialActiveLearner(asyncflow)
        # ... register tasks ...

        # 4. Instrument the Control Loop
        async for state in learner.start(max_iter=10):
            # Log metrics at each iteration step
            mlflow.log_metric("mse", state.metric_value, step=state.iteration)
            mlflow.log_metric("labeled_count", state.labeled_count, step=state.iteration)
            
            print(f"Iteration {state.iteration}: MSE {state.metric_value}")

        # 5. Log Final Artifacts
        mlflow.sklearn.log_model(final_model, "surrogate_model")
```

---

## What is Tracked?

### Parameters
Parameters are typically logged once at the beginning of the run to record the experimental setup.
*   Iteration limits
*   Stopping criteria thresholds
*   Initial sample sizes
*   Batch selection counts

### Metrics
Metrics are logged at **each iteration step** using the `step` parameter in `mlflow.log_metric()`. This allows you to view learning curves and performance trends over time in the MLflow UI.
*   **Performance:** MSE, Accuracy, R-squared
*   **Workflow State:** Number of labeled samples, remaining pool size
*   **Adaptive Features:** Current uncertainty scores, selection batch sizes

### Artifacts and Model Registry
At the end of the ROSE workflow, you can save:
*   **The Model:** Register the final surrogate model in the MLflow Model Registry for deployment.
*   **Visualizations:** Save plots of error reduction vs. iteration or sample size.
*   **Data States:** Save the final labeled dataset for future reference.

---

## Advanced: MLflowROSETracker Helper

For more complex workflows, the provided example includes an `MLflowROSETracker` helper class. It encapsulates common tracking logic, making the main workflow code cleaner:

```python
tracker = MLflowROSETracker("My_Complex_Experiment")
tracker.start_experiment(config)

async for state in learner.start(max_iter=15):
    # Automatically handles extraction and logging of relevant metrics
    tracker.log_iteration(state)

tracker.log_model(model, X_sample, y_sample)
tracker.end_experiment(success=True)
```

For the full implementation of this helper, see the [mlflow_rose.py source code](https://github.com/radical-cybertools/ROSE/blob/main/examples/integrations/mlflow/mlflow_rose.py).
