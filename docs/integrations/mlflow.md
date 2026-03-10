# MLflow Integration

ROSE ships a plug-and-play `MLflowTracker` that wires MLflow into any learner with a single
line. No MLflow calls belong inside your `async for` loop.

```bash
pip install rose[mlflow]
```

---

## Quick start

```python
from rose.integrations.mlflow_tracker import MLflowTracker

learner.add_tracker(
    MLflowTracker(
        experiment_name="surrogate-v1",
        run_name="gp-adaptive-kernel",   # optional
    )
)

async for state in learner.start(max_iter=30):
    print(f"iter {state.iteration}: mse={state.metric_value:.4f}")
    # tracking is fully automatic — no mlflow calls here
```

View results:

```bash
mlflow ui --port 5000
# Open http://localhost:5000 → experiment "surrogate-v1"
```

A complete runnable example is at
`examples/integrations/tracking/mlflow/run_me_tracker.py`.

---

## What gets logged automatically

### Parameters — logged once in `on_start`

The entire pipeline manifest is logged as MLflow parameters without any user annotation:

| MLflow param | Source |
|---|---|
| `learner_type` | Learner class name |
| `criterion/metric_name` | `as_stop_criterion(metric_name=...)` |
| `criterion/threshold` | `as_stop_criterion(threshold=...)` |
| `criterion/operator` | `as_stop_criterion(operator=...)` |
| `task.<name>.as_executable` | Per registered task |
| `task.<name>.<kwarg>` | Extra decorator kwargs (e.g. `num_gpus`) |

### Metrics — logged per iteration in `on_iteration`

| MLflow metric | Source |
|---|---|
| `<metric_name>` (e.g. `mean_squared_error_mse`) | Stop criterion value |
| Any scalar in `state.state` | Auto-extracted from task `dict` returns |

Every key returned in a task's `dict` result appears as a metric — zero annotation required.

### Tags — logged in `on_stop`

| MLflow tag | Value |
|---|---|
| `stop_reason` | `"criterion_met"` / `"max_iter_reached"` / `"stopped"` / `"error"` |
| `final_iteration` | Last completed iteration number |

---

## Adaptive config changes

When you call `learner.set_next_config(config)` to change hyperparameters between iterations,
the new config appears in the next `IterationState.current_config`. MLflow captures this
automatically in `on_iteration` — no manual `log_params()` call needed.

```python
configs = {
    0:  LearnerConfig(training=TaskConfig(kwargs={"--lr": 3e-4})),
    10: LearnerConfig(training=TaskConfig(kwargs={"--lr": 1e-4})),
    20: LearnerConfig(training=TaskConfig(kwargs={"--lr": 3e-5})),
}

async for state in learner.start(max_iter=30):
    next_iter = state.iteration + 1
    if next_iter in configs:
        learner.set_next_config(configs[next_iter])
    # MLflow records the config change — no manual call needed
```

---

## Multiple trackers

Attach MLflow alongside other trackers — they are independent observers:

```python
from rose.integrations.mlflow_tracker import MLflowTracker

learner.add_tracker(HPC_FileTracker("run.jsonl"))       # safety net
learner.add_tracker(MLflowTracker(experiment_name="x")) # experiment comparison
```

---

## `MLflowTracker` vs manual wiring

The previous ROSE documentation showed a manual pattern where MLflow calls were placed
inside the `async for` loop. That approach is now deprecated in favour of `add_tracker()`.

| | `MLflowTracker` | Manual wiring |
|---|---|---|
| Pipeline manifest as params | Automatic | Must write `log_params(...)` manually |
| Metrics per iteration | Automatic | Must call `log_metric(...)` inside loop |
| Stop reason tag | Automatic | Requires try/finally |
| MLflow code in control loop | None | Yes |

!!! tip
    If you need to log model artifacts (e.g. `mlflow.sklearn.log_model`) or custom plots,
    add that logic to a subclass of `MLflowTracker` by overriding `on_stop`:

    ```python
    class MLflowArtifactTracker(MLflowTracker):
        def on_stop(self, final_state, reason: str) -> None:
            super().on_stop(final_state, reason)
            if final_state and reason in ("criterion_met", "max_iter_reached"):
                model = load_model(final_state.get("checkpoint_path"))
                mlflow.sklearn.log_model(model, artifact_path="surrogate_model")
    ```
