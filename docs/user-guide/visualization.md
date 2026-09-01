# Visualization

ROSE does not ship a plotting module — visualizing a run's progress is left to whichever
tool already fits your workflow: raw `IterationState` fields, the native file tracker, or
an MLflow/ClearML dashboard if `tracking.backend` is already wired up.

!!! note
    There is no `rose.plot` or similar utility. The three approaches below all read data
    ROSE already produces — none requires changes to your task code.

---

## Real-time, in-loop

Every iteration yields an `IterationState` with `iteration`, `metric_value`, and
`metric_history` (the full list of past metric values so far):

```python
import matplotlib.pyplot as plt

async for state in learner.start(max_iter=20):
    print(f"[iter {state.iteration}] mse={state.metric_value:.4f}")

plt.plot(state.metric_history)
plt.xlabel("iteration")
plt.ylabel("mse")
plt.show()
```

This is the fastest path when you just want a quick look right after a local run.

---

## Native file tracker

For HPC runs where you want a durable, preemption-safe record, attach `HPC_FileTracker`
(see `examples/integrations/tracking/basic.py`) — it appends one JSON line per iteration:

```python
learner.add_tracker(HPC_FileTracker("run.jsonl"))
```

Replay and plot after the run, or from a different machine entirely:

```python
import pandas as pd

df = pd.read_json("run.jsonl", lines=True)
iterations = df[df.event == "iteration"]
iterations.plot(x="iteration", y="mse", logy=True)
```

Any numeric value your tasks return as a `dict` (e.g. `n_labeled`, `train_mse`) is captured
automatically and available as its own column.

---

## MLflow / ClearML dashboards

If `tracking.backend: mlflow` or `tracking.backend: clearml` is set in your spec (or you
attach `MLflowTracker`/`ClearMLTracker` directly in the Python API), every iteration's
metrics are already logged with no extra code. The respective web UIs are the recommended
way to compare runs and overlay parallel-learner series:

- [MLflow integration](../integrations/mlflow.md) — `mlflow ui --port 5000`, metric curves
  per run, run comparison.
- [ClearML integration](../integrations/clearml.md) — **Scalars** tab, parallel learners
  shown as separate series under the same title.

!!! tip
    Prefer the dashboards over building your own plots once you have more than a couple of
    runs to compare — both tools already handle multi-run overlay and filtering.
