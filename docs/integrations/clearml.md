# ClearML Integration

ROSE ships a plug-and-play `ClearMLTracker` that wires ClearML into any learner with a
single line. For parallel learners, each sub-learner's metrics appear as separate series
inside the same task — directly overlaid in the ClearML UI for convergence comparison.

```bash
pip install rose[clearml]
```

---

## Quick start

```python
from rose.integrations.clearml_tracker import ClearMLTracker

learner.add_tracker(
    ClearMLTracker(
        project_name="ROSE-Materials-UQ",
        task_name="ensemble-run-01",
    )
)

async for state in learner.start(learner_names=["A", "B"], max_iter=15):
    print(f"[{state.learner_id}] iter {state.iteration}: mse={state.metric_value:.4f}")
    # tracking is fully automatic — no clearml calls here
```

Open the ClearML web UI, navigate to project `ROSE-Materials-UQ`, and select the task
`ensemble-run-01`. The **Scalars** tab shows overlaid curves per learner.

A complete runnable example is at
`examples/integrations/tracking/clearml/run_me.py`.

---

## What gets logged automatically

### Hyperparameters — logged once in `on_start`

The entire pipeline manifest is connected to the ClearML task without any user annotation:

| ClearML hyperparameter | Source |
|---|---|
| `learner_type` | Learner class name |
| `criterion/metric_name` | `as_stop_criterion(metric_name=...)` |
| `criterion/threshold` | `as_stop_criterion(threshold=...)` |
| `criterion/operator` | `as_stop_criterion(operator=...)` |
| `task/<name>/as_executable` | Per registered task |
| `task/<name>/<key>` | Explicit `log_params` dict declared in task decorator |

### Scalars — logged per iteration in `on_iteration`

| ClearML scalar | Source |
|---|---|
| `<metric_name>` (e.g. `mean_squared_error_mse`) | Stop criterion value |
| Any numeric key in `state.state` | Auto-extracted from task `dict` returns |

For **parallel learners**, `state.learner_id` is included automatically. The tracker logs
each state as a separate `series` inside the same scalar title, making per-learner curves
directly comparable without any user code.

### Task tags — logged in `on_stop`

| ClearML tag | Value |
|---|---|
| `stop:<reason>` | `stop:criterion_met` / `stop:max_iter_reached` / `stop:stopped` / `stop:error` |
| `final_iter:<n>` | Last completed iteration number |

Tags make it easy to filter tasks in the ClearML UI by outcome.

---

## Parallel learner comparison

The ClearML tracker is designed with parallel learners in mind. Each `on_iteration` call
carries `state.learner_id` — the tracker logs each learner as a separate scalar series
under the same title:

```
Scalars tab in ClearML UI:
  ┌─ mean_squared_error_mse ──────────────────────────────┐
  │  ensemble-A  ───────────\                             │
  │  ensemble-B  ────────────\──────────────────────────  │
  └───────────────────────────────────────────────────────┘
```

No user code is required to achieve this — `state.learner_id` is already set by the
parallel learner framework.

---

## Multiple trackers

Attach ClearML alongside other trackers — they are independent observers:

```python
from rose.integrations.clearml_tracker import ClearMLTracker

learner.add_tracker(HPC_FileTracker("run.jsonl"))   # safety net on HPC
learner.add_tracker(ClearMLTracker(project_name="x", task_name="y"))
```

---

## Extending `ClearMLTracker`

To log additional artifacts (model checkpoints, prediction plots) override `on_stop`:

```python
from rose.integrations.clearml_tracker import ClearMLTracker

class ClearMLCheckpointTracker(ClearMLTracker):
    def on_stop(self, final_state, reason: str) -> None:
        # Log model checkpoint as a ClearML artifact before closing the task
        if final_state and self._task:
            checkpoint = final_state.get("model_checkpoint")
            if checkpoint:
                self._task.upload_artifact(
                    name="best_model",
                    artifact_object=checkpoint,
                )
        super().on_stop(final_state, reason)
```
