# ROSE Tracking Integrations

ROSE has a pluggable tracking system built on the `TrackerBase` protocol. You attach a tracker
once with `learner.add_tracker(...)` — before calling `start()` — and the learner calls it
automatically at every lifecycle point. No tracking code inside your `async for` loop.

```
learner.add_tracker(MyTracker(...))   ← one line, before start()

async for state in learner.start(...):
    # your control logic only — tracking is automatic
```

Three trackers are available out of the box:

| Tracker | File | Dependency |
|---------|------|------------|
| `HPC_FileTracker` | `run_me.py` | none (stdlib only) |
| `MLflowTracker` | `mlflow/run_me_tracker.py` | `pip install rose[mlflow]` |
| `ClearMLTracker` | `clearml/run_me.py` | `pip install rose[clearml]` |

---

## Tracker lifecycle

Every tracker receives four calls from the learner:

| Method | When | What it receives |
|--------|------|-----------------|
| `on_start(manifest)` | Once, at `add_tracker()` | Full pipeline manifest: task names, criterion threshold/operator, learner type |
| `on_iteration(state)` | Once per iteration, before `yield` | Complete `IterationState` snapshot: metric, registered task outputs, `learner_id` |
| `on_stop(final_state, reason)` | Once, in `finally` | Last state + stop reason: `"criterion_met"` / `"max_iter_reached"` / `"stopped"` / `"error"` |
| `on_state_update(key, value)` | Real-time, inside tasks | Individual `register_state()` calls — fires mid-iteration before snapshot is built |

`on_state_update` is **sequential-learner only**. For parallel learners, use `state.state` dict
in `on_iteration` — it already contains the consolidated snapshot of all task outputs.

---

## 1 — HPC FileTracker (no external dependencies)

**Science:** Branin-Hoo 2D GP surrogate — uncertainty sampling active learning on a canonical
benchmark with known global optima. Runs entirely with stdlib + numpy + scikit-learn.

**Tracker:** Append-only JSON Lines file. Each call to `on_iteration` appends one line —
atomic at the POSIX level. If the HPC job is preempted, all completed iterations are already
on disk and can be inspected or resumed without rerunning the whole pipeline.

`on_state_update` captures the GP log-marginal-likelihood (LML) as a live stream record
each time it is registered mid-iteration inside the training task, before the full iteration
snapshot is built.

```bash
python run_me.py

# Inspect the output:
python -c "
import pandas
df = pandas.read_json('branin_run.jsonl', lines=True)
print(df[df.event == 'iteration'][['iteration', 'mse', 'n_labeled', 'log_marginal_likelihood']])
"
```

**What gets logged automatically:**

```
{"event": "start",  "learner_type": "SequentialActiveLearner", "criterion": {"metric": "mean_squared_error_mse", "threshold": 0.01}, ...}
{"event": "live",   "key": "log_marginal_likelihood", "value": -10.3, "ts": 1234567890.1}   ← mid-iteration stream
{"event": "iteration", "iteration": 0, "mse": 0.014, "n_labeled": 25, "n_pool": 290, ...}
{"event": "iteration", "iteration": 1, "mse": 0.0012, "n_labeled": 35, "n_pool": 280, ...}
{"event": "stop",   "reason": "criterion_met", "final_iteration": 1, "final_mse": 0.0012}
```

**Implement your own `HPC_FileTracker`:**

```python
import json, time
from pathlib import Path
from rose import TrackerBase, PipelineManifest, IterationState

class HPC_FileTracker(TrackerBase):
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.write_text("")   # truncate on new run
        self._t0 = 0.0

    def on_start(self, manifest: PipelineManifest) -> None:
        self._t0 = time.time()
        self._write({"event": "start", "learner_type": manifest.learner_type})

    def on_iteration(self, state: IterationState) -> None:
        self._write({
            "event": "iteration",
            "iteration": state.iteration,
            "metric": state.metric_value,
            **state.state,   # all task outputs registered via register_state()
        })

    def on_stop(self, final_state, reason: str) -> None:
        self._write({"event": "stop", "reason": reason,
                     "elapsed_s": round(time.time() - self._t0, 3)})

    def on_state_update(self, key: str, value) -> None:
        self._write({"event": "live", "key": key, "value": value})

    def _write(self, record: dict) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

learner.add_tracker(HPC_FileTracker("run.jsonl"))
```

---

## 2 — MLflow Tracker

**Science:** Rosenbrock 2D GP surrogate with an *adaptive kernel schedule* — the GP kernel
length-scale is tightened in three stages as the surrogate matures, mimicking curriculum-style
surrogate refinement. `set_next_config()` injects the new kernel parameters at iteration
boundaries and MLflow captures the config change automatically in the next `on_iteration()` call.

**Tracker:** `MLflowTracker` from `rose.integrations.mlflow_tracker`. Logs the pipeline
manifest as run parameters on start, per-iteration metrics as MLflow scalars, and stop reason
as a tag on stop. `on_state_update` streams live metrics as `live.<key>` before the iteration
snapshot is built.

```bash
pip install rose[mlflow] scikit-learn numpy

python mlflow/run_me_tracker.py

# View results:
mlflow ui --port 5000
# Open http://localhost:5000 → experiment "ROSE-Rosenbrock-Surrogate"
```

**What gets logged automatically:**

```
Params (on_start):
  learner_type = "SequentialActiveLearner"
  criterion/metric_name = "mean_squared_error_mse"
  criterion/threshold = 0.002
  task.simulation.as_executable = False
  task.training.as_executable = False
  ...

Metrics (on_iteration, per step):
  MEAN_SQUARED_ERROR_MSE    ← stop criterion value
  n_labeled                 ← from register_state() in active_learn
  n_pool
  train_mse                 ← from register_state() in training
  log_marginal_likelihood   ← from register_state() in training
  length_scale_used

Live metrics (on_state_update):
  live.log_marginal_likelihood  ← streams mid-iteration

Tags (on_stop):
  stop_reason = "criterion_met"
  final_iteration = "4"
```

**Wire it:**

```python
from rose.integrations.mlflow_tracker import MLflowTracker

learner.add_tracker(
    MLflowTracker(
        experiment_name="my-surrogate",
        run_name="gp-v1",
    )
)
async for state in learner.start(max_iter=30):
    ...   # no mlflow calls here — all automatic
```

---

## 3 — ClearML Tracker

**Science:** 5D materials formation energy prediction using a two-learner ensemble
(`"ensemble-A"`, `"ensemble-B"`, different random seeds). Predictive entropy quantifies
inter-model disagreement — the ensemble stops when mean entropy falls below 0.02 nats,
meaning both models agree sufficiently on unseen structures.

**Tracker:** `ClearMLTracker` from `rose.integrations.clearml_tracker`. Each yielded
`IterationState` carries `state.learner_id` (`"ensemble-A"` or `"ensemble-B"`), so
both learners' scalar curves appear in the same ClearML task and can be overlaid for
direct convergence comparison.

```bash
pip install rose[clearml] scikit-learn numpy

python clearml/run_me.py

# Open ClearML web UI → project "ROSE-Materials-UQ"
# Scalars tab: overlay mse curves for ensemble-A vs ensemble-B
```

**What gets logged automatically:**

```
Hyperparameters (on_start):
  learner_type = "ParallelUQLearner"
  criterion/metric_name = "mean_squared_error_mse"
  criterion/threshold = 0.005
  ...

Scalars (on_iteration, per learner per step):
  title=MEAN_SQUARED_ERROR_MSE  series=value  ← stop criterion
  title=n_labeled               series=value
  title=n_pool                  series=value
  title=train_mse               series=value

Task tags (on_stop):
  stop:criterion_met   or   stop:max_iter_reached
  final_iter:7
```

**Wire it:**

```python
from rose.integrations.clearml_tracker import ClearMLTracker

learner.add_tracker(
    ClearMLTracker(
        project_name="my-project",
        task_name="ensemble-run-01",
    )
)
async for state in learner.start(learner_names=["A", "B"], max_iter=15):
    print(f"[{state.learner_id}] iter={state.iteration} mse={state.metric_value:.4f}")
```

---

## Using multiple trackers simultaneously

Trackers are independent observers — attach as many as you want:

```python
from rose.integrations.mlflow_tracker import MLflowTracker
from rose.integrations.clearml_tracker import ClearMLTracker

learner.add_tracker(HPC_FileTracker("run.jsonl"))       # always-on safety net
learner.add_tracker(MLflowTracker(experiment_name="x")) # experiment comparison
learner.add_tracker(ClearMLTracker(project_name="x", task_name="y"))  # team dashboard

async for state in learner.start(max_iter=20):
    ...  # all three trackers fire automatically at each iteration
```

If one tracker raises an exception, the others are unaffected and the learner continues.

---

## Writing a custom tracker

Any class with the four methods is a valid tracker — no import from ROSE required:

```python
class SlackTracker:
    """Post a Slack message when the run finishes."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._iters = 0

    def on_start(self, manifest): pass
    def on_iteration(self, state): self._iters += 1
    def on_state_update(self, key, value): pass

    def on_stop(self, final_state, reason: str) -> None:
        import urllib.request, json
        msg = (f"ROSE run finished after {self._iters} iterations. "
               f"Reason: {reason}. "
               f"Final metric: {final_state.metric_value if final_state else 'N/A'}")
        data = json.dumps({"text": msg}).encode()
        urllib.request.urlopen(urllib.request.Request(
            self._url, data=data, headers={"Content-Type": "application/json"}
        ))

learner.add_tracker(SlackTracker("https://hooks.slack.com/..."))
```
