# Tracking and Observability

ROSE has a pluggable tracking system that lets you record what happened in a run — metrics,
pipeline configuration, stop reason — without changing a single line of your workflow code.

You attach a tracker **once**, after registering all tasks and before calling `start()`.
The learner calls it automatically at every lifecycle point. No tracking code belongs
inside your `async for` loop.

```python
# 1. Register tasks first
@learner.training_task(as_executable=False)
async def train(*args, **kwargs): ...

# 2. Attach tracker — on_start(manifest) fires here with the complete pipeline manifest
learner.add_tracker(MyTracker(...))

# 3. Start the loop
async for state in learner.start(max_iter=20):
    # your control logic only — tracking is fully automatic
    if state.metric_value and state.metric_value < 0.001:
        break
```

`add_tracker()` fires `on_start(manifest)` immediately. All task decorators must have been
applied beforehand so the manifest is complete.

---

## Two data channels

ROSE separates **control** data from **observability** data. They serve different purposes and
flow through different channels.

| Channel | Object | Purpose | Lifetime |
|---------|--------|---------|----------|
| **Control** | `IterationState` | User reads it to decide `break`, `set_next_config()`, etc. | Ephemeral — a fresh object each iteration |
| **Observability** | Tracker callbacks | Record what happened for reproducibility, debugging, comparison | Persistent — written to disk / database / external service |

The **same `IterationState` object is the source for both channels.** The tracker adds context
(run identity, pipeline manifest, stop reason) that is irrelevant to the control loop.

---

## The `TrackerBase` protocol

Any class with these three methods is a valid tracker. All methods have default no-op
implementations — implement only what you need.

```python
from rose import TrackerBase, PipelineManifest, IterationState

class MyTracker(TrackerBase):

    def on_start(self, manifest: PipelineManifest) -> None:
        """Called once inside add_tracker(), immediately when it is invoked.

        All task decorators must have fired before add_tracker() is called —
        the manifest is built at that moment and contains the full pipeline.
        """

    def on_iteration(self, state: IterationState) -> None:
        """Called once per iteration, just before yield state.

        state.state contains the consolidated snapshot of all task outputs
        extracted from dict return values during this iteration.
        """

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        """Called once in the finally block of start().

        Fires on normal completion, criterion met, external stop(), user break,
        and exceptions. reason is one of:
          "criterion_met"    — stop criterion threshold was reached
          "max_iter_reached" — all iterations completed normally
          "stopped"          — learner.stop() was called or user broke the loop
          "error"            — an unhandled exception occurred
        """
```

### Lifecycle diagram

```
# Register tasks first, then:
add_tracker(t)
    └─ t.on_start(manifest)           ← fires immediately; manifest is complete because tasks were registered first

start() called
    ├─ iteration 0
    │   ├─ simulation task → returns {"n_labeled": 25, ...}
    │   ├─ training task   → returns {"train_mse": 0.03, "lml": -10.1, ...}
    │   ├─ active_learn task
    │   ├─ criterion task
    │   ├─ build_iteration_state() → IterationState snapshot (state.state from dict returns)
    │   ├─ t.on_iteration(state)    ← full snapshot with all state keys
    │   └─ yield state              ← user's async for loop body runs
    │
    ├─ iteration 1 ... (same pattern)
    │
    └─ finally
        └─ t.on_stop(final_state, reason)   ← always fires
```

---

## `PipelineManifest` — what `on_start` receives

`PipelineManifest` captures the full pipeline definition at decoration time. It is built
automatically from the task function dicts already populated by the decorators — no user
annotation required.

```python
@dataclass
class TaskManifest:
    func_name: str        # decorated function's __name__
    func_module: str      # decorated function's __module__
    as_executable: bool   # True = executable_task, False = function_task
    decor_kwargs: dict    # HPC backend kwargs (num_gpus, ranks, memory) — opaque to trackers
    log_params: dict      # explicit tracking metadata declared at decoration time

@dataclass
class CriterionManifest(TaskManifest):
    metric_name: str      # e.g. "mean_squared_error_mse"
    threshold: float      # e.g. 0.01
    operator: str         # e.g. "<"

@dataclass
class PipelineManifest:
    learner_type: str                    # class name, e.g. "SequentialActiveLearner"
    tasks: dict[str, TaskManifest]       # keyed: "simulation", "training", etc.
    criterion: CriterionManifest | None
    parallel_count: int | None           # None for sequential
```

Example — reading the manifest in `on_start`:

```python
def on_start(self, manifest: PipelineManifest) -> None:
    print(manifest.learner_type)                    # "SequentialActiveLearner"
    print(manifest.tasks.keys())                    # dict_keys(['simulation', 'training', 'active_learn'])
    print(manifest.criterion.metric_name)           # "mean_squared_error_mse"
    print(manifest.criterion.threshold)             # 0.01
    print(manifest.tasks["training"].as_executable) # False
    print(manifest.tasks["training"].log_params)    # {"num_gpus": 4, "kernel": "rbf"}
    # manifest.tasks["training"].decor_kwargs holds HPC backend args — not logged by trackers
```

### Two-channel decorator design

Task decorators accept two separate keyword groups with different destinations:

```python
@learner.training_task(
    as_executable=False,
    num_gpus=4,                               # → HPC backend only, opaque to trackers
    log_params={"num_gpus": 4, "kernel": "rbf"}  # → trackers only, via TaskManifest.log_params
)
async def training(*args, **kwargs):
    ...
```

- **`decor_kwargs`** (any keyword not recognised by ROSE) flow to the HPC execution backend
  (RADICAL-Pilot, Ray, etc.) as task requirements. They are intentionally not logged —
  they may include resource specs, auth tokens, or backend-specific values.
- **`log_params`** is the explicit opt-in channel: only what you list here reaches trackers.
  Nothing is logged unless you declare it.

---

## How task outputs reach the tracker

When a task returns a `dict`, ROSE automatically extracts each key-value pair into
`IterationState.state`. These values are then available in every `on_iteration` call:

```python
@learner.training_task(as_executable=False)
async def training(*args, **kwargs):
    model = fit_model(...)
    return {
        "train_loss": 0.032,    # → state.state["train_loss"]
        "val_loss": 0.041,      # → state.state["val_loss"]
        "n_params": 125000,     # → state.state["n_params"]
    }
# tracker.on_iteration sees state.state = {"train_loss": 0.032, "val_loss": 0.041, ...}
```

### Reading state from `IterationState`

```python
async for state in learner.start(max_iter=20):
    # Access via attribute-style shortcut
    print(state.train_loss)          # registered as "train_loss" by training task
    print(state.n_labeled)          # registered as "n_labeled" by active_learn task

    # Or via get() with a default
    lml = state.get("log_marginal_likelihood", default=None)

    # Or iterate all registered keys
    for key, value in state.state.items():
        print(f"  {key} = {value}")
```

---

## Parallel learner tracking

For `ParallelActiveLearner`, `ParallelReinforcementLearner`, and `ParallelUQLearner`,
each `IterationState` carries a `learner_id` identifying the sub-learner. Use `on_iteration`
to observe all sub-learners — the `IterationState.state` dict contains the full consolidated
snapshot from all tasks run by that sub-learner:

```python
def on_iteration(self, state: IterationState) -> None:
    # state.learner_id = "learner-0", "learner-1", etc.
    # state.state = full snapshot from that sub-learner's tasks
    print(f"[{state.learner_id}] iter={state.iteration} metric={state.metric_value}")
    for key, value in state.state.items():
        if isinstance(value, (int, float)):
            self.log_metric(f"{state.learner_id}/{key}", value, step=state.iteration)
```

---

## Built-in trackers

### HPC FileTracker — no external dependencies

The simplest production tracker: append-only JSON Lines file. One record per event.
Atomic at the POSIX level — survives job preemption with all completed iterations intact.

```bash
pip install rose scikit-learn numpy
python examples/integrations/tracking/run_me.py
```

```python
import json, time
from pathlib import Path
from rose import TrackerBase, PipelineManifest, IterationState

class HPC_FileTracker(TrackerBase):
    """Append-only JSON Lines tracker — safe for HPC job preemption."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.write_text("")
        self._t0 = 0.0

    def on_start(self, manifest: PipelineManifest) -> None:
        self._t0 = time.time()
        self._write({
            "event": "start",
            "learner_type": manifest.learner_type,
            "criterion": {
                "metric": manifest.criterion.metric_name,
                "threshold": manifest.criterion.threshold,
            } if manifest.criterion else None,
        })

    def on_iteration(self, state: IterationState) -> None:
        self._write({
            "event": "iteration",
            "iteration": state.iteration,
            "elapsed_s": round(time.time() - self._t0, 3),
            "metric": state.metric_value,
            "should_stop": state.should_stop,
            **{k: v for k, v in state.state.items()
               if isinstance(v, (int, float, str, bool))},
        })

    def on_stop(self, final_state, reason: str) -> None:
        self._write({
            "event": "stop",
            "reason": reason,
            "elapsed_s": round(time.time() - self._t0, 3),
            "final_iteration": final_state.iteration if final_state else None,
        })

    def _write(self, record: dict) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

# Usage
learner.add_tracker(HPC_FileTracker("run.jsonl"))

# Post-processing
# import pandas
# df = pandas.read_json("run.jsonl", lines=True)
# df[df.event == "iteration"].plot(x="iteration", y="metric", logy=True)
```

### MLflow Tracker

Logs the pipeline manifest as run parameters, per-iteration metrics as MLflow scalars, and
stop reason as a tag. See [MLflow Integration](../integrations/mlflow.md) for full details.

```bash
pip install rose[mlflow]
python examples/integrations/tracking/mlflow/run_me_tracker.py
```

```python
from rose.integrations.mlflow_tracker import MLflowTracker

learner.add_tracker(
    MLflowTracker(
        experiment_name="surrogate-v1",
        run_name="gp-adaptive-kernel",
    )
)
async for state in learner.start(max_iter=30):
    ...   # no mlflow calls here
```

### ClearML Tracker

Logs hyperparameters, per-learner scalar curves, and stop tags. Parallel learner runs appear
as separate series in the same task — directly comparable in the ClearML UI.
See [ClearML Integration](../integrations/clearml.md) for full details.

```bash
pip install rose[clearml]
python examples/integrations/tracking/clearml/run_me.py
```

```python
from rose.integrations.clearml_tracker import ClearMLTracker

learner.add_tracker(
    ClearMLTracker(
        project_name="ROSE-Materials-UQ",
        task_name="ensemble-run-01",
    )
)
async for state in learner.start(learner_names=["A", "B"], max_iter=15):
    ...   # no clearml calls here
```

---

## Stacking multiple trackers

Trackers are independent observers. If one raises an exception the others are unaffected and
the learner continues.

```python
learner.add_tracker(HPC_FileTracker("run.jsonl"))        # always-on safety net
learner.add_tracker(MLflowTracker(experiment_name="x"))  # experiment comparison
learner.add_tracker(ClearMLTracker(project_name="x", task_name="y"))

async for state in learner.start(max_iter=20):
    ...  # all three trackers fire at every lifecycle point
```

---

## Writing a custom tracker

Any class with those three methods is a valid tracker — no import from ROSE required, no
registration, no base class to inherit. The `TrackerBase` class provides default no-ops so
you only need to define the methods you care about:

```python
from rose import TrackerBase   # optional — provides default no-ops

class SlackTracker(TrackerBase):
    """Post a Slack message when the run finishes."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._iters = 0

    def on_iteration(self, state) -> None:
        self._iters += 1

    def on_stop(self, final_state, reason: str) -> None:
        import urllib.request, json as _json
        metric = f"{final_state.metric_value:.4f}" if final_state else "N/A"
        msg = (f"ROSE run finished — {self._iters} iterations, "
               f"reason: {reason}, final metric: {metric}")
        data = _json.dumps({"text": msg}).encode()
        req = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)

learner.add_tracker(SlackTracker("https://hooks.slack.com/services/..."))
```

The same pattern works for any backend: databases (SQLite, PostgreSQL), cloud storage
(S3, GCS), monitoring tools (Prometheus, Grafana), or custom dashboards.

---

## `add_tracker()` vs manual wiring

| | `add_tracker()` | Manual wiring in `async for` |
|---|---|---|
| Params logged | Automatically from manifest | You write `log_params(...)` |
| Metrics logged | Automatically each iteration | You write `log_metric(...)` inside loop |
| Stop reason | Automatically via `on_stop` | You need try/finally or post-loop logic |
| Tracker code in control loop | None | Yes |
| Works with parallel learners | Yes | Yes (but `state.learner_id` must be handled manually) |

!!! tip
    If you find yourself writing tracking calls inside your `async for` loop, that logic belongs
    in a `TrackerBase` subclass instead. The loop body should contain only decisions:
    `break`, `set_next_config()`, and application-level logic.
