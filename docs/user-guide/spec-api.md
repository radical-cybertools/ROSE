# YAML Spec API

The YAML spec API lets you declare a ROSE workflow as a data file instead of Python code. The spec is schema-validated when loaded — missing slots, unknown keys, and type errors are caught before any infrastructure starts. Task functions are plain Python functions with no decorators required. The same spec file works with any ROSE learner type: Active Learning, Reinforcement Learning, or UQ.

---


!!! note
    The YAML spec currently supports only the four built-in `learner.type` values listed below:

    - `SequentialActiveLearner`
    - `ParallelActiveLearner`
    - `SequentialReinforcementLearner`
    - `ParallelReinforcementLearner`
    - `SequentialUQLearner`
    - `ParallelUQLearner`

    Custom `Learner` subclasses are not yet expressible in YAML — `LearnerBuilder` raises `ValueError` for any other `learner.type` value. Use the Python API (decorator-based, e.g. `SequentialActiveLearner(asyncflow)`) if you need a custom
    learner implementation.


## Loading a Spec

For the common case — build a learner and run it — `LearnerBuilder` loads and validates the YAML itself; you never need to touch a config object:

```python
from rose.spec.builder import LearnerBuilder

builder = LearnerBuilder("workflow.yaml", asyncflow)  # validates schema on load
cfg     = builder.config                              # typed WorkflowConfig, if you need it
```

Reach for `load_spec` instead when you need `WorkflowSpec`-level features: spec variants via [`workflow_with()`](#spec-variants-with-workflow_with), import validation, or the `.workflow` coroutine used by [ROSE as a Service](rose-aas.md):

```python
from rose.spec import load_spec

spec = load_spec("workflow.yaml")  # validates schema on load
cfg  = spec.config                 # typed WorkflowConfig object
```

Both raise `ValueError` with a precise message on any schema violation. Neither imports task modules by default — see [Import Validation](#import-validation) to enable that check.

---

## Resource Specification via `task_description`

Every task — `simulation`, `training`, `active_learn`, `environment`, `update`, … — can
carry an optional `task_description`. In the Python API this is a parameter on the
decorated function itself, with a default value:

```python
@acl.simulation_task(as_executable=False)
async def simulate(*args, task_description={"process_templates": [(4, {})]}, **kwargs):
    ...
```

The accepted keys depend entirely on which execution backend your `asyncflow` session is
using — see asyncflow's
[Execution Backends guide](https://radical-cybertools.github.io/radical.asyncflow/exec_backends/?h=task_desc#assign-resources-for-your-application-task)
for the full per-backend reference.

The YAML spec exposes the same dict as a `task_description:` key on the corresponding task
slot — see the side-by-side example in [Task Types](#task-types) below.

---

## Sequential Active Learner

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**Python API**

```python
from rose.al import SequentialActiveLearner

acl = SequentialActiveLearner(asyncflow)

@acl.simulation_task(as_executable=False)
async def simulate(*args, **kwargs):
    ...  # return simulation result

@acl.training_task(as_executable=False)
async def train(sim_result, **kwargs):
    ...  # return trained model

@acl.active_learn_task(as_executable=False)
async def active_learn(sim_result, model, **kwargs):
    ...  # return updated dataset

@acl.as_stop_criterion(
    metric_name="mse",
    threshold=0.01,
    operator="<",
)
async def check_mse(*args, **kwargs):
    ...  # return float metric

async for state in acl.start(max_iter=5):
    print(f"iter {state.iteration}  "
          f"mse={state.metric_value:.4f}")

await asyncflow.shutdown()
```

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**YAML Spec**

```yaml
learner:
  type: sequential_active_learner
  max_iter: 5

simulation:
  type: python
  function: tasks:simulate

training:
  type: python
  function: tasks:train

active_learn:
  type: python
  function: tasks:active_learn

stop_criterion:
  metric: mse
  threshold: 0.01
  operator: "<"
  evaluator:
    type: python
    function: tasks:check_mse
```

```python
# run it
from rose.spec.builder import LearnerBuilder

builder = LearnerBuilder("workflow.yaml", asyncflow)
cfg     = builder.config
learner = builder.build()

async for state in learner.start(
    max_iter=cfg.learner.max_iter
):
    print(f"iter {state.iteration}  "
          f"mse={state.metric_value:.4f}")
```

</div>
</div>

Task functions referenced by `function: tasks:simulate` are ordinary Python callables in a `tasks.py` module — no ROSE decorators required:

```python
# tasks.py
def simulate(*args, **kwargs):
    ...  # return simulation result

def train(sim_result, **kwargs):
    ...  # return trained model

def active_learn(sim_result, model, **kwargs):
    ...  # return updated dataset

def check_mse(*args, **kwargs):
    ...  # return float metric value
```

---

## Parallel Learner — Shared Tasks

When all parallel learners run the same task implementations, declare the task slots at the top level exactly as in the sequential case.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**Python API**

```python
from rose.al import ParallelActiveLearner

acl = ParallelActiveLearner(asyncflow)

@acl.simulation_task(as_executable=False)
async def simulate(*args, **kwargs):
    ...

@acl.training_task(as_executable=False)
async def train(sim_result, **kwargs):
    ...

@acl.active_learn_task(as_executable=False)
async def active_learn(sim_result, model, **kwargs):
    ...

@acl.as_stop_criterion(
    metric_name="mse",
    threshold=0.01,
)
async def check_mse(*args, **kwargs):
    ...

async for state in acl.start(
    parallel_learners=2,
    max_iter=5,
):
    print(f"learner {state.learner_id}  "
          f"iter {state.iteration}")
```

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**YAML Spec**

```yaml
learner:
  type: parallel_active_learner
  max_iter: 5
  parallel_learners: 2

simulation:
  type: python
  function: tasks:simulate

training:
  type: python
  function: tasks:train

active_learn:
  type: python
  function: tasks:active_learn

stop_criterion:
  metric: mse
  threshold: 0.01
  operator: "<"
  evaluator:
    type: python
    function: tasks:check_mse
```

```python
# run it
builder = LearnerBuilder("workflow.yaml", asyncflow)
cfg     = builder.config
learner = builder.build()

async for state in learner.start(
    max_iter=cfg.learner.max_iter,
    parallel_learners=cfg.learner.parallel_learners,
):
    print(f"learner {state.learner_id}  "
          f"iter {state.iteration}")
```

</div>
</div>

Each learner receives a unique `learner_id` kwarg (0, 1, …) in its task calls. Task functions can use it to write to isolated files or namespaces.

---

## Parallel Learner — Distinct Tasks (`learners:` block)

When each parallel learner needs its own task implementations — different models, different executables, different command-line flags — use the `learners:` block. Each entry gets a `label` that is injected as `learner_label` into every task kwarg.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**Python API**

```python
acl = ParallelActiveLearner(asyncflow)

# Manual routing by learner_id
TRAIN_CMD = {
    0: "python train.py --model linear",
    1: "python train.py --model ridge",
}

@acl.simulation_task
async def simulate(*args, **kwargs):
    return "python sim.py"

@acl.training_task
async def train(*args, **kwargs):
    return TRAIN_CMD[kwargs["learner_id"]]

@acl.active_learn_task
async def active_learn(*args, **kwargs):
    return "python active_learn.py"

@acl.as_stop_criterion(
    metric_name="mse", threshold=0.01
)
async def check_mse(*args, **kwargs):
    ...

# Build per-learner configs manually
from rose.learner import LearnerConfig, TaskConfig
lcs = [
    LearnerConfig(
        simulation={i: TaskConfig(kwargs={"learner_id": i})
                    for i in range(6)},
        ...
    )
    for lid in range(2)
]

async for state in acl.start(
    parallel_learners=2,
    max_iter=5,
    learner_configs=lcs,
):
    ...
```

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**YAML Spec**

```yaml
learner:
  type: parallel_active_learner
  max_iter: 5

learners:
  - label: linear_regression
    simulation:
      type: shell
      command: python sim.py --label a
    training:
      type: shell
      command: python train.py --label a --model linear
    active_learn:
      type: shell
      command: python active_learn.py --label a

  - label: ridge_regression
    simulation:
      type: shell
      command: python sim.py --label b
    training:
      type: shell
      command: python train.py --label b --model ridge
    active_learn:
      type: shell
      command: python active_learn.py --label b

stop_criterion:
  metric: mse
  threshold: 0.01
  operator: "<"
  evaluator:
    type: python
    function: tasks:check_mse
```

```python
# run it — builder handles routing + LearnerConfig
builder = LearnerBuilder("workflow.yaml", asyncflow)
cfg     = builder.config
learner = builder.build()
lcs     = builder.build_learner_configs()

async for state in learner.start(
    max_iter=cfg.learner.max_iter,
    parallel_learners=len(lcs),
    learner_configs=lcs,
):
    label = cfg.learners[state.learner_id].label
    print(f"[{label}]  iter {state.iteration}")
```

</div>
</div>

The builder creates a single dispatch closure per slot that routes to the correct command or function based on `learner_id`. The `learner_id` kwarg is consumed by the routing layer and never reaches the user function.

!!! note
    All learners in a `learners:` block must use the same task type (`python` or `shell`) for each slot. Mixed types within a slot are rejected at load time.

---

## Task Types

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**`type: python`**

Calls a Python function directly on the worker. Data flows in-memory between tasks. Function must be importable via `module:callable` syntax.

```yaml
simulation:
  type: python
  function: my_package.tasks:simulate
```

```python
# tasks.py
def simulate(*args, **kwargs):
    # receives positional results from
    # prior tasks as *args
    return {"X": ..., "y": ...}
```

The function must be async. The return value is passed as the first positional argument to the next task in the chain.

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**`type: shell`**

Runs in a single or multi-process on the worker. Data flows through files on disk. The command string is the return value — ROSE submits it as an executable.

```yaml
simulation:
  type: shell
  command: python sim.py
```

With `parameters:` placeholders:

```yaml
simulation:
  type: shell
  command: python sim.py --dataset {dataset}
```

`{dataset}` is filled from `parameters.dataset` at runtime via `str.format_map(kwargs)`.

The criterion evaluator must print a single float to stdout — ROSE captures stdout as the metric value.

</div>
</div>

### `task_description` example

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**Python API**

```python
@acl.simulation_task(as_executable=False)
async def simulate(*args, task_description={"process_templates": [(4, {})]}, **kwargs):
    ...
```

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**YAML Spec**

```yaml
simulation:
  type: python
  function: tasks:simulate
  task_description:
    process_templates:
      - [4, {}]
```

</div>
</div>

!!! note
    For parallel learners using the `learners:` block, all entries must use the same `task_description` for each slot — the backend registers it once at task-registration time.

---

## Stop Criterion

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; overflow: hidden;" markdown>
<div style="min-width: 0; overflow-x: auto;" markdown>

**Python API**

```python
@acl.as_stop_criterion(
    metric_name="mse",
    threshold=0.01,
    operator="<",
    as_executable=False,
)
async def check_mse(*args, **kwargs):
    ...  # return float
```

</div>
<div style="min-width: 0; overflow-x: auto;" markdown>

**YAML Spec**

```yaml
stop_criterion:
  metric: mse
  threshold: 0.01
  operator: "<"        # default; also: >, ==, <=, >=
  evaluator:
    type: python
    function: tasks:check_mse
```

</div>
</div>

The evaluator function follows the same `*args, **kwargs` convention as task functions. ROSE stops the loop when `metric_value <operator> threshold` is satisfied, or when `max_iter` is reached — whichever comes first.

---

## Parameters

The `parameters:` block defines key-value pairs that are injected into every task's `**kwargs` at every iteration:

```yaml
parameters:
  dataset: my_dataset
  batch_size: 32
  growing_pool: false
```

```python
def simulate(*args, **kwargs):
    dataset      = kwargs["dataset"]      # "my_dataset"
    batch_size   = kwargs["batch_size"]   # 32
    growing_pool = kwargs["growing_pool"] # False
    iteration    = kwargs["iteration"]    # 0, 1, 2, ...
    ...
```

### Reserved keys

The following keys are injected automatically by the builder and must not appear in `parameters:`:

| Key | Available in | Description |
|-----|-------------|-------------|
| `iteration` | all tasks | Current loop counter (0-based) |
| `pythonpath` | all tasks | Contents of `remote.pythonpath` as a list |
| `learner_id` | parallel tasks | Integer index of the learner (0, 1, …) |
| `learner_label` | parallel tasks (when `label` set) | Human-readable learner name from `learners[].label` |

---

## Remote Config

```yaml
remote:
  pythonpath:
    - /path/to/my/task/modules
    - /path/to/shared/utilities
  backends: [dragon_v3]   # optional — default shown; use [concurrent] for CPU-only runs
```

### `remote.pythonpath`

`remote.pythonpath` entries are added to `sys.path` on the remote worker before any task module is imported. They are also injected into every task's `kwargs["pythonpath"]` (as a list), so task functions can construct file paths without duplicating the value in `parameters:`:

```python
def train(sim_result, **kwargs):
    pythonpath = kwargs.get("pythonpath", [])
    base = pythonpath[0] if pythonpath else ""
    script = Path(base) / "scripts" / "train_model.py"
    ...
```

`remote.pythonpath` is the single edit point for the remote worker path — changing it updates both the import path and the kwarg.

### `remote.backends`

`remote.backends` selects which [Rhapsody](https://github.com/radical-cybertools/rhapsody) backends are instantiated on the remote orbit endpoint session. The value is a list of backend name strings:

| Value | Backend class | When to use |
|---|---|---|
| `dask` | `DaskExecutionBackend` | Dask-based execution |
| `concurrent` | `ConcurrentExecutionBackend` | CPU-only runs, no Dragon needed |
| `dragon` | `DragonExecutionBackendV3` |  For highly compute intensive surrogates |



Example — switch to `concurrent` for a CPU-only endpoint:

```yaml
remote:
  pythonpath: [...]
  backends: [concurrent]
```

`backends` is consumed only at engine-creation time and is **not** injected into task `kwargs`.

### `remote.target` — bootstrapping the endpoint for `rose run --remote`

`remote.target` tells `rose run <yaml> --remote` how to launch the remote
orbit endpoint itself, instead of assuming one is already running. Omit it
to keep today's manual-bootstrap behavior (start the endpoint yourself, use
`--local`/the orbit backend directly).

```yaml
remote:
  backends: [dragon_v3]
  target:
    kind: sfapi            # iri | sfapi | psij
    endpoint: nersc          # iri/sfapi only: nersc | olcf
    resource_id: perlmutter
    account: amsc007
    queue_name: debug
    walltime_min: 30
    n_nodes: 1
    constraint: cpu
    home_dir: /global/u2/m/merzky
    tunnel: none            # none | forward | reverse
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `remote.target.kind` | `iri` \| `sfapi` \| `psij` | yes | Bootstrap mechanism. Use `sfapi` for NERSC (IRI's `/compute/*` routes are broken server-side there); `iri` for OLCF; `psij` to submit via an already-connected login-node endpoint instead of IRI/SFAPI. |
| `remote.target.endpoint` | string | `iri`/`sfapi` only | `nersc` or `olcf`. |
| `remote.target.resource_id` | string | `iri`/`sfapi` only | Target compute resource, e.g. `perlmutter`, `odo`. |
| `remote.target.home_dir` | string | `iri`/`sfapi` only | User `$HOME` on the target — resolves the endpoint wrapper script path. |
| `remote.target.login_host` | string | when `tunnel: forward` | Login host to tunnel through. |
| `remote.target.edge_name` | string | `psij` only, unless `remote.embedded: true` | Name of the already-connected login-node endpoint to submit through. Not needed when `remote.embedded: true` — PsiJ runs on the embedded broker itself. |
| `remote.target.executor` | string | no | PsiJ executor name (default: `local`). |
| `remote.target.account` | string | yes | Allocation/project account. |
| `remote.target.queue_name` | string | no | Queue/partition. |
| `remote.target.walltime_min` | int | no | Default `30`. |
| `remote.target.n_nodes` | int | no | Default `1`. |
| `remote.target.constraint` | string | no | Scheduler constraint (e.g. `cpu`). |
| `remote.target.reservation` | string | no | Reservation name. |
| `remote.target.workdir` | string | no | Job working directory. |
| `remote.target.environment` | dict | no | Extra environment variables for the bootstrap job. |
| `remote.target.setup` | list[str] | no | Shell setup lines run before the endpoint starts. |
| `remote.target.tunnel` | `none` \| `forward` \| `reverse` | no | Default `none`. SSH tunnel mode for the endpoint's broker connection. |

Credentials are read from disk/env, never stored in the spec: `iri` reads a
bearer token from `~/.amsc/token_<endpoint>`; `sfapi` reads a client ID from
`$SFAPI_CLIENT_ID` and a private key from `~/.amsc/sfapi_key_<endpoint>.pem`.

### `remote.embedded` — run without a standalone broker

`remote.embedded: true` hosts the ORBIT broker inside the `rose run --remote`
process itself (`EndpointRuntime`/`EmbeddedBroker`) instead of connecting to
one already running elsewhere. No separate `radical-orbit-broker.py`
deployment is needed. Mutually exclusive with `remote.broker_url`.

```yaml
remote:
  embedded: true
  target:
    kind: psij
    account: amsc007
    queue_name: debug
    # edge_name omitted — PsiJ runs on the embedded broker itself
```

The embedded broker is still a **real** broker: it needs the same
operator-placed cert/key/token under `~/.radical/orbit` (or their env-var
redirects) as a standalone one — see `radical.orbit`'s `CLAUDE.md`
`TOKEN_RECIPE` if those aren't set up yet.

Combined with `target.kind: psij`, `target.edge_name` is not required — PsiJ
loads directly on the embedded broker (this only works when the process
running `rose run --remote` itself has batch-scheduler access, e.g. you're
on a login node). `target.kind: iri`/`sfapi` work the same as without
`embedded` — only where the broker lives changes.

---

## Tracking

```yaml
tracking:
  backend: mlflow      # mlflow | clearml | none (default)
  experiment: my-exp   # experiment/project name
  run_name: run-01     # optional run label
```

See [MLflow integration](../integrations/mlflow.md) and [ClearML integration](../integrations/clearml.md) for configuration details.

---

## Spec Variants with `workflow_with()`

`workflow_with()` returns a new `WorkflowSpec` with selective overrides applied. The original spec is never mutated:

```python
base_spec  = load_spec("workflow.yaml")
test_spec  = base_spec.workflow_with(max_iter=2, parameters={"dataset": "test_ds"})
large_spec = base_spec.workflow_with(max_iter=50, parameters={"batch_size": 128})
```

Accepted override keys:

- `parameters` — merged (not replaced) into the existing `parameters:` block
- Any `learner` field: `max_iter`, `parallel_learners`
- Any other top-level spec field

Unknown keys raise `ValueError` immediately.

---

## Import Validation

By default, `load_spec` does not import task modules — this is intentional when `remote.pythonpath` points to paths that only exist on the remote worker. To catch `function:` typos locally during development:

```python
spec = load_spec("workflow.yaml", validate_imports=True)
```

With `validate_imports=True`, every `module:callable` string is resolved in the current environment. All failures are collected and reported in a single `ValueError` before any infrastructure starts. Use this during development when task files are locally accessible.

---

## Spec Reference

### Top-level keys

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `learner` | object | yes | — | Learner type and loop settings |
| `learner.type` | string | yes | — | `sequential_active_learner`, `parallel_active_learner`, `sequential_reinforcement_learner`, `uq_active_learner` |
| `learner.max_iter` | int | no | `0` | Maximum iterations |
| `learner.parallel_learners` | int | no | `2` | Parallel learner count when `learners:` is absent |
| `simulation` / `training` / `active_learn` | TaskDef | AL yes | — | Task slots for Active Learning |
| `environment` / `update` | TaskDef | RL yes | — | Task slots for Reinforcement Learning |
| `prediction` / `active_learn` / `uncertainty` | TaskDef | UQ yes | — | Task slots for UQ-based AL |
| `learners` | list | no | — | Per-learner task definitions for heterogeneous parallel |
| `stop_criterion` | object | yes | — | Stopping condition |
| `parameters` | dict | no | `{}` | User-defined kwargs injected into all tasks |
| `remote.pythonpath` | list[str] | no | `[]` | Paths added to `sys.path` on worker; injected as `pythonpath` kwarg |
| `remote.backends` | list[str] | no | `["dragon_v3"]` | Rhapsody backends requested on the remote orbit session |
| `remote.broker_url` | string | no | `$RADICAL_ORBIT_BROKER_URL` | Broker URL override for `--remote` (no on-disk fallback — orbit resolves URL from CLI/API/env only) |
| `remote.target` | object | no | `null` | Bootstrap config for `--remote` — see [`remote.target`](#remotetarget--bootstrapping-the-endpoint-for-rose-run---remote) |
| `remote.embedded` | bool | no | `false` | Host the broker in-process instead of connecting to one — see [`remote.embedded`](#remoteembedded--run-without-a-standalone-broker) |
| `tracking.backend` | string | no | `none` | `mlflow` / `clearml` / `none` |
| `tracking.experiment` | string | no | `ROSE-Spec` | Experiment name |
| `tracking.run_name` | string | no | `null` | Run label |


### TaskDef fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `type` | `python` \| `shell` | yes | Task execution mode |
| `function` | string | if `type: python` | `module:callable` — dotted module path and function name |
| `command` | string | if `type: shell` | Shell command; supports `{param}` placeholders filled from `parameters:` |
| `task_description` | dict | no | Resource hints forwarded to the execution backend |
