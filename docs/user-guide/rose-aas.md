# ROSE as a Service

This page describes ROSE's service model: how the active-learning/RL loop you define stays under your control while the actual simulation and training work executes on HPC, and what that does and does not require from you today.

---

## Core idea: BYOF — Bring Your Own Workflow

ROSE does not own the science. It owns the **orchestration loop**: submit a task, wait for it, decide what runs next, check a stop criterion, repeat — until `max_iter` or the criterion is met.

You bring:

- A `simulate` / `train` / `active_learn` (or `environment` / `update`, or `prediction` / `uncertainty`) implementation, as a plain Python function or a shell command.
- The environment that implementation needs (packages, data, scripts).
- The decision of where it should run — a laptop, a login node, or a leadership-class HPC allocation.

ROSE brings the orchestration: dependency-correct task submission, iteration bookkeeping, stop-criterion evaluation, checkpoint-safe state, and tracking integration. The [YAML Spec API](spec-api.md) is the declarative form of this contract — `function: tasks:simulate` is a pointer to *your* code, not a hook into ROSE's.

This is the same boundary that makes ROSE usable across domains without ROSE having to understand any of them: it sees `*args, **kwargs` in, a return value out, nothing about what happened in between.

---

## Two places work can happen

A ROSE learner doesn't run anything itself — it submits tasks through whichever `asyncflow` execution backend you hand it. Two backends matter for the "as a Service" framing:

| Backend | Where the orchestration loop runs | Where tasks execute | Documented at |
|---|---|---|---|
| `DragonExecutionBackendV3` (RHAPSODY) | Wherever your script runs — it submits a pilot job and blocks inside it | Inside the pilot job it just submitted | [Target Resources](target-resources.md) |
| Edge backend (RHAPSODY, `bridge_url` + `edge_name`) | Wherever your script runs — does **not** need to be the HPC machine | Inside a separate, already-running edge agent — possibly on a different machine entirely | this page |

The first model is "submit a job, then run my loop inside it." The second is "run my loop here; dispatch tasks to a job running somewhere else." That second model is what makes ROSE service-like: the loop's control plane (your `learner.start()` call, deciding when to stop, talking to MLflow/ClearML) is decoupled from the compute plane (the HPC allocation actually executing `simulate`/`train`).

---

## How ROSE operates within a job

The edge backend connects two things over a bridge:

1. **An edge agent**, running inside an HPC job allocation. The job itself is requested through whatever your site normally uses to get an allocation — it doesn't have to be requested by ROSE. Once the job starts, the edge agent comes up inside it and stays alive for the allocation's lifetime, ready to accept task descriptions.
2. **Your orchestration process**, running anywhere — your laptop, a long-lived service host, a CI runner. It builds the learner from your spec (`LearnerBuilder(cfg, asyncflow).build()`), starts the loop (`learner.start(...)`), and for every `simulation`/`training`/`active_learn`/... task it submits, the asyncflow engine forwards that task description over the bridge to the edge, waits for the result, and resumes the loop.

The practical effect: you submit a ROSE workflow to HPC without an interactive session on the cluster, and without your laptop needing to stay connected to the scheduler — only to the bridge. The job allocation is what's expensive and scheduler-queued; your control process is cheap and can be restarted independently of it. Stop-criterion checks, `set_next_config()` decisions, and tracking calls all happen on your side of the bridge, on every iteration, with no per-iteration job resubmission.

This is additive to the model in [Target Resources](target-resources.md), not a replacement — both go through the same `WorkflowEngine`/`LearnerBuilder` plumbing. Which backend you choose only changes *where* the edge lives; the spec, the learner, and the loop semantics are identical either way.

---

## What ROSE automates for you today

- **The loop itself** — iterate, await, check criterion, repeat, with no boilerplate beyond defining your tasks and the threshold.
- **Preemption-safe state** — every completed iteration is durable before the next starts, so a killed job loses at most the in-flight iteration.
- **Tracking** — `tracking.backend: mlflow | clearml` in the spec wires a tracker once; every iteration's metrics, params, and lifecycle events are reported automatically, with no tracking code inside your task functions.
- **Heterogeneous dispatch** — the same loop drives CPU-only, GPU, MPI, or shell-executable tasks, and (via the `learners:` block) distinct task implementations per parallel learner.

## What ROSE does **not** yet automate: data movement

Today, getting a simulation's output into the training task's hands is entirely your task code's responsibility:

- `type: python` slots pass the previous task's return value in-memory, as the first positional argument — this is implicit in the task type, not something you declare.
- `type: shell` slots pass nothing automatically; your command's stdout becomes the next task's input only if your scripts agree on a file path or convention outside the spec (this is exactly the pattern in the M3DC1 use case, where `simulate`/`train`/`active_learn` hand-roll a namespaced directory of files to communicate).

**Planned:** a `DataExchangeProtocol` field in the YAML spec — `FileBased` or `MemoryBased` — that makes this an explicit, ROSE-managed choice instead of an implicit consequence of task type:

- `MemoryBased` formalizes what `type: python` already does today — in-process object passing between tasks on the same worker.
- `FileBased` would let ROSE own staging a per-iteration (and per-learner, for parallel runs) working directory, instead of every use case reimplementing its own namespace/path convention by hand inside task code.

This isn't implemented yet — it's the next piece of surface area on the spec, and the natural place to close the "no environment/output contract" gap without ROSE needing to know anything about the science moving through it.

---

## What you bring, concretely

| You supply | ROSE does not check this for you (today) |
|---|---|
| Task implementations (`simulate`, `train`, `active_learn`, criterion evaluator, ...) matching the `*args, **kwargs` convention | Whether return shapes match what the next task expects |
| The runtime environment those implementations need on the worker | No environment/dependency manifest in the spec |
| Access to wherever the edge agent's job runs (account, queue, allocation) | Out of scope for the spec layer entirely |
| Your data, and a convention for how tasks find it (today: `parameters:` + your own file paths) | No declared output/data contract yet — see `DataExchangeProtocol` above |
| Correct `remote.pythonpath` if task modules aren't already importable where the edge runs | `validate_imports=True` only checks importability in *your local* environment |

None of this is unique to the service model — it's the same BYOF boundary as running ROSE locally. What the service model changes is *where* that boundary sits physically: your task code and its dependencies need to be reachable from the edge's job allocation, not from your laptop.
