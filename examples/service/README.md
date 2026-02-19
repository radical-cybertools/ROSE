# ROSE Service Examples

This directory contains examples for running ROSE workflows through the **ROSE Service** — a daemon-based workflow manager that accepts workflow submissions, tracks execution state, and provides real-time status updates.

The service uses file-based IPC: the manager polls a local directory for request files, and clients (CLI or Python) write JSON requests into that directory. A shared `registry.json` file reflects the live state of all workflows.

**Files in this directory:**

| File | Description |
|------|-------------|
| `service_test.yaml` | Minimal test workflow using `/bin/echo` (no dependencies required) |
| `service_real.yaml` | Real workflow using `ParallelActiveLearner` with Python scripts |
| `run_service.py` | Integration example: launches, submits, monitors, and shuts down programmatically |
| `verify_service.py` | Demonstrates workflow cancellation flow |

---

## Workflow YAML Format

Both examples use a YAML file to define the workflow. The service loads this file, instantiates the appropriate learner, and registers the component tasks.

```yaml
learner:
  type: SequentialActiveLearner   # or ParallelActiveLearner

components:
  simulation:
    type: script                  # or "function" for a Python callable
    path: /bin/echo
    config:
      args: ["Simulation Step"]
  training:
    type: script
    path: /bin/echo
    config:
      args: ["Training Step"]
  active_learn:
    type: script
    path: /bin/echo
    config:
      args: ["AL Step"]

config:
  max_iterations: 3
  work_dir: /tmp/rose_test
```

---

## Option 1 — CLI (Two Terminals)

The CLI is the simplest way to interact with ROSE service. You need two terminal sessions: one to run the service daemon and one to submit and monitor workflows.

> The `--job-id` flag identifies the service instance. If you are inside a SLURM job, it defaults to `$SLURM_JOB_ID`. For local usage, it defaults to `local_job_0`. Both terminals must use the same job ID.

### Terminal 1 — Start the Service

```bash
rose launch
```

The service starts and blocks, printing log output as workflows are received and executed. Keep this terminal open for the lifetime of the session.

To use a custom job ID (e.g. for running multiple isolated services):

```bash
rose launch --job-id my_session
```

### Terminal 2 — Submit and Monitor a Workflow

**Submit a workflow:**

```bash
rose submit examples/service/service_test.yaml
```

Output:

```
Submitted workflow request.
Request ID:  3f2a1b4c-...
Workflow ID: wf.3f2a1b4c
```

**Check workflow status:**

```bash
rose status wf.3f2a1b4c
```

Output (example):

```json
{
  "wf_id": "wf.3f2a1b4c",
  "state": "running",
  "workflow_file": "/path/to/service_test.yaml",
  "start_time": 1700000000.0,
  "end_time": null,
  "stats": {
    "iteration": 2,
    "metric_value": null
  },
  "error": null
}
```

**List all workflows:**

```bash
rose status
```

**Cancel a running workflow:**

```bash
rose cancel wf.3f2a1b4c
```

**Shut down the service when done:**

```bash
rose shutdown
```

> The service in Terminal 1 will exit gracefully after receiving the shutdown request.

If you launched with a custom `--job-id`, pass the same flag to all client commands:

```bash
rose submit --job-id my_session examples/service/service_test.yaml
rose status  --job-id my_session
rose shutdown --job-id my_session
```

---

## Option 2 — Python Client

Use `ServiceClient` from `rose.service.client` to drive the service programmatically — useful for integration scripts, notebooks, or automated pipelines.

See [`run_service.py`](run_service.py) for a complete working example. It covers the full lifecycle:

1. Start `ServiceManager` as a background `asyncio` task
2. Initialize `ServiceClient` with the same job ID
3. Call `client.submit_workflow(path)` to submit a YAML workflow
4. Derive the workflow ID with `ServiceClient.get_wf_id(req_id)`
5. Poll `client.get_workflow_status(wf_id)` until the state reaches `COMPLETED`, `FAILED`, or `CANCELED`
6. Call `client.shutdown()` to stop the service

Run it with:

```bash
python examples/service/run_service.py
```

See [`verify_service.py`](verify_service.py) for an example of cancellation via `client.cancel_workflow(wf_id)`.

**Key `ServiceClient` methods:**

| Method | Description |
|--------|-------------|
| `submit_workflow(path)` | Submit a YAML workflow file; returns a `req_id` |
| `ServiceClient.get_wf_id(req_id)` | Derive the workflow ID from a request ID |
| `get_workflow_status(wf_id)` | Return the current state dict for a workflow |
| `list_workflows()` | Return all workflows from the registry |
| `cancel_workflow(wf_id)` | Request cancellation of a running workflow |
| `shutdown()` | Send a graceful shutdown request to the service |

---

## Option 3 — REST API *(upcoming)*

> **Not yet implemented.** A REST API for ROSE service is planned for a future release.

The REST API will expose the same operations as the CLI and Python client over HTTP, making it possible to submit and monitor workflows from any language or tool (e.g. `curl`, JavaScript, or remote machines).

Planned endpoints:

```
POST   /workflows              Submit a new workflow
GET    /workflows              List all workflows
GET    /workflows/{wf_id}      Get status of a specific workflow
DELETE /workflows/{wf_id}      Cancel a workflow
POST   /shutdown               Gracefully stop the service
```

When available, a workflow submission will look like:

```bash
curl -X POST http://localhost:8080/workflows \
     -H "Content-Type: application/json" \
     -d '{"workflow_file": "/path/to/workflow.yaml"}'
```

Stay tuned for updates.
