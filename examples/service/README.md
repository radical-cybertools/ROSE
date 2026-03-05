# ROSE Service Examples

This directory contains examples for running ROSE workflows through the **ROSE Service** — a daemon-based workflow manager that accepts workflow submissions, tracks execution state, and provides real-time status updates.

The service uses file-based IPC: the manager polls a local directory for request files, and clients (CLI or Python) write JSON requests into that directory. A shared `registry.json` file reflects the live state of all workflows.

**Files in this directory:**

| File | Description |
|------|-------------|
| `service_test.yaml` | Minimal test workflow using `/bin/echo` (no dependencies required) |
| `service_real.yaml` | Real workflow using `ParallelActiveLearner` with Python scripts |
| `debug_workflow.yaml` | Fast test workflow for plugin testing (2 iterations) |
| `run_service.py` | Integration example: launches, submits, monitors, and shuts down programmatically |
| `verify_service.py` | Demonstrates workflow cancellation flow |
| `example_rose_plugin.py` | REST API example using RADICAL-Edge plugin |

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
rose launch --job-id job.000001
```

### Terminal 2 — Submit and Monitor a Workflow

**Submit a workflow:**

```bash
rose submit --job-id job.000001 examples/service/service_test.yaml
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

## Option 3 — REST API via RADICAL-Edge Plugin

The ROSE plugin for RADICAL-Edge provides a REST API for workflow management. This is the recommended approach for remote access and integration with other services.

**Architecture:**
```
Client (Python/curl/browser)
    ↓ HTTP/REST
RADICAL-Edge Bridge
    ↓ WebSocket
Edge Service (with ROSE plugin)
    ↓
WorkflowEngine / Learners (embedded)
```

The plugin embeds workflow execution directly — no separate `rose launch` daemon required.

### Prerequisites

1. RADICAL-Edge bridge running
2. RADICAL-Edge service running with ROSE plugin loaded

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rose/register_session` | Register a new session |
| `POST` | `/rose/submit/{sid}` | Submit a workflow |
| `GET` | `/rose/status/{sid}/{wf_id}` | Get workflow status |
| `GET` | `/rose/workflows/{sid}` | List all workflows |
| `POST` | `/rose/cancel/{sid}/{wf_id}` | Cancel a workflow |
| `POST` | `/rose/unregister_session/{sid}` | Close session |

### Python Client Example

See [`example_rose_plugin.py`](example_rose_plugin.py) for a complete working example.

```python
from radical.edge import BridgeClient
import rose.service.api.rest  # Register plugin

# Connect to bridge
bc = BridgeClient(url='https://localhost:8000')
edges = bc.list_edges()
ec = bc.get_edge_client(edges[0])

# Get ROSE plugin client
rose = ec.get_plugin('rose')

# Submit workflow
result = rose.submit_workflow('/path/to/workflow.yaml')
wf_id = result['wf_id']

# Monitor status
status = rose.get_workflow_status(wf_id)
print(f"State: {status['state']}")

# List all workflows
workflows = rose.list_workflows()

# Cancel if needed
rose.cancel_workflow(wf_id)

# Cleanup
rose.close()
bc.close()
```

### Notifications

The plugin sends real-time notifications via SSE when workflow state changes:

```python
def on_state_change(topic, data):
    print(f"Workflow {data['wf_id']}: {data['state']}")

rose.on_workflow_state(on_state_change)
```

### Running the Example

```bash
# Set bridge URL
export RADICAL_BRIDGE_URL=https://localhost:8000

# Run example
python example_rose_plugin.py --workflow debug_workflow.yaml
```

---

## Additional Files

| File | Description |
|------|-------------|
| `example_rose_plugin.py` | REST API example using RADICAL-Edge plugin |
| `debug_workflow.yaml` | Fast test workflow (2 iterations, ~6 seconds) |
