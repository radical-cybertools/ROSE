"""
ROSE Plugin for RADICAL-Edge
=============================

This module provides a RADICAL-Edge plugin for ROSE (Remote Online Smart
Experiment) workflow management. It enables submission, monitoring, and
cancellation of Active Learning workflows through REST endpoints.

Architecture
------------
The plugin embeds workflow execution directly within the Edge service,
eliminating the need for a separate ServiceManager daemon. Each RoseSession
maintains its own WorkflowEngine and executes learner loops as async tasks.

::

    Client (Python/curl/browser)
        ↓ HTTP/REST
    RADICAL-Edge Bridge
        ↓ WebSocket
    Edge Service (with ROSE plugin)
        ↓
    WorkflowEngine / Learners (embedded)

Components
----------
- **PluginRose**: The plugin class registered with RADICAL-Edge. Defines REST
  routes and UI configuration for portal integration.

- **RoseSession**: Server-side session managing workflow execution. Each
  session lazily initializes a WorkflowEngine and tracks all submitted
  workflows.

- **RoseClient**: Application-side client providing synchronous methods for
  workflow operations.

REST Endpoints
--------------
- ``POST /rose/register_session`` - Create a new session
- ``POST /rose/submit/{sid}`` - Submit a workflow YAML
- ``GET  /rose/status/{sid}/{wf_id}`` - Get workflow status
- ``GET  /rose/workflows/{sid}`` - List all workflows
- ``POST /rose/cancel/{sid}/{wf_id}`` - Cancel a workflow
- ``POST /rose/unregister_session/{sid}`` - Close session

Notifications
-------------
The plugin sends real-time notifications via SSE when workflow state changes.
Clients can subscribe using ``RoseClient.on_workflow_state(callback)``.

See Also
--------
- ``rose.service.manager.WorkflowLoader`` - YAML parsing and learner creation
- ``rose.al.active_learner`` - SequentialActiveLearner, ParallelActiveLearner
- ``radical.edge.plugin_base.Plugin`` - Base plugin class
"""

__author__ = "RADICAL Development Team"
__email__ = "radical@radical-project.org"
__copyright__ = "Copyright 2024, RADICAL@Rutgers"
__license__ = "MIT"


import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from radical.asyncflow import LocalExecutionBackend, WorkflowEngine
from radical.edge.client import PluginClient
from radical.edge.plugin_base import Plugin
from radical.edge.plugin_session_base import PluginSession
from radical.edge.ui_schema import (
    UIConfig,
    UIField,
    UIForm,
    UIFormSubmit,
    UIMonitor,
    UINotifications,
)
from starlette.responses import JSONResponse

from rose.al.active_learner import ParallelActiveLearner
from rose.service.manager import WorkflowLoader
from rose.service.models import Workflow, WorkflowState

log = logging.getLogger("radical.edge")


# ------------------------------------------------------------------------------
#
class RoseSession(PluginSession):
    """ROSE session (service-side).

    Directly manages workflow execution using AsyncFlow, eliminating the need for a separate
    ServiceManager process.
    """

    # --------------------------------------------------------------------------
    #
    def __init__(self, sid: str):
        """Initialize a RoseSession.

        Args:
            sid (str): Unique session identifier assigned by the plugin.
        """
        super().__init__(sid)

        self._workflows: dict[str, Workflow] = {}
        self._learner_tasks: dict[str, asyncio.Task] = {}
        self._engine: WorkflowEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._initialized = False

    # --------------------------------------------------------------------------
    #
    async def _ensure_engine(self):
        """Lazily initialize the workflow engine."""
        if self._engine is not None:
            return

        async with self._engine_lock:
            if self._engine is not None:
                return

            log.info(f"[{self.sid}] Initializing workflow engine")
            backend = LocalExecutionBackend()
            self._engine = await WorkflowEngine.create(backend)
            self._initialized = True
            log.info(f"[{self.sid}] Workflow engine ready")

    # --------------------------------------------------------------------------
    #
    async def submit_workflow(self, workflow_file: str) -> dict:
        """Submit a workflow YAML file for execution.

        Args:
            workflow_file (str): Absolute or relative path to the workflow YAML.

        Returns:
            dict: ``{wf_id}`` — the workflow ID.
        """
        self._check_active()
        await self._ensure_engine()

        # Generate workflow ID
        wf_id = f"wf.{uuid.uuid4().hex[:8]}"

        # Create workflow record
        wf = Workflow(wf_id=wf_id, state=WorkflowState.SUBMITTED, workflow_file=workflow_file)
        self._workflows[wf_id] = wf

        # Notify submission
        if self._notify:
            self._notify(
                "workflow_state",
                {"wf_id": wf_id, "state": "SUBMITTED", "workflow_file": workflow_file},
            )

        # Start workflow execution in background
        task = asyncio.create_task(self._run_workflow(wf))
        self._learner_tasks[wf_id] = task

        log.info(f"[{self.sid}] Submitted workflow {wf_id}: {workflow_file}")
        return {"wf_id": wf_id}

    # --------------------------------------------------------------------------
    #
    async def _run_workflow(self, wf: Workflow):
        """Execute a workflow (runs as background task)."""
        wf_id = wf.wf_id

        try:
            # Initialize
            wf.state = WorkflowState.INITIALIZING
            self._notify_state(wf)

            # Load workflow definition
            wf_def = WorkflowLoader.load_yaml(wf.workflow_file)
            learner, initial_config = WorkflowLoader.create_learner(wf_id, wf_def, self._engine)
            wf.learner_instance = learner

            # Run
            wf.state = WorkflowState.RUNNING
            wf.start_time = time.time()
            self._notify_state(wf)

            config = wf_def.get("config", {})
            learner_cfg = wf_def.get("learner", {})
            max_iter = config.get("max_iterations", learner_cfg.get("max_iterations", 10))

            log.info(f"[{self.sid}] Running workflow {wf_id} (max_iterations={max_iter})")

            if isinstance(learner, ParallelActiveLearner):
                parallel = config.get("parallel_learners", learner_cfg.get("parallel_learners", 2))
                configs = [initial_config] * parallel if initial_config else None

                results = await learner.start(
                    parallel_learners=parallel, max_iter=max_iter, learner_configs=configs
                )
                wf.stats = {"parallel_results": [str(r) for r in results]}

            else:
                # Sequential learner - async iterator
                async for state in learner.start(max_iter=max_iter, initial_config=initial_config):
                    wf.stats = state.to_dict()
                    log.info(
                        f"[{self.sid}] {wf_id} iteration {state.iteration} "
                        f"(metric={state.metric_value})"
                    )
                    self._notify_state(wf)

            # Completed
            wf.state = WorkflowState.COMPLETED
            wf.end_time = time.time()
            log.info(f"[{self.sid}] Workflow {wf_id} completed")

        except asyncio.CancelledError:
            wf.state = WorkflowState.CANCELED
            wf.end_time = time.time()
            log.info(f"[{self.sid}] Workflow {wf_id} canceled")

        except Exception as e:
            wf.state = WorkflowState.FAILED
            wf.error = str(e)
            wf.end_time = time.time()
            log.error(f"[{self.sid}] Workflow {wf_id} failed: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self._notify_state(wf)
            self._learner_tasks.pop(wf_id, None)

    # --------------------------------------------------------------------------
    #
    def _notify_state(self, wf: Workflow):
        """Send workflow state notification."""
        if self._notify:
            self._notify(
                "workflow_state",
                {"wf_id": wf.wf_id, "state": wf.state.value, "stats": wf.stats, "error": wf.error},
            )

    # --------------------------------------------------------------------------
    #
    async def get_workflow_status(self, wf_id: str) -> dict:
        """Return the current status of a workflow.

        Args:
            wf_id (str): The workflow ID.

        Returns:
            dict: Workflow state dictionary.

        Raises:
            HTTPException(404): If the workflow ID is not found.
        """
        self._check_active()

        wf = self._workflows.get(wf_id)
        if not wf:
            raise HTTPException(status_code=404, detail=f"workflow '{wf_id}' not found")

        return wf.to_dict()

    # --------------------------------------------------------------------------
    #
    async def list_workflows(self) -> dict:
        """List all workflows in this session.

        Returns:
            dict: Mapping ``wf_id → state dict``.
        """
        self._check_active()

        return {wf_id: wf.to_dict() for wf_id, wf in self._workflows.items()}

    # --------------------------------------------------------------------------
    #
    async def cancel_workflow(self, wf_id: str) -> dict:
        """Cancel a running workflow.

        Args:
            wf_id (str): The workflow ID to cancel.

        Returns:
            dict: ``{wf_id}`` confirming the cancellation.
        """
        self._check_active()

        wf = self._workflows.get(wf_id)
        if not wf:
            raise HTTPException(status_code=404, detail=f"workflow '{wf_id}' not found")

        if wf.state not in (
            WorkflowState.RUNNING,
            WorkflowState.INITIALIZING,
            WorkflowState.SUBMITTED,
        ):
            raise HTTPException(status_code=400, detail=f"workflow '{wf_id}' not running")

        # Stop the learner
        if wf.learner_instance:
            wf.learner_instance.stop()

        # Cancel the task
        task = self._learner_tasks.get(wf_id)
        if task and not task.done():
            task.cancel()

        log.info(f"[{self.sid}] Canceling workflow {wf_id}")

        if self._notify:
            self._notify("workflow_state", {"wf_id": wf_id, "state": "CANCELING"})

        return {"wf_id": wf_id}

    # --------------------------------------------------------------------------
    #
    async def close(self) -> dict:
        """Close this session, stopping all workflows and cleaning up."""
        log.info(f"[{self.sid}] Closing session")

        # Stop all learners
        for wf in self._workflows.values():
            if wf.learner_instance and wf.state == WorkflowState.RUNNING:
                wf.learner_instance.stop()

        # Cancel all tasks
        for task in self._learner_tasks.values():
            if not task.done():
                task.cancel()

        if self._learner_tasks:
            await asyncio.gather(*self._learner_tasks.values(), return_exceptions=True)
            self._learner_tasks.clear()

        # Shutdown engine
        if self._engine:
            await self._engine.shutdown()
            self._engine = None

        return await super().close()


# ------------------------------------------------------------------------------
#
class RoseClient(PluginClient):
    """Application-side client for the ROSE plugin.

    Provides a thin sync wrapper over the HTTP endpoints exposed by
    ``PluginRose``.
    """

    # --------------------------------------------------------------------------
    #
    def on_workflow_state(self, callback):
        """Register a callback for workflow state change notifications.

        Args:
            callback: A callable(topic, data) to invoke on state changes.
        """
        self.register_notification_callback(callback)

    # --------------------------------------------------------------------------
    #
    def off_workflow_state(self, callback):
        """Unregister a workflow state change callback.

        Args:
            callback: The callback to unregister.
        """
        self.unregister_notification_callback(callback)

    # --------------------------------------------------------------------------
    #
    def submit_workflow(self, workflow_file: str) -> dict:
        """Submit a workflow YAML file.

        Args:
            workflow_file (str): Path to the workflow YAML file.

        Returns:
            dict: ``{wf_id}``.
        """
        if not self.sid:
            raise RuntimeError("No active session")

        resp = self._http.post(
            self._url(f"submit/{self.sid}"), json={"workflow_file": workflow_file}
        )
        resp.raise_for_status()

        return resp.json()

    # --------------------------------------------------------------------------
    #
    def get_workflow_status(self, wf_id: str) -> dict:
        """Get the current status of a workflow.

        Args:
            wf_id (str): Workflow ID.

        Returns:
            dict: Workflow state dictionary.
        """
        if not self.sid:
            raise RuntimeError("No active session")

        resp = self._http.get(self._url(f"status/{self.sid}/{wf_id}"))
        resp.raise_for_status()

        return resp.json()

    # --------------------------------------------------------------------------
    #
    def list_workflows(self) -> dict:
        """List all workflows in the session.

        Returns:
            dict: Registry mapping ``wf_id → state dict``.
        """
        if not self.sid:
            raise RuntimeError("No active session")

        resp = self._http.get(self._url(f"workflows/{self.sid}"))
        resp.raise_for_status()

        return resp.json()

    # --------------------------------------------------------------------------
    #
    def cancel_workflow(self, wf_id: str) -> dict:
        """Cancel a running workflow.

        Args:
            wf_id (str): Workflow ID to cancel.

        Returns:
            dict: ``{wf_id}``.
        """
        if not self.sid:
            raise RuntimeError("No active session")

        resp = self._http.post(self._url(f"cancel/{self.sid}/{wf_id}"))
        resp.raise_for_status()

        return resp.json()


# ------------------------------------------------------------------------------
#
class PluginRose(Plugin):
    """ROSE plugin for RADICAL-Edge.

    Exposes workflow management via REST endpoints, with embedded execution
    (no separate ServiceManager process required).

    Routes:
    - POST /rose/register_session
    - POST /rose/unregister_session/{sid}
    - POST /rose/submit/{sid}
    - GET  /rose/status/{sid}/{wf_id}
    - GET  /rose/workflows/{sid}
    - POST /rose/cancel/{sid}/{wf_id}
    """

    plugin_name = "rose"
    session_class = RoseSession
    client_class = RoseClient
    version = "0.2.0"
    session_ttl = 0  # No timeout - workflows can run for hours/days

    ui_config = UIConfig(
        icon="🌹",
        title="ROSE Active Learning",
        description="Submit and monitor Active Learning workflows",
        refresh_button=True,
        forms=[
            UIForm(
                id="submit",
                title="Submit Workflow",
                layout="single",
                fields=[
                    UIField(
                        name="workflow_file",
                        type="text",
                        label="Workflow File",
                        placeholder="/path/to/workflow.yaml",
                        required=True,
                    )
                ],
                submit=UIFormSubmit(label="Submit", style="success", endpoint="submit/{sid}"),
            )
        ],
        monitors=[
            UIMonitor(
                id="workflows",
                title="Workflows",
                type="task_list",
                css_class="workflow-list",
                empty_text="No workflows submitted yet",
                auto_load="workflows/{sid}",
            )
        ],
        notifications=UINotifications(
            topic="workflow_state", id_field="wf_id", state_field="state"
        ),
    )

    # --------------------------------------------------------------------------
    #
    def __init__(self, app: FastAPI, instance_name: str = "rose"):
        """Initialize the ROSE plugin, registering all routes."""
        super().__init__(app, instance_name)

        self.add_route_post("submit/{sid}", self.submit_workflow)
        self.add_route_get("status/{sid}/{wf_id}", self.get_workflow_status)
        self.add_route_get("workflows/{sid}", self.list_workflows)
        self.add_route_post("cancel/{sid}/{wf_id}", self.cancel_workflow)

        self._log_routes()

    # --------------------------------------------------------------------------
    #
    async def submit_workflow(self, request: Request) -> JSONResponse:
        """Submit a workflow YAML file."""
        sid = request.path_params["sid"]
        data = await request.json()

        return await self._forward(
            sid, RoseSession.submit_workflow, workflow_file=data.get("workflow_file")
        )

    # --------------------------------------------------------------------------
    #
    async def get_workflow_status(self, request: Request) -> JSONResponse:
        """Return the status of a specific workflow."""
        sid = request.path_params["sid"]
        wf_id = request.path_params["wf_id"]

        return await self._forward(sid, RoseSession.get_workflow_status, wf_id=wf_id)

    # --------------------------------------------------------------------------
    #
    async def list_workflows(self, request: Request) -> JSONResponse:
        """List all workflows in the session."""
        sid = request.path_params["sid"]

        return await self._forward(sid, RoseSession.list_workflows)

    # --------------------------------------------------------------------------
    #
    async def cancel_workflow(self, request: Request) -> JSONResponse:
        """Cancel a running workflow."""
        sid = request.path_params["sid"]
        wf_id = request.path_params["wf_id"]

        return await self._forward(sid, RoseSession.cancel_workflow, wf_id=wf_id)


# ------------------------------------------------------------------------------
