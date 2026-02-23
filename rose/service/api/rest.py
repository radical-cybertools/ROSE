
__author__    = 'RADICAL Development Team'
__email__     = 'radical@radical-project.org'
__copyright__ = 'Copyright 2024, RADICAL@Rutgers'
__license__   = 'MIT'


import asyncio
import uuid
import logging

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from radical.edge.plugin_session_base import PluginSession
from radical.edge.plugin_base         import Plugin
from radical.edge.client              import PluginClient

from rose.service.client import ServiceClient


log = logging.getLogger("radical.edge")


# ------------------------------------------------------------------------------
#
class RoseSession(PluginSession):
    """
    ROSE session (service-side).

    Wraps a ``ServiceClient`` instance, forwarding workflow submission,
    status queries, cancellation, and service shutdown to a running ROSE
    service identified by its job ID.
    """

    # --------------------------------------------------------------------------
    #
    def __init__(self, sid: str, job_id: str = 'local_job_0'):
        """
        Initialize a RoseSession.

        Args:
            sid    (str): Unique session identifier assigned by the plugin.
            job_id (str): The ROSE service job ID to connect to.
                          Defaults to 'local_job_0' (local, non-SLURM usage).
        """
        super().__init__(sid)

        self._job_id = job_id
        self._client = ServiceClient(job_id)


    # --------------------------------------------------------------------------
    #
    async def submit_workflow(self, workflow_file: str) -> dict:
        """
        Submit a workflow YAML file to the ROSE service.

        Args:
            workflow_file (str): Absolute or relative path to the workflow YAML.

        Returns:
            dict: ``{req_id, wf_id}`` — the request ID and the derived workflow ID.
        """
        self._check_active()

        req_id = await asyncio.to_thread(self._client.submit_workflow,
                                         workflow_file)
        wf_id  = ServiceClient.get_wf_id(req_id)

        return {'req_id': req_id, 'wf_id': wf_id}


    # --------------------------------------------------------------------------
    #
    async def get_workflow_status(self, wf_id: str) -> dict:
        """
        Return the current status of a workflow.

        Args:
            wf_id (str): The workflow ID (e.g. ``wf.3f2a1b4c``).

        Returns:
            dict: Workflow state dictionary from the service registry.

        Raises:
            HTTPException(404): If the workflow ID is not found.
        """
        self._check_active()

        status = await asyncio.to_thread(self._client.get_workflow_status,
                                         wf_id)
        if not status:
            raise HTTPException(status_code=404,
                                detail=f"workflow '{wf_id}' not found")

        return status


    # --------------------------------------------------------------------------
    #
    async def list_workflows(self) -> dict:
        """
        List all workflows tracked by the ROSE service.

        Returns:
            dict: Full registry mapping ``wf_id → state dict``.
        """
        self._check_active()

        return await asyncio.to_thread(self._client.list_workflows)


    # --------------------------------------------------------------------------
    #
    async def cancel_workflow(self, wf_id: str) -> dict:
        """
        Request cancellation of a running workflow.

        Args:
            wf_id (str): The workflow ID to cancel.

        Returns:
            dict: ``{req_id, wf_id}`` confirming the cancellation request.
        """
        self._check_active()

        req_id = await asyncio.to_thread(self._client.cancel_workflow, wf_id)

        return {'req_id': req_id, 'wf_id': wf_id}


    # --------------------------------------------------------------------------
    #
    async def shutdown(self) -> dict:
        """
        Send a graceful shutdown request to the ROSE service.

        Returns:
            dict: ``{req_id}`` confirming the shutdown request was queued.
        """
        self._check_active()

        req_id = await asyncio.to_thread(self._client.shutdown)

        return {'req_id': req_id}


    # --------------------------------------------------------------------------
    #
    async def close(self) -> dict:
        """
        Close this session. ServiceClient is stateless so no teardown needed.
        """
        self._client = None

        return await super().close()


# ------------------------------------------------------------------------------
#
class RoseClient(PluginClient):
    """
    Application-side client for the ROSE plugin.

    Provides a thin sync wrapper over the HTTP endpoints exposed by
    ``PluginRose``, mirroring the same operations available through the
    ``rose`` CLI and ``ServiceClient``.
    """

    # --------------------------------------------------------------------------
    #
    def register_session(self, job_id: str = 'local_job_0'):
        """
        Register a session with the ROSE plugin, binding it to a job ID.

        Args:
            job_id (str): The ROSE service job ID to connect to.
                          Defaults to 'local_job_0'.
        """
        resp = self._http.post(self._url('register_session'),
                               json={'job_id': job_id})
        resp.raise_for_status()
        self._sid = resp.json()['sid']


    # --------------------------------------------------------------------------
    #
    def submit_workflow(self, workflow_file: str) -> dict:
        """
        Submit a workflow YAML file.

        Args:
            workflow_file (str): Path to the workflow YAML file.

        Returns:
            dict: ``{req_id, wf_id}``.
        """
        if not self.sid:
            raise RuntimeError('No active session')

        resp = self._http.post(self._url(f'submit/{self.sid}'),
                               json={'workflow_file': workflow_file})
        resp.raise_for_status()

        return resp.json()


    # --------------------------------------------------------------------------
    #
    def get_workflow_status(self, wf_id: str) -> dict:
        """
        Get the current status of a workflow.

        Args:
            wf_id (str): Workflow ID.

        Returns:
            dict: Workflow state dictionary.
        """
        if not self.sid:
            raise RuntimeError('No active session')

        resp = self._http.get(self._url(f'status/{self.sid}/{wf_id}'))
        resp.raise_for_status()

        return resp.json()


    # --------------------------------------------------------------------------
    #
    def list_workflows(self) -> dict:
        """
        List all workflows in the connected ROSE service.

        Returns:
            dict: Registry mapping ``wf_id → state dict``.
        """
        if not self.sid:
            raise RuntimeError('No active session')

        resp = self._http.get(self._url(f'workflows/{self.sid}'))
        resp.raise_for_status()

        return resp.json()


    # --------------------------------------------------------------------------
    #
    def cancel_workflow(self, wf_id: str) -> dict:
        """
        Cancel a running workflow.

        Args:
            wf_id (str): Workflow ID to cancel.

        Returns:
            dict: ``{req_id, wf_id}``.
        """
        if not self.sid:
            raise RuntimeError('No active session')

        resp = self._http.post(self._url(f'cancel/{self.sid}/{wf_id}'))
        resp.raise_for_status()

        return resp.json()


    # --------------------------------------------------------------------------
    #
    def shutdown(self) -> dict:
        """
        Send a shutdown request to the ROSE service.

        Returns:
            dict: ``{req_id}``.
        """
        if not self.sid:
            raise RuntimeError('No active session')

        resp = self._http.post(self._url(f'shutdown/{self.sid}'))
        resp.raise_for_status()

        return resp.json()


# ------------------------------------------------------------------------------
#
class PluginRose(Plugin):
    """
    ROSE plugin for RADICAL-Edge.

    Exposes ROSE-as-a-Service workflow management via REST endpoints,
    enabling remote submission and monitoring of Active Learning workflows
    through the RADICAL-Edge bridge infrastructure.

    Standard routes inherited from Plugin:
    - POST /rose/register_session
    - POST /rose/unregister_session/{sid}
    - GET  /rose/echo/{sid}
    - GET  /rose/version
    - GET  /rose/list_sessions

    ROSE-specific routes:
    - POST /rose/submit/{sid}
    - GET  /rose/status/{sid}/{wf_id}
    - GET  /rose/workflows/{sid}
    - POST /rose/cancel/{sid}/{wf_id}
    - POST /rose/shutdown/{sid}
    """

    plugin_name   = 'rose'
    session_class = RoseSession
    client_class  = RoseClient
    version       = '0.1.0'


    # --------------------------------------------------------------------------
    #
    def __init__(self, app: FastAPI, instance_name: str = 'rose'):
        """
        Initialize the ROSE plugin, registering all routes.

        Args:
            app           (FastAPI): The FastAPI application instance.
            instance_name (str):     Plugin namespace. Defaults to 'rose'.
        """
        super().__init__(app, instance_name)

        self.add_route_post('submit/{sid}',          self.submit_workflow)
        self.add_route_get ('status/{sid}/{wf_id}',  self.get_workflow_status)
        self.add_route_get ('workflows/{sid}',        self.list_workflows)
        self.add_route_post('cancel/{sid}/{wf_id}',  self.cancel_workflow)
        self.add_route_post('shutdown/{sid}',         self.shutdown)

        self._log_routes()


    # --------------------------------------------------------------------------
    #
    async def register_session(self, request: Request) -> JSONResponse:
        """
        Register a new ROSE session, binding it to a specific job ID.

        Overrides the base implementation to accept an optional ``job_id``
        from the request body (defaults to ``'local_job_0'``).

        Args:
            request (Request): JSON body may contain ``{"job_id": "..."}``

        Returns:
            JSONResponse: ``{sid}`` — the assigned session ID.
        """
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass

        job_id = body.get('job_id', 'local_job_0')

        async with self._id_lock:
            sid = f'session.{uuid.uuid4().hex[:8]}'

        self._sessions[sid] = self._create_session(sid, job_id=job_id)
        log.info(f'[{self.instance_name}] Registered session {sid} '
                 f'(job_id={job_id})')

        return JSONResponse({'sid': sid})


    # --------------------------------------------------------------------------
    #
    async def submit_workflow(self, request: Request) -> JSONResponse:
        """
        Submit a workflow YAML file to the ROSE service.

        Args:
            request (Request): Path param ``sid``.
                               JSON body: ``{"workflow_file": "/path/to/wf.yaml"}``

        Returns:
            JSONResponse: ``{req_id, wf_id}``
        """
        sid  = request.path_params['sid']
        data = await request.json()

        return await self._forward(sid, RoseSession.submit_workflow,
                                   workflow_file=data.get('workflow_file'))


    # --------------------------------------------------------------------------
    #
    async def get_workflow_status(self, request: Request) -> JSONResponse:
        """
        Return the status of a specific workflow.

        Args:
            request (Request): Path params ``sid``, ``wf_id``.

        Returns:
            JSONResponse: Workflow state dictionary.
        """
        sid   = request.path_params['sid']
        wf_id = request.path_params['wf_id']

        return await self._forward(sid, RoseSession.get_workflow_status,
                                   wf_id=wf_id)


    # --------------------------------------------------------------------------
    #
    async def list_workflows(self, request: Request) -> JSONResponse:
        """
        List all workflows tracked by the ROSE service.

        Args:
            request (Request): Path param ``sid``.

        Returns:
            JSONResponse: Registry dict ``{wf_id → state dict}``.
        """
        sid = request.path_params['sid']

        return await self._forward(sid, RoseSession.list_workflows)


    # --------------------------------------------------------------------------
    #
    async def cancel_workflow(self, request: Request) -> JSONResponse:
        """
        Cancel a running workflow.

        Args:
            request (Request): Path params ``sid``, ``wf_id``.

        Returns:
            JSONResponse: ``{req_id, wf_id}``
        """
        sid   = request.path_params['sid']
        wf_id = request.path_params['wf_id']

        return await self._forward(sid, RoseSession.cancel_workflow,
                                   wf_id=wf_id)


    # --------------------------------------------------------------------------
    #
    async def shutdown(self, request: Request) -> JSONResponse:
        """
        Send a graceful shutdown request to the ROSE service.

        Args:
            request (Request): Path param ``sid``.

        Returns:
            JSONResponse: ``{req_id}``
        """
        sid = request.path_params['sid']

        return await self._forward(sid, RoseSession.shutdown)


# ------------------------------------------------------------------------------
