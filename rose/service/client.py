import json
import uuid
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class ServiceClient:
    """Client for interacting with the ROSE Service via File-based IPC.
    
    Attributes:
        job_id (str): The SLURM job ID where the service is running.
        service_root (Path): Root directory for service IPC (~/.rose/services/<job_id>).
    """

    @staticmethod
    def get_wf_id(req_id: str) -> str:
        """Derive Workflow ID from Request ID."""
        return f"wf.{req_id[:8]}"

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.service_root = Path.home() / ".rose" / "services" / str(job_id)
        self.requests_dir = self.service_root / "requests"
        self.registry_file = self.service_root / "registry.json"

        if not self.service_root.exists():
            # It's possible the service hasn't started creating dirs yet, 
            # or the job ID is wrong. We don't raise immediately to allow 
            # retry logic in scripts, but warn if needed.
            logger.warning(f"Service root {self.service_root} does not exist yet.")

    def _write_request(self, action: str, payload: Dict[str, Any]) -> str:
        """Write a request file to the requests directory."""
        req_id = str(uuid.uuid4())
        request_data = {
            "id": req_id,
            "action": action,
            "timestamp": time.time(),
            "payload": payload
        }
        
        # Ensure requests dir exists (client might start before service creates it? 
        # Better to assume service creates it, but safe to check)
        if not self.requests_dir.exists():
             raise RuntimeError(f"Service requests directory not found: {self.requests_dir}")

        req_file = self.requests_dir / f"{action}_{req_id}.json"
        with open(req_file, "w") as f:
            json.dump(request_data, f, indent=2)
        
        return req_id

    def submit_workflow(self, workflow_file: str) -> str:
        """Submit a workflow file to the service.
        
        Args:
            workflow_file (str): Path to the workflow YAML file.
            
        Returns:
            str: Request ID (not yet the wf_id, which depends on service processing).
        """
        abs_path = str(Path(workflow_file).resolve())
        return self._write_request("submit", {"workflow_file": abs_path})

    def cancel_workflow(self, wf_id: str) -> str:
        """Request cancellation of a workflow.
        
        Args:
            wf_id (str): The workflow ID to cancel.
        """
        return self._write_request("cancel", {"wf_id": wf_id})

    def shutdown(self) -> str:
        """Request graceful shutdown of the service."""
        return self._write_request("shutdown", {})

    def get_workflow_status(self, wf_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow from the registry.
        
        Args:
            wf_id (str): Workflow ID.
            
        Returns:
            dict: Workflow state dict or None if not found.
        """
        registry = self._read_registry()
        return registry.get(wf_id)
    
    def list_workflows(self) -> Dict[str, Any]:
        """List all workflows in the registry."""
        return self._read_registry()

    def _read_registry(self) -> Dict[str, Any]:
        """Read and parse the registry file."""
        if not self.registry_file.exists():
            return {}
        
        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Race condition on read or empty file
            return {}
