from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import time

class WorkflowState(Enum):
    """Lifecycle states for a ROSE Workflow/Learner."""
    SUBMITTED = "SUBMITTED"           # Received but not yet started
    INITIALIZING = "INITIALIZING"     # Loading config and resources
    RUNNING = "RUNNING"               # Active execution
    COMPLETED = "COMPLETED"           # Finished successfully
    FAILED = "FAILED"                 # Terminated with error
    CANCELED = "CANCELED"             # Stopped by user request

@dataclass
class Workflow:
    """Represents a managed workflow (learner) instance."""
    wf_id: str
    state: WorkflowState = WorkflowState.SUBMITTED
    workflow_file: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Internal reference to the actual Learner object
    # This is not serialized to JSON
    learner_instance: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation for external monitoring."""
        return {
            "wf_id": self.wf_id,
            "state": self.state.value,
            "workflow_file": self.workflow_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stats": self.stats,
            "error": self.error
        }
