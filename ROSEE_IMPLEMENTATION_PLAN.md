# ROSEE Implementation Plan

## Overview

ROSEE is an agent-enablement layer on top of ROSE that:
1. Toolifies ROSE capabilities (Python functions + MCP protocol)
2. Provides standardized workflow state for agent consumption
3. Enables per-iteration agent steering with timeout/default fallback

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User's Decision Function                     │
│   decision_fn(state: WorkflowState) -> Action                   │
│   (LLM with tools, RL policy, bandit, or custom logic)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ActiveLearningAgent                            │
│   • Wraps ROSE learner                                          │
│   • Observe → Decide → Apply → Execute loop                     │
│   • Timeout with default fallback                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ROSEE Tools                               │
│   • Python functions + MCP protocol                             │
│   • select_samples, set_hyperparams, get_uncertainty, stop      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ROSE Layer                                   │
│   SequentialActiveLearner, ParallelActiveLearner, etc.          │
│   HPC execution via RADICAL-AsyncFlow                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
rosee/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py                 # BaseAgent with observe/apply/execute
│   ├── active_learning.py      # ActiveLearningAgent
│   └── reinforcement.py        # ReinforcementLearningAgent
├── state/
│   ├── __init__.py
│   ├── workflow_state.py       # WorkflowState dataclass
│   └── actions.py              # Action, ActionType definitions
├── tools/
│   ├── __init__.py
│   ├── base.py                 # BaseTool class
│   ├── sample_selection.py     # SelectSamplesTool
│   ├── hyperparameters.py      # SetHyperparametersTool
│   ├── uncertainty.py          # GetUncertaintyTool
│   ├── control.py              # StopTool, ContinueTool
│   └── registry.py             # ToolRegistry for discovery
├── mcp/
│   ├── __init__.py
│   ├── server.py               # MCP server exposing tools
│   └── schemas.py              # MCP tool schemas
├── defaults/
│   ├── __init__.py
│   └── policies.py             # Default fallback policies
└── utils/
    ├── __init__.py
    └── timeout.py              # Timeout utilities
```

---

## Phase 1: Core State and Actions

### 1.1 WorkflowState (`rosee/state/workflow_state.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class WorkflowState:
    """Standardized state for agent consumption."""

    # Identity
    workflow_id: str
    learner_type: str  # 'sequential', 'parallel', 'algorithm_selector'

    # Progress
    iteration: int
    max_iterations: Optional[int]

    # Primary metric
    metric_name: str
    metric_value: float
    metric_threshold: Optional[float]
    metric_history: List[float] = field(default_factory=list)

    # Data state
    labeled_count: int = 0
    unlabeled_count: int = 0
    samples_per_iteration: List[int] = field(default_factory=list)

    # Uncertainty (if available)
    uncertainty_scores: Optional[np.ndarray] = None
    mean_uncertainty: Optional[float] = None

    # Current configuration
    current_config: Dict = field(default_factory=dict)

    # Resource usage
    compute_used: float = 0.0  # Normalized units
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'workflow_id': self.workflow_id,
            'learner_type': self.learner_type,
            'iteration': self.iteration,
            'max_iterations': self.max_iterations,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_threshold': self.metric_threshold,
            'metric_history': self.metric_history,
            'labeled_count': self.labeled_count,
            'unlabeled_count': self.unlabeled_count,
            'mean_uncertainty': self.mean_uncertainty,
            'current_config': self.current_config,
            'compute_used': self.compute_used,
            'elapsed_seconds': self.elapsed_seconds,
        }

    def to_prompt(self) -> str:
        """Convert to natural language for LLM agents."""
        lines = [
            f"## Workflow State (Iteration {self.iteration})",
            f"",
            f"**Progress:**",
            f"- Iteration: {self.iteration}" + (f" / {self.max_iterations}" if self.max_iterations else ""),
            f"- Metric ({self.metric_name}): {self.metric_value:.6f}" + (f" (target: {self.metric_threshold})" if self.metric_threshold else ""),
            f"- Metric history: {[f'{v:.4f}' for v in self.metric_history[-5:]]}",
            f"",
            f"**Data:**",
            f"- Labeled samples: {self.labeled_count}",
            f"- Unlabeled samples: {self.unlabeled_count}",
        ]

        if self.mean_uncertainty is not None:
            lines.append(f"- Mean uncertainty: {self.mean_uncertainty:.4f}")

        lines.extend([
            f"",
            f"**Resources:**",
            f"- Compute used: {self.compute_used:.2f} units",
            f"- Elapsed time: {self.elapsed_seconds:.1f}s",
        ])

        if self.current_config:
            lines.extend([
                f"",
                f"**Current config:**",
            ])
            for k, v in self.current_config.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    def to_vector(self) -> np.ndarray:
        """Convert to numeric vector for RL agents."""
        # Normalize and concatenate numeric features
        features = [
            self.iteration / (self.max_iterations or 100),
            self.metric_value,
            self.labeled_count / max(self.labeled_count + self.unlabeled_count, 1),
            self.mean_uncertainty or 0.0,
            self.compute_used / 1000,  # Normalize
        ]
        # Add recent metric history (padded)
        history = self.metric_history[-5:] + [0.0] * (5 - len(self.metric_history[-5:]))
        features.extend(history)

        return np.array(features, dtype=np.float32)
```

### 1.2 Actions (`rosee/state/actions.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(Enum):
    """Types of actions an agent can take."""
    SELECT_SAMPLES = "select_samples"
    SET_HYPERPARAMETERS = "set_hyperparameters"
    GET_UNCERTAINTY = "get_uncertainty"
    CONTINUE = "continue"
    STOP = "stop"


@dataclass
class Action:
    """Action to be applied to workflow."""

    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def select_samples(
        cls,
        strategy: str = "uncertainty",
        count: int = 10,
        indices: Optional[List[int]] = None
    ) -> "Action":
        """Create sample selection action."""
        return cls(
            action_type=ActionType.SELECT_SAMPLES,
            parameters={
                "strategy": strategy,
                "count": count,
                "indices": indices
            }
        )

    @classmethod
    def set_hyperparameters(cls, **kwargs) -> "Action":
        """Create hyperparameter setting action."""
        return cls(
            action_type=ActionType.SET_HYPERPARAMETERS,
            parameters=kwargs
        )

    @classmethod
    def get_uncertainty(cls, metric: str = "predictive_entropy") -> "Action":
        """Create uncertainty query action."""
        return cls(
            action_type=ActionType.GET_UNCERTAINTY,
            parameters={"metric": metric}
        )

    @classmethod
    def continue_iteration(cls) -> "Action":
        """Create continue action (use defaults)."""
        return cls(action_type=ActionType.CONTINUE)

    @classmethod
    def stop(cls, reason: str = "agent_decision") -> "Action":
        """Create stop action."""
        return cls(
            action_type=ActionType.STOP,
            parameters={"reason": reason}
        )


@dataclass
class ActionResult:
    """Result of applying an action."""

    success: bool
    action_type: ActionType
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
```

---

## Phase 2: Tools (Python Functions)

### 2.1 Base Tool (`rosee/tools/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..state.actions import Action, ActionResult
from ..state.workflow_state import WorkflowState


@dataclass
class ToolSpec:
    """Specification for a tool (for LLM function calling)."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    required: list[str]


class BaseTool(ABC):
    """Base class for ROSEE tools."""

    name: str
    description: str

    @abstractmethod
    def get_spec(self) -> ToolSpec:
        """Get tool specification for LLM."""
        pass

    @abstractmethod
    def execute(self, state: WorkflowState, **kwargs) -> ActionResult:
        """Execute tool and return result."""
        pass

    def to_action(self, **kwargs) -> Action:
        """Convert tool call to Action."""
        pass
```

### 2.2 Sample Selection Tool (`rosee/tools/sample_selection.py`)

```python
from typing import List, Optional

from .base import BaseTool, ToolSpec
from ..state.actions import Action, ActionResult, ActionType
from ..state.workflow_state import WorkflowState


class SelectSamplesTool(BaseTool):
    """Tool for selecting samples in active learning."""

    name = "select_samples"
    description = """Select samples from the unlabeled pool for the next iteration.

    Strategies:
    - 'uncertainty': Select most uncertain samples (exploitation)
    - 'diversity': Select most diverse samples (exploration)
    - 'random': Random selection (baseline)
    - 'hybrid': Balance uncertainty and diversity

    You can also provide explicit indices if you have specific samples in mind."""

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["uncertainty", "diversity", "random", "hybrid"],
                        "description": "Sample selection strategy"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of samples to select",
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Explicit indices to select (overrides strategy)"
                    }
                }
            },
            required=["strategy", "count"]
        )

    def execute(
        self,
        state: WorkflowState,
        strategy: str = "uncertainty",
        count: int = 10,
        indices: Optional[List[int]] = None
    ) -> ActionResult:
        """Execute sample selection."""
        # Validation
        if indices is not None:
            if any(i >= state.unlabeled_count for i in indices):
                return ActionResult(
                    success=False,
                    action_type=ActionType.SELECT_SAMPLES,
                    error=f"Invalid indices: max index is {state.unlabeled_count - 1}"
                )
            selected = indices
        else:
            if count > state.unlabeled_count:
                count = state.unlabeled_count
            selected = None  # Will be computed during apply

        return ActionResult(
            success=True,
            action_type=ActionType.SELECT_SAMPLES,
            data={
                "strategy": strategy,
                "count": count,
                "indices": selected
            }
        )

    def to_action(
        self,
        strategy: str = "uncertainty",
        count: int = 10,
        indices: Optional[List[int]] = None
    ) -> Action:
        return Action.select_samples(strategy=strategy, count=count, indices=indices)
```

### 2.3 Hyperparameters Tool (`rosee/tools/hyperparameters.py`)

```python
from typing import Any, Dict, Optional

from .base import BaseTool, ToolSpec
from ..state.actions import Action, ActionResult, ActionType
from ..state.workflow_state import WorkflowState


class SetHyperparametersTool(BaseTool):
    """Tool for setting training hyperparameters."""

    name = "set_hyperparameters"
    description = """Set hyperparameters for the next training iteration.

    Common parameters:
    - learning_rate: Learning rate for optimizer (e.g., 0.001)
    - batch_size: Training batch size (e.g., 32, 64, 128)
    - epochs: Number of training epochs (e.g., 10, 50, 100)

    Only set parameters you want to change; others keep their current values."""

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate",
                        "minimum": 1e-7,
                        "maximum": 1.0
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size",
                        "minimum": 1,
                        "maximum": 4096
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Number of epochs",
                        "minimum": 1,
                        "maximum": 1000
                    }
                },
                "additionalProperties": True  # Allow custom params
            },
            required=[]
        )

    def execute(
        self,
        state: WorkflowState,
        **kwargs
    ) -> ActionResult:
        """Execute hyperparameter setting."""
        return ActionResult(
            success=True,
            action_type=ActionType.SET_HYPERPARAMETERS,
            data=kwargs
        )

    def to_action(self, **kwargs) -> Action:
        return Action.set_hyperparameters(**kwargs)
```

### 2.4 Uncertainty Tool (`rosee/tools/uncertainty.py`)

```python
from typing import Dict

from .base import BaseTool, ToolSpec
from ..state.actions import Action, ActionResult, ActionType
from ..state.workflow_state import WorkflowState


class GetUncertaintyTool(BaseTool):
    """Tool for querying uncertainty metrics."""

    name = "get_uncertainty"
    description = """Get uncertainty information about the current model and unlabeled pool.

    Available metrics:
    - 'predictive_entropy': Entropy of predicted probabilities
    - 'mutual_information': Information between predictions and model params
    - 'predictive_variance': Variance of predictions (regression)
    - 'margin': Difference between top two class probabilities

    Returns statistics about uncertainty distribution."""

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "predictive_entropy",
                            "mutual_information",
                            "predictive_variance",
                            "margin"
                        ],
                        "description": "Uncertainty metric to compute"
                    }
                }
            },
            required=["metric"]
        )

    def execute(
        self,
        state: WorkflowState,
        metric: str = "predictive_entropy"
    ) -> ActionResult:
        """Get uncertainty statistics from state."""
        if state.uncertainty_scores is None:
            return ActionResult(
                success=False,
                action_type=ActionType.GET_UNCERTAINTY,
                error="No uncertainty scores available"
            )

        import numpy as np
        scores = state.uncertainty_scores

        return ActionResult(
            success=True,
            action_type=ActionType.GET_UNCERTAINTY,
            data={
                "metric": metric,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "percentiles": {
                    "25": float(np.percentile(scores, 25)),
                    "50": float(np.percentile(scores, 50)),
                    "75": float(np.percentile(scores, 75)),
                    "90": float(np.percentile(scores, 90))
                }
            }
        )

    def to_action(self, metric: str = "predictive_entropy") -> Action:
        return Action.get_uncertainty(metric=metric)
```

### 2.5 Control Tools (`rosee/tools/control.py`)

```python
from .base import BaseTool, ToolSpec
from ..state.actions import Action, ActionResult, ActionType
from ..state.workflow_state import WorkflowState


class StopTool(BaseTool):
    """Tool to stop the workflow."""

    name = "stop"
    description = """Stop the active learning workflow.

    Use when:
    - Target metric has been reached
    - Metric is no longer improving (converged)
    - Budget (compute/samples) is exhausted
    - Further training would not be beneficial"""

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for stopping"
                    }
                }
            },
            required=["reason"]
        )

    def execute(self, state: WorkflowState, reason: str = "") -> ActionResult:
        return ActionResult(
            success=True,
            action_type=ActionType.STOP,
            data={"reason": reason}
        )

    def to_action(self, reason: str = "agent_decision") -> Action:
        return Action.stop(reason=reason)


class ContinueTool(BaseTool):
    """Tool to continue with default settings."""

    name = "continue"
    description = """Continue to the next iteration with default settings.

    Use when current configuration is working well and no changes needed."""

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}},
            required=[]
        )

    def execute(self, state: WorkflowState) -> ActionResult:
        return ActionResult(
            success=True,
            action_type=ActionType.CONTINUE,
            data={}
        )

    def to_action(self) -> Action:
        return Action.continue_iteration()
```

### 2.6 Tool Registry (`rosee/tools/registry.py`)

```python
from typing import Dict, List

from .base import BaseTool, ToolSpec
from .sample_selection import SelectSamplesTool
from .hyperparameters import SetHyperparametersTool
from .uncertainty import GetUncertaintyTool
from .control import StopTool, ContinueTool


class ToolRegistry:
    """Registry of available ROSEE tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default ROSEE tools."""
        default_tools = [
            SelectSamplesTool(),
            SetHyperparametersTool(),
            GetUncertaintyTool(),
            StopTool(),
            ContinueTool()
        ]
        for tool in default_tools:
            self.register(tool)

    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_specs(self) -> List[ToolSpec]:
        """Get all tool specifications (for LLM)."""
        return [tool.get_spec() for tool in self._tools.values()]

    def to_openai_format(self) -> List[Dict]:
        """Convert to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters
                }
            }
            for spec in self.get_specs()
        ]

    def to_anthropic_format(self) -> List[Dict]:
        """Convert to Anthropic tool use format."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.parameters
            }
            for spec in self.get_specs()
        ]
```

---

## Phase 3: Active Learning Agent

### 3.1 Base Agent (`rosee/agents/base.py`)

```python
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ..state.workflow_state import WorkflowState
from ..state.actions import Action, ActionType
from ..tools.registry import ToolRegistry
from ..defaults.policies import DefaultPolicy


class BaseAgent(ABC):
    """Base class for ROSEE agents."""

    def __init__(
        self,
        learner: Any,  # ROSE learner
        decision_fn: Callable[[WorkflowState], Action],
        decision_timeout: float = 30.0,
        default_policy: Optional[DefaultPolicy] = None
    ):
        self.learner = learner
        self.decide = decision_fn
        self.decision_timeout = decision_timeout
        self.default_policy = default_policy or DefaultPolicy()
        self.tools = ToolRegistry()

        # State tracking
        self._iteration = 0
        self._metric_history = []
        self._start_time = None

    @abstractmethod
    def _observe(self) -> WorkflowState:
        """Gather current state from learner."""
        pass

    @abstractmethod
    def _apply(self, action: Action) -> None:
        """Apply action to configure learner."""
        pass

    @abstractmethod
    async def _execute_iteration(self) -> Any:
        """Execute one iteration via ROSE."""
        pass

    async def _decide_with_timeout(self, state: WorkflowState) -> Action:
        """Call decision function with timeout and fallback."""
        try:
            # Handle both sync and async decision functions
            if asyncio.iscoroutinefunction(self.decide):
                action = await asyncio.wait_for(
                    self.decide(state),
                    timeout=self.decision_timeout
                )
            else:
                action = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.decide, state
                    ),
                    timeout=self.decision_timeout
                )
            return action
        except asyncio.TimeoutError:
            print(f"Decision timeout ({self.decision_timeout}s), using default policy")
            return self.default_policy.decide(state)
        except Exception as e:
            print(f"Decision error: {e}, using default policy")
            return self.default_policy.decide(state)

    @abstractmethod
    async def run(self, max_iter: int) -> Any:
        """Run the agent-controlled workflow."""
        pass
```

### 3.2 Active Learning Agent (`rosee/agents/active_learning.py`)

```python
import time
from typing import Any, Callable, Optional

from rose.al import SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig

from .base import BaseAgent
from ..state.workflow_state import WorkflowState
from ..state.actions import Action, ActionType
from ..defaults.policies import DefaultPolicy


class ActiveLearningAgent(BaseAgent):
    """Agent that controls a ROSE SequentialActiveLearner."""

    def __init__(
        self,
        learner: SequentialActiveLearner,
        decision_fn: Callable[[WorkflowState], Action],
        decision_timeout: float = 30.0,
        default_policy: Optional[DefaultPolicy] = None
    ):
        super().__init__(learner, decision_fn, decision_timeout, default_policy)

        # AL-specific state
        self._current_config = LearnerConfig()
        self._labeled_count = 0
        self._unlabeled_count = 0
        self._uncertainty_scores = None
        self._pending_action: Optional[Action] = None

    def _observe(self) -> WorkflowState:
        """Gather current state from the ROSE learner."""
        # Get metric info from learner
        metric_history = list(self.learner.metric_values_per_iteration.values())
        current_metric = metric_history[-1] if metric_history else float('inf')

        # Get criterion info if available
        metric_name = "unknown"
        threshold = None
        if self.learner.criterion_function:
            metric_name = self.learner.criterion_function.get("metric_name", "unknown")
            threshold = self.learner.criterion_function.get("threshold")

        # Build state
        state = WorkflowState(
            workflow_id=str(id(self.learner)),
            learner_type="sequential",
            iteration=self._iteration,
            max_iterations=None,  # Set during run()
            metric_name=metric_name,
            metric_value=current_metric,
            metric_threshold=threshold,
            metric_history=metric_history,
            labeled_count=self._labeled_count,
            unlabeled_count=self._unlabeled_count,
            uncertainty_scores=self._uncertainty_scores,
            mean_uncertainty=(
                float(self._uncertainty_scores.mean())
                if self._uncertainty_scores is not None
                else None
            ),
            current_config=self._config_to_dict(),
            compute_used=0.0,  # TODO: Track from RADICAL
            elapsed_seconds=time.time() - self._start_time if self._start_time else 0.0
        )

        return state

    def _config_to_dict(self) -> dict:
        """Convert current LearnerConfig to dict."""
        config = {}
        if self._current_config.training:
            if isinstance(self._current_config.training, TaskConfig):
                config.update(self._current_config.training.kwargs)
        return config

    def _apply(self, action: Action) -> None:
        """Apply action to configure the learner for next iteration."""
        if action.action_type == ActionType.SELECT_SAMPLES:
            # Store selection params for next iteration
            self._pending_action = action

        elif action.action_type == ActionType.SET_HYPERPARAMETERS:
            # Update training config
            params = action.parameters
            current_kwargs = {}
            if self._current_config.training:
                if isinstance(self._current_config.training, TaskConfig):
                    current_kwargs = self._current_config.training.kwargs.copy()

            # Map common names to CLI args
            if "learning_rate" in params:
                current_kwargs["--lr"] = str(params["learning_rate"])
            if "batch_size" in params:
                current_kwargs["--batch_size"] = str(params["batch_size"])
            if "epochs" in params:
                current_kwargs["--epochs"] = str(params["epochs"])

            # Add any additional params
            for k, v in params.items():
                if k not in ["learning_rate", "batch_size", "epochs"]:
                    current_kwargs[f"--{k}"] = str(v)

            self._current_config.training = TaskConfig(kwargs=current_kwargs)

        elif action.action_type == ActionType.CONTINUE:
            # Keep current config
            pass

        elif action.action_type == ActionType.STOP:
            # Handled in run loop
            pass

    async def _execute_iteration(self) -> Any:
        """Execute one AL iteration via ROSE learner."""
        # Build iteration-specific config
        iter_config = LearnerConfig(
            simulation=self._current_config.simulation,
            training=self._current_config.training,
            active_learn=self._current_config.active_learn,
            criterion=self._current_config.criterion
        )

        # If we have a pending sample selection action, encode it
        if self._pending_action and self._pending_action.action_type == ActionType.SELECT_SAMPLES:
            params = self._pending_action.parameters
            al_kwargs = {
                "--strategy": params.get("strategy", "uncertainty"),
                "--count": str(params.get("count", 10))
            }
            if params.get("indices"):
                al_kwargs["--indices"] = ",".join(map(str, params["indices"]))

            iter_config.active_learn = TaskConfig(kwargs=al_kwargs)
            self._pending_action = None

        # Execute via ROSE
        # Note: This is simplified - actual implementation needs to handle
        # the internal teach() loop structure
        result = await self._run_single_iteration(iter_config)

        self._iteration += 1
        return result

    async def _run_single_iteration(self, config: LearnerConfig) -> Any:
        """Run a single iteration with given config."""
        # This hooks into ROSE's internal iteration mechanism
        # Implementation depends on how we modify ROSE or wrap it

        # For now, use the teach() with max_iter=1
        # In practice, we'd need finer control or a modified ROSE API
        pass

    async def run(self, max_iter: int) -> dict:
        """Run the agent-controlled active learning workflow."""
        self._start_time = time.time()
        results = {
            "iterations": 0,
            "final_metric": None,
            "metric_history": [],
            "stop_reason": None
        }

        for i in range(max_iter):
            self._iteration = i

            # 1. Observe current state
            state = self._observe()
            state.max_iterations = max_iter

            # 2. Agent makes decision (with timeout)
            action = await self._decide_with_timeout(state)

            # 3. Check for stop action
            if action.action_type == ActionType.STOP:
                results["stop_reason"] = action.parameters.get("reason", "agent_decision")
                break

            # 4. Apply action to configure learner
            self._apply(action)

            # 5. Execute iteration via ROSE
            iter_result = await self._execute_iteration()

            # 6. Update results
            results["iterations"] = i + 1
            results["metric_history"] = list(
                self.learner.metric_values_per_iteration.values()
            )
            if results["metric_history"]:
                results["final_metric"] = results["metric_history"][-1]

        if results["stop_reason"] is None:
            results["stop_reason"] = "max_iterations"

        return results
```

---

## Phase 4: Default Policies

### 4.1 Default Policy (`rosee/defaults/policies.py`)

```python
from typing import Optional
import numpy as np

from ..state.workflow_state import WorkflowState
from ..state.actions import Action


class DefaultPolicy:
    """Default policy used when agent times out or errors."""

    def __init__(
        self,
        default_strategy: str = "uncertainty",
        default_sample_count: int = 10,
        convergence_threshold: float = 0.001,
        patience: int = 5
    ):
        self.default_strategy = default_strategy
        self.default_sample_count = default_sample_count
        self.convergence_threshold = convergence_threshold
        self.patience = patience

    def decide(self, state: WorkflowState) -> Action:
        """Make a default decision based on state."""

        # Check for convergence (no improvement for `patience` iterations)
        if len(state.metric_history) >= self.patience:
            recent = state.metric_history[-self.patience:]
            improvement = abs(recent[0] - recent[-1])
            if improvement < self.convergence_threshold:
                return Action.stop(reason="converged (default policy)")

        # Check if threshold reached
        if state.metric_threshold is not None:
            if state.metric_value <= state.metric_threshold:
                return Action.stop(reason="threshold reached (default policy)")

        # Default: continue with uncertainty sampling
        return Action.select_samples(
            strategy=self.default_strategy,
            count=self.default_sample_count
        )


class AdaptiveDefaultPolicy(DefaultPolicy):
    """Adaptive default policy that adjusts based on progress."""

    def decide(self, state: WorkflowState) -> Action:
        """Make adaptive default decision."""

        # First check stop conditions
        base_action = super().decide(state)
        if base_action.action_type.value == "stop":
            return base_action

        # Adapt sample count based on uncertainty
        if state.mean_uncertainty is not None:
            # High uncertainty -> more samples
            if state.mean_uncertainty > 0.7:
                count = min(self.default_sample_count * 2, state.unlabeled_count)
            elif state.mean_uncertainty < 0.3:
                count = max(self.default_sample_count // 2, 1)
            else:
                count = self.default_sample_count
        else:
            count = self.default_sample_count

        # Adapt strategy based on iteration
        if state.iteration < 5:
            strategy = "diversity"  # Explore early
        elif state.iteration > 20:
            strategy = "uncertainty"  # Exploit late
        else:
            strategy = "hybrid"  # Balance in middle

        return Action.select_samples(strategy=strategy, count=count)
```

---

## Phase 5: MCP Server

### 5.1 MCP Server (`rosee/mcp/server.py`)

```python
import json
from typing import Any, Dict, List

from ..tools.registry import ToolRegistry
from ..state.workflow_state import WorkflowState


class ROSEEMCPServer:
    """MCP server exposing ROSEE tools."""

    def __init__(self, tool_registry: ToolRegistry):
        self.tools = tool_registry
        self._current_state: WorkflowState = None

    def set_state(self, state: WorkflowState):
        """Update current workflow state."""
        self._current_state = state

    def get_tool_list(self) -> List[Dict]:
        """Get list of available tools in MCP format."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.parameters
            }
            for spec in self.tools.get_specs()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        tool = self.tools.get(name)
        if tool is None:
            return {
                "error": f"Unknown tool: {name}",
                "available_tools": [t.name for t in self.tools.list_tools()]
            }

        if self._current_state is None:
            return {"error": "No workflow state available"}

        result = tool.execute(self._current_state, **arguments)

        return {
            "success": result.success,
            "action_type": result.action_type.value,
            "data": result.data,
            "error": result.error
        }

    def handle_request(self, request: Dict) -> Dict:
        """Handle MCP request."""
        method = request.get("method")

        if method == "tools/list":
            return {
                "tools": self.get_tool_list()
            }

        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(self.call_tool(name, arguments))
                    }
                ]
            }

        elif method == "resources/read":
            # Return current state as a resource
            if self._current_state:
                return {
                    "contents": [
                        {
                            "uri": "rosee://workflow/state",
                            "mimeType": "application/json",
                            "text": json.dumps(self._current_state.to_dict())
                        }
                    ]
                }
            return {"contents": []}

        return {"error": f"Unknown method: {method}"}
```

### 5.2 MCP Schemas (`rosee/mcp/schemas.py`)

```python
"""MCP protocol schemas for ROSEE."""

SERVER_INFO = {
    "name": "rosee",
    "version": "0.1.0",
    "description": "ROSEE - Agent-enabled interface for ROSE workflows"
}

CAPABILITIES = {
    "tools": {},
    "resources": {
        "subscribe": False,
        "listChanged": False
    }
}

RESOURCE_TEMPLATES = [
    {
        "uriTemplate": "rosee://workflow/state",
        "name": "Workflow State",
        "description": "Current state of the active learning workflow",
        "mimeType": "application/json"
    }
]
```

---

## Phase 6: Usage Examples

### 6.1 Basic LLM-Controlled AL

```python
import asyncio
from anthropic import Anthropic

from rose.al import SequentialActiveLearner
from radical.asyncflow import WorkflowEngine, RadicalExecutionBackend

from rosee.agents import ActiveLearningAgent
from rosee.state import WorkflowState, Action
from rosee.tools import ToolRegistry


async def main():
    # Setup ROSE
    engine = await RadicalExecutionBackend({'resource': 'local.localhost', 'runtime': 30})
    asyncflow = await WorkflowEngine.create(engine)
    learner = SequentialActiveLearner(asyncflow)

    # Define ROSE tasks (as usual)
    @learner.simulation_task
    async def simulation(*args, **kwargs):
        return 'python sim.py'

    @learner.training_task
    async def training(*args, **kwargs):
        return 'python train.py'

    @learner.active_learn_task
    async def active_learn(*args, **kwargs):
        return 'python active.py'

    # Create LLM decision function
    client = Anthropic()
    tools = ToolRegistry()

    def llm_decision(state: WorkflowState) -> Action:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You control an active learning workflow. Use the tools to decide what to do next.",
            messages=[
                {"role": "user", "content": state.to_prompt()}
            ],
            tools=tools.to_anthropic_format()
        )

        # Parse tool call from response
        for block in response.content:
            if block.type == "tool_use":
                tool = tools.get(block.name)
                return tool.to_action(**block.input)

        # No tool called - default to continue
        return Action.continue_iteration()

    # Create ROSEE agent
    agent = ActiveLearningAgent(
        learner=learner,
        decision_fn=llm_decision,
        decision_timeout=60.0  # 60 second timeout
    )

    # Run
    result = await agent.run(max_iter=20)
    print(f"Completed: {result}")

    await asyncflow.shutdown()


asyncio.run(main())
```

### 6.2 RL Policy Decision Function

```python
import torch
import torch.nn as nn

from rosee.state import WorkflowState, Action


class RLPolicy(nn.Module):
    """Simple RL policy network."""

    def __init__(self, state_dim: int = 10, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, state_vector):
        return self.net(state_vector)


def create_rl_decision_fn(policy: RLPolicy) -> callable:
    """Create decision function from RL policy."""

    action_map = {
        0: lambda: Action.select_samples(strategy="uncertainty", count=10),
        1: lambda: Action.select_samples(strategy="diversity", count=10),
        2: lambda: Action.select_samples(strategy="hybrid", count=20),
        3: lambda: Action.stop(reason="rl_policy_decision")
    }

    def decision_fn(state: WorkflowState) -> Action:
        state_vector = torch.tensor(state.to_vector()).unsqueeze(0)
        with torch.no_grad():
            logits = policy(state_vector)
            action_idx = logits.argmax(dim=1).item()
        return action_map[action_idx]()

    return decision_fn


# Usage
policy = RLPolicy()
policy.load_state_dict(torch.load("trained_policy.pt"))

agent = ActiveLearningAgent(
    learner=learner,
    decision_fn=create_rl_decision_fn(policy)
)
```

### 6.3 Simple Rule-Based Decision Function

```python
from rosee.state import WorkflowState, Action


def rule_based_decision(state: WorkflowState) -> Action:
    """Simple rule-based decision function."""

    # Stop if converged
    if len(state.metric_history) >= 3:
        recent_improvement = abs(state.metric_history[-1] - state.metric_history[-3])
        if recent_improvement < 0.001:
            return Action.stop(reason="converged")

    # Stop if target reached
    if state.metric_threshold and state.metric_value <= state.metric_threshold:
        return Action.stop(reason="target_reached")

    # Early iterations: explore with diversity
    if state.iteration < 5:
        return Action.select_samples(strategy="diversity", count=20)

    # High uncertainty: exploit
    if state.mean_uncertainty and state.mean_uncertainty > 0.5:
        return Action.select_samples(strategy="uncertainty", count=15)

    # Default: hybrid
    return Action.select_samples(strategy="hybrid", count=10)


agent = ActiveLearningAgent(
    learner=learner,
    decision_fn=rule_based_decision
)
```

---

## Implementation Order

### Sprint 1: Foundation
1. `rosee/state/workflow_state.py` - WorkflowState dataclass
2. `rosee/state/actions.py` - Action and ActionType
3. `rosee/defaults/policies.py` - DefaultPolicy

### Sprint 2: Tools
4. `rosee/tools/base.py` - BaseTool class
5. `rosee/tools/sample_selection.py` - SelectSamplesTool
6. `rosee/tools/hyperparameters.py` - SetHyperparametersTool
7. `rosee/tools/uncertainty.py` - GetUncertaintyTool
8. `rosee/tools/control.py` - StopTool, ContinueTool
9. `rosee/tools/registry.py` - ToolRegistry

### Sprint 3: Agent
10. `rosee/agents/base.py` - BaseAgent
11. `rosee/agents/active_learning.py` - ActiveLearningAgent
12. Integration testing with ROSE

### Sprint 4: MCP
13. `rosee/mcp/schemas.py` - MCP schemas
14. `rosee/mcp/server.py` - ROSEEMCPServer
15. End-to-end testing

### Sprint 5: Examples & Docs
16. Example: LLM-controlled AL
17. Example: RL policy
18. Example: Rule-based
19. Documentation

---

## Open Questions

1. **ROSE modification**: Does ROSE need a new API for single-iteration execution, or can we wrap the existing `teach()` loop?

2. **State access**: How do we access uncertainty scores and data counts from ROSE learner internals?

3. **Resource tracking**: How do we get compute usage metrics from RADICAL-Pilot?

4. **Parallel agent**: Should `ParallelActiveLearningAgent` allow per-learner decision functions?
