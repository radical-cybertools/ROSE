import sys as _sys

from rose.learner import IterationState, Learner, LearnerConfig, TaskConfig
from rose.metrics import *  # noqa: F403
from rose.spec import WorkflowSpec, load_spec
from rose.tracking import PipelineManifest, TrackerBase

from . import active_learning as al
from . import reinforcement_learning as rl
from . import uncertainty_quantification as uq
from .active_learning import active_learner, selector
from .reinforcement_learning import reinforcement_learner
from .uncertainty_quantification import uq_active_learner, uq_learner, uq_scorer

_sys.modules["rose.al"] = al
_sys.modules["rose.rl"] = rl
_sys.modules["rose.uq"] = uq
_sys.modules["rose.al.active_learner"] = _sys.modules["rose.active_learning.active_learner"]
_sys.modules["rose.al.selector"] = _sys.modules["rose.active_learning.selector"]
_sys.modules["rose.rl.experience"] = _sys.modules["rose.reinforcement_learning.experience"]
_sys.modules["rose.rl.reinforcement_learner"] = _sys.modules[
    "rose.reinforcement_learning.reinforcement_learner"
]
_sys.modules["rose.uq.uq_active_learner"] = _sys.modules[
    "rose.uncertainty_quantification.uq_active_learner"
]
_sys.modules["rose.uq.uq_learner"] = _sys.modules["rose.uncertainty_quantification.uq_learner"]
_sys.modules["rose.uq.uq_scorer"] = _sys.modules["rose.uncertainty_quantification.uq_scorer"]
del _sys

__all__ = [
    # Short-name subpackage aliases
    "al",
    "rl",
    "uq",
    # Submodules
    "active_learner",
    "selector",
    "reinforcement_learner",
    "uq_learner",
    "uq_scorer",
    "uq_active_learner",
    # Classes / configs
    "Learner",
    "LearnerConfig",
    "TaskConfig",
    "IterationState",
    # Tracking
    "TrackerBase",
    "PipelineManifest",
    # YAML spec layer
    "load_spec",
    "WorkflowSpec",
]
