from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Task definition ───────────────────────────────────────────────────────────
class TaskDef(BaseModel):
    type: Literal["shell", "python"]
    command: str | None = None  # required when type=="shell"
    function: str | None = None  # required when type=="python"; "module:callable"
    task_description: dict[str, Any] | None = None  # resource hints forwarded to asyncflow backend

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_fields(self) -> TaskDef:
        if self.type == "shell" and not self.command:
            raise ValueError("shell task requires 'command'")
        if self.type == "python":
            if not self.function:
                raise ValueError("python task requires 'function'")
            if ":" not in self.function:
                raise ValueError("'function' must use 'module:callable' syntax")
        return self


# ── Stop criterion ────────────────────────────────────────────────────────────
class StopCriterionDef(BaseModel):
    metric: str
    threshold: float
    operator: Literal["<", ">", "==", "<=", ">="] = "<"
    evaluator: TaskDef

    model_config = {"extra": "forbid"}


# ── Learner spec (YAML key: "learner") ───────────────────────────────────────
_REQUIRED_SLOTS: dict[str, frozenset[str]] = {
    "sequential_active_learner": frozenset({"simulation", "training", "active_learn"}),
    "parallel_active_learner": frozenset({"simulation", "training", "active_learn"}),
    "sequential_reinforcement_learner": frozenset({"environment", "update"}),
    "uq_active_learner": frozenset(
        {"simulation", "training", "prediction", "active_learn", "uncertainty"}
    ),
}
_ALL_SLOTS: frozenset[str] = frozenset().union(*_REQUIRED_SLOTS.values())


class LearnerSpec(BaseModel):
    type: str
    max_iter: int = 0
    parallel_learners: int = 2  # used only when learners: is absent for parallel types

    model_config = {"extra": "forbid"}


# ── Per-learner definition (parallel learners only) ───────────────────────────
class LearnerDef(BaseModel):
    label: str = ""
    simulation: TaskDef | None = None
    training: TaskDef | None = None
    active_learn: TaskDef | None = None
    environment: TaskDef | None = None
    update: TaskDef | None = None
    prediction: TaskDef | None = None
    uncertainty: TaskDef | None = None

    model_config = {"extra": "forbid"}

    @property
    def tasks(self) -> dict[str, TaskDef]:
        return {s: getattr(self, s) for s in _ALL_SLOTS if getattr(self, s) is not None}


# ── Remote / tracking ────────────────────────────────────────────────────────
class TargetConfig(BaseModel):
    """Bootstrap config for `rose run --remote`: how to launch the remote ORBIT endpoint before the
    workflow's tasks run on it.

    ``kind`` selects the launch mechanism:
    - ``iri``/``sfapi``: submit a job to `resource_id` through the broker's
      ``iri_connect``/``sfapi_connect`` plugin (NERSC currently should use
      ``sfapi`` — IRI's ``/compute/*`` routes are broken server-side there;
      OLCF has no such issue and stays on ``iri``).
    - ``psij``: submit a job via PsiJ. Normally on an already-connected
      login-node endpoint (``edge_name``); with ``remote.embedded: true``,
      ``edge_name`` is not needed — PsiJ runs on the embedded broker itself
      (see ``RemoteConfig.embedded``).
    """

    kind: Literal["iri", "sfapi", "psij"]

    # iri / sfapi
    endpoint: str | None = None  # 'nersc' | 'olcf'
    resource_id: str | None = None
    home_dir: str | None = None  # user $HOME on target; resolves the wrapper path
    login_host: str | None = None  # for tunnel='forward'

    # psij
    edge_name: str | None = None  # login-node endpoint already in the topology
    executor: str | None = None

    # shared submission attributes
    account: str | None = None
    queue_name: str | None = None
    walltime_min: int = 30
    n_nodes: int = 1
    constraint: str | None = None
    reservation: str | None = None
    workdir: str | None = None
    environment: dict[str, str] = {}
    setup: list[str] = []
    tunnel: Literal["none", "forward", "reverse"] = "none"

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_fields(self) -> TargetConfig:
        if self.kind in ("iri", "sfapi"):
            if not self.endpoint:
                raise ValueError(f"remote.target.kind={self.kind!r} requires 'endpoint'")
            if not self.resource_id:
                raise ValueError(f"remote.target.kind={self.kind!r} requires 'resource_id'")
            if not self.home_dir:
                raise ValueError(f"remote.target.kind={self.kind!r} requires 'home_dir'")
        # 'psij' + 'edge_name' is checked at the RemoteConfig level: whether
        # edge_name is required depends on RemoteConfig.embedded, a sibling
        # field this model can't see.
        if not self.account:
            raise ValueError(f"remote.target.kind={self.kind!r} requires 'account'")
        return self


class RemoteConfig(BaseModel):
    pythonpath: list[str] = []
    backends: list[str] = ["dragon_v3"]
    broker_url: str | None = None
    target: TargetConfig | None = None
    # Host the ORBIT broker in-process (EmbeddedBroker) instead of connecting
    # to an already-running one — no separate radical-orbit-broker.py
    # deployment needed. Mutually exclusive with broker_url. Combined with
    # target.kind: psij, target.edge_name is not required — PsiJ runs on the
    # embedded broker itself.
    embedded: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_embedded(self) -> RemoteConfig:
        if self.embedded and self.broker_url:
            raise ValueError(
                "remote.embedded and remote.broker_url are mutually exclusive "
                "— the embedded broker provides its own URL"
            )
        if (
            self.target is not None
            and self.target.kind == "psij"
            and not self.embedded
            and not self.target.edge_name
        ):
            raise ValueError(
                "remote.target.kind='psij' requires 'edge_name' (unless remote.embedded is true)"
            )
        return self


class TrackingConfig(BaseModel):
    backend: Literal["mlflow", "clearml", "none"] = "none"
    experiment: str = "ROSE-Spec"
    run_name: str | None = None

    model_config = {"extra": "forbid"}


# Keys the builder always injects — users must not put these in parameters:
_RESERVED_PARAMETER_KEYS: frozenset[str] = frozenset(
    {"learner_id", "learner_label", "iteration", "pythonpath"}
)


# ── Top-level spec ────────────────────────────────────────────────────────────
class WorkflowConfig(BaseModel):
    learner: LearnerSpec
    # Task slots — explicit fields keep extra="forbid" and enable IDE autocomplete.
    # Access non-None slots as a dict via the .tasks property.
    simulation: TaskDef | None = None
    training: TaskDef | None = None
    active_learn: TaskDef | None = None
    environment: TaskDef | None = None
    update: TaskDef | None = None
    prediction: TaskDef | None = None
    uncertainty: TaskDef | None = None
    learners: list[LearnerDef] | None = None
    stop_criterion: StopCriterionDef
    parameters: dict[str, Any] = Field(default_factory=dict)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    model_config = {"extra": "forbid"}

    @property
    def tasks(self) -> dict[str, TaskDef]:
        return {s: getattr(self, s) for s in _ALL_SLOTS if getattr(self, s) is not None}

    @model_validator(mode="after")
    def _validate_task_slots(self) -> WorkflowConfig:
        ltype = self.learner.type
        required = _REQUIRED_SLOTS.get(ltype)
        if required is None:
            raise ValueError(
                f"Unknown learner type '{ltype}'. Supported: {sorted(_REQUIRED_SLOTS.keys())}"
            )
        is_parallel = ltype == "parallel_active_learner"

        if is_parallel and self.learners is not None:
            for ld in self.learners:
                missing = required - set(ld.tasks.keys())
                if missing:
                    raise ValueError(f"Learner '{ld.label}' missing: {sorted(missing)}")
                extra = set(ld.tasks.keys()) - required
                if extra:
                    raise ValueError(f"Learner '{ld.label}' unexpected fields: {sorted(extra)}")
            for slot in required:
                types = {ld.tasks[slot].type for ld in self.learners}
                if len(types) > 1:
                    raise ValueError(
                        f"Slot '{slot}' has mixed types across learners: {types}. "
                        "All learners must use the same type for a given slot."
                    )
                descs = [dict(ld.tasks[slot].task_description or {}) for ld in self.learners]
                if len(descs) > 1 and any(d != descs[0] for d in descs[1:]):
                    raise ValueError(
                        f"Slot '{slot}' has different task_description values across learners. "
                        "asyncflow registers task_description once per slot at registration time — "
                        "all learners must use the same value. Use identical task_description "
                        "across all learners or omit it from all but the first."
                    )
        else:
            present = set(self.tasks.keys())
            missing = required - present
            if missing:
                raise ValueError(f"learner type '{ltype}' requires: {sorted(missing)}")
            extra = present - required
            if extra:
                raise ValueError(f"Unexpected task fields for '{ltype}': {sorted(extra)}")
        return self

    @model_validator(mode="after")
    def _validate_parameters(self) -> WorkflowConfig:
        conflicts = _RESERVED_PARAMETER_KEYS & set(self.parameters.keys())
        if conflicts:
            raise ValueError(
                f"'parameters' must not use reserved keys: {sorted(conflicts)}. "
                "These are injected automatically by the spec builder."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
