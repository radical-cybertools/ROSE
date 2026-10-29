from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator


def _check_callable_spec(spec: str) -> None:
    """Validate "module:callable" syntax: both parts present."""
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn:
        raise ValueError(f"'function' must use 'module:callable' syntax, got {spec!r}")


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
            _check_callable_spec(self.function)
        return self


# ── Stop criterion ────────────────────────────────────────────────────────────
class StopCriterionDef(BaseModel):
    metric: str
    threshold: float
    operator: Literal["<", ">", "==", "<=", ">="] = "<"
    evaluator: TaskDef

    model_config = {"extra": "forbid"}


# ── Stream (YAML key: "stream") ──────────────────────────────────────────────
class CallbackDef(BaseModel):
    function: str  # "module:callable", called with the IterationState

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_function(self) -> CallbackDef:
        _check_callable_spec(self.function)
        return self


class CallableDef(CallbackDef):
    """A client-side factory: ``function`` is "module:callable", called with args/kwargs."""

    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class StreamDef(BaseModel):
    """Drives the learner with streamed data, see rose.stream.Stream.

    Sources are factories returning async iterators; they and ``on_model_ready`` run on the client
    (also with ``--remote``), not on the workers.
    """

    batch_size: int = Field(1, ge=1)
    max_wait: float | None = Field(None, gt=0)
    conflate: bool = False
    sources: list[CallableDef] = Field(min_length=1)
    on_model_ready: CallbackDef | None = None

    model_config = {"extra": "forbid"}


# Learner types that accept a stream, and the slot that receives the window
_STREAM_WINDOW_SLOT: dict[str, str] = {
    "sequential_active_learner": "simulation",
    "parallel_active_learner": "simulation",
    "sequential_reinforcement_learner": "environment",
    "uq_active_learner": "simulation",
}


# ── Learner spec (YAML key: "learner") ───────────────────────────────────────
_REQUIRED_SLOTS: dict[str, frozenset[str]] = {
    "sequential_active_learner": frozenset({"simulation", "training", "active_learn"}),
    "parallel_active_learner": frozenset({"simulation", "training", "active_learn"}),
    "sequential_reinforcement_learner": frozenset({"environment", "update"}),
    "uq_active_learner": frozenset({"simulation", "training", "prediction", "active_learn"}),
}
_ALL_SLOTS: frozenset[str] = frozenset().union(*_REQUIRED_SLOTS.values())


class LearnerSpec(BaseModel):
    type: str
    max_iter: int = 0
    parallel_learners: int = 2  # used only when learners: is absent for parallel types
    # uq_active_learner only
    model_names: list[str] | None = None
    num_predictions: int | None = Field(None, ge=1)
    publish_on: Literal["all", "any"] | None = None  # with stream only

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
    # How long `--remote` waits for the bootstrapped endpoint to register
    # with the broker before giving up (queue wait time, not walltime_min
    # above — that's the HPC job's own duration once it starts running).
    endpoint_timeout_min: int = 30
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
    backends: list[str] = ["dragon"]
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
    learners: list[LearnerDef] | None = None
    stop_criterion: StopCriterionDef | None = None  # required unless stream is set
    # uq_active_learner: the uncertainty task and its threshold (metric = uq metric name)
    uncertainty_criterion: StopCriterionDef | None = None
    stream: StreamDef | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    model_config = {"extra": "forbid"}

    # directory of the YAML file: client-side callables (stream sources) are imported from there
    _source_dir: Path | None = PrivateAttr(default=None)

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
        if self.learners is not None and not is_parallel:
            raise ValueError(
                f"'learners' is only supported for parallel_active_learner, not '{ltype}'"
            )

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
    def _validate_uq(self) -> WorkflowConfig:
        learner = self.learner
        if learner.type == "uq_active_learner":
            if not learner.model_names:
                raise ValueError("uq_active_learner requires 'learner.model_names'")
            if len(set(learner.model_names)) != len(learner.model_names):
                raise ValueError("'learner.model_names' must be unique")
            if self.uncertainty_criterion is None:
                raise ValueError("uq_active_learner requires 'uncertainty_criterion'")
            if learner.publish_on is not None and self.stream is None:
                raise ValueError("'learner.publish_on' requires 'stream'")
        else:
            # `is not None`, not fields-set: workflow_with() re-validates explicit None values
            uq_only = {
                "learner.model_names": learner.model_names,
                "learner.num_predictions": learner.num_predictions,
                "learner.publish_on": learner.publish_on,
                "uncertainty_criterion": self.uncertainty_criterion,
            }
            given = sorted(k for k, v in uq_only.items() if v is not None)
            if given:
                raise ValueError(f"{given} only apply to uq_active_learner")
        return self

    @model_validator(mode="after")
    def _validate_stream(self) -> WorkflowConfig:
        if self.stream is None:
            if self.stop_criterion is None:
                raise ValueError("'stop_criterion' is required unless 'stream' is set")
            return self
        ltype = self.learner.type
        slot = _STREAM_WINDOW_SLOT.get(ltype)
        if slot is None:
            raise ValueError(
                f"'stream' is not supported for learner type '{ltype}'. "
                f"Supported: {sorted(_STREAM_WINDOW_SLOT)}"
            )
        # shell tasks build their command from kwargs only and would silently drop the window
        window_tasks = [getattr(self, slot)]
        if self.learners is not None:
            window_tasks = [ld.tasks.get(slot) for ld in self.learners]
        if any(td is not None and td.type == "shell" for td in window_tasks):
            raise ValueError(
                f"with 'stream', the '{slot}' task receives the window and must be a python task "
                "(shell tasks cannot receive it)"
            )
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
        cfg = cls.model_validate(raw)
        cfg._source_dir = Path(path).resolve().parent
        return cfg
