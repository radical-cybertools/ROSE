from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Task definition ───────────────────────────────────────────────────────────
class TaskDef(BaseModel):
    type: Literal["shell", "python"]
    command: str | None = None    # required when type=="shell"
    function: str | None = None   # required when type=="python"; "module:callable"
    task_description: dict[str, Any] | None = None  # resource hints forwarded to asyncflow backend

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_fields(self) -> "TaskDef":
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
    "sequential_active_learner":        frozenset({"simulation", "training", "active_learn"}),
    "parallel_active_learner":          frozenset({"simulation", "training", "active_learn"}),
    "sequential_reinforcement_learner": frozenset({"environment", "update"}),
    "uq_active_learner":                frozenset({"simulation", "training", "prediction",
                                                   "active_learn", "uncertainty"}),
}
_ALL_SLOTS: frozenset[str] = frozenset().union(*_REQUIRED_SLOTS.values())


class LearnerSpec(BaseModel):
    type: str
    max_iter: int = 0
    parallel_learners: int = 2    # used only when candidates is absent for parallel types

    model_config = {"extra": "forbid"}


# ── Per-candidate definition (parallel learners only) ─────────────────────────
class CandidateDef(BaseModel):
    label:        str            = ""
    simulation:   TaskDef | None = None
    training:     TaskDef | None = None
    active_learn: TaskDef | None = None
    environment:  TaskDef | None = None
    update:       TaskDef | None = None
    prediction:   TaskDef | None = None
    uncertainty:  TaskDef | None = None

    model_config = {"extra": "forbid"}

    @property
    def tasks(self) -> dict[str, TaskDef]:
        return {s: getattr(self, s) for s in _ALL_SLOTS if getattr(self, s) is not None}


# ── Remote / tracking ────────────────────────────────────────────────────────
class RemoteConfig(BaseModel):
    pythonpath: list[str] = []

    model_config = {"extra": "forbid"}


class TrackingConfig(BaseModel):
    backend: Literal["mlflow", "clearml", "none"] = "none"
    experiment: str = "ROSE-Spec"
    run_name: str | None = None

    model_config = {"extra": "forbid"}


# ── Top-level spec ────────────────────────────────────────────────────────────
class SpecConfig(BaseModel):
    learner:        LearnerSpec
    # Task slots — explicit fields keep extra="forbid" and enable IDE autocomplete.
    # Access non-None slots as a dict via the .tasks property.
    simulation:     TaskDef | None = None
    training:       TaskDef | None = None
    active_learn:   TaskDef | None = None
    environment:    TaskDef | None = None
    update:         TaskDef | None = None
    prediction:     TaskDef | None = None
    uncertainty:    TaskDef | None = None
    candidates:     list[CandidateDef] | None = None
    stop_criterion: StopCriterionDef
    remote:         RemoteConfig   = Field(default_factory=RemoteConfig)
    tracking:       TrackingConfig = Field(default_factory=TrackingConfig)

    model_config = {"extra": "forbid"}

    @property
    def tasks(self) -> dict[str, TaskDef]:
        return {s: getattr(self, s) for s in _ALL_SLOTS if getattr(self, s) is not None}

    @model_validator(mode="after")
    def _validate_task_slots(self) -> "SpecConfig":
        ltype    = self.learner.type
        required = _REQUIRED_SLOTS.get(ltype)
        if required is None:
            raise ValueError(
                f"Unknown learner type '{ltype}'. "
                f"Supported: {sorted(_REQUIRED_SLOTS.keys())}"
            )
        is_parallel = ltype == "parallel_active_learner"

        if is_parallel and self.candidates is not None:
            for c in self.candidates:
                missing = required - set(c.tasks.keys())
                if missing:
                    raise ValueError(f"Candidate '{c.label}' missing: {sorted(missing)}")
                extra = set(c.tasks.keys()) - required
                if extra:
                    raise ValueError(f"Candidate '{c.label}' unexpected fields: {sorted(extra)}")
            for slot in required:
                types = {c.tasks[slot].type for c in self.candidates}
                if len(types) > 1:
                    raise ValueError(
                        f"Slot '{slot}' has mixed types across candidates: {types}. "
                        "All candidates must use the same type for a given slot."
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpecConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
