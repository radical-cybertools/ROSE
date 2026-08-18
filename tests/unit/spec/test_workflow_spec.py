"""Unit tests for WorkflowSpec — load_spec and workflow_with()."""

import sys
import textwrap
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from rose.spec import WorkflowSpec, load_spec

SEQ_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 5

    simulation:
      type: python
      function: mymod:sim

    training:
      type: python
      function: mymod:train

    active_learn:
      type: python
      function: mymod:select

    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator:
        type: python
        function: mymod:eval

    parameters:
      dataset: base_ds
      scale: 1.0
""")


def _make_spec(tmp_path, yaml=SEQ_YAML):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml)
    return load_spec(p)


# ── workflow_with: learner field override ─────────────────────────────────────


def test_workflow_with_max_iter(tmp_path):
    spec = _make_spec(tmp_path)
    new = spec.workflow_with(max_iter=2)
    assert new.config.learner.max_iter == 2
    assert spec.config.learner.max_iter == 5  # original unchanged


def test_workflow_with_does_not_mutate_original(tmp_path):
    spec = _make_spec(tmp_path)
    _ = spec.workflow_with(max_iter=1)
    assert spec.config.learner.max_iter == 5


# ── workflow_with: parameters merge ──────────────────────────────────────────


def test_workflow_with_parameters_merges(tmp_path):
    spec = _make_spec(tmp_path)
    new = spec.workflow_with(parameters={"dataset": "test_ds"})
    assert new.config.parameters["dataset"] == "test_ds"
    assert new.config.parameters["scale"] == 1.0  # existing key preserved


def test_workflow_with_parameters_adds_new_key(tmp_path):
    spec = _make_spec(tmp_path)
    new = spec.workflow_with(parameters={"lr": 0.001})
    assert new.config.parameters["lr"] == 0.001
    assert new.config.parameters["dataset"] == "base_ds"


def test_workflow_with_parameters_does_not_mutate_original(tmp_path):
    spec = _make_spec(tmp_path)
    _ = spec.workflow_with(parameters={"dataset": "other"})
    assert spec.config.parameters["dataset"] == "base_ds"


# ── workflow_with: combined overrides ─────────────────────────────────────────


def test_workflow_with_combined(tmp_path):
    spec = _make_spec(tmp_path)
    new = spec.workflow_with(max_iter=3, parameters={"dataset": "test_ds", "scale": 2.0})
    assert new.config.learner.max_iter == 3
    assert new.config.parameters["dataset"] == "test_ds"
    assert new.config.parameters["scale"] == 2.0


# ── workflow_with: unknown field raises ───────────────────────────────────────


def test_workflow_with_unknown_field_raises(tmp_path):
    spec = _make_spec(tmp_path)
    with pytest.raises(ValueError, match="unknown spec field"):
        spec.workflow_with(nonexistent_key=42)


# ── workflow_with: returns WorkflowSpec with callable .workflow ───────────────


def test_workflow_with_returns_workflow_spec(tmp_path):
    spec = _make_spec(tmp_path)
    new = spec.workflow_with(max_iter=1)
    assert isinstance(new, WorkflowSpec)
    assert callable(new.workflow)


# ── validate_imports ──────────────────────────────────────────────────────────

IMPORTABLE_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 1

    simulation:
      type: python
      function: os.path:join

    training:
      type: python
      function: os.path:dirname

    active_learn:
      type: python
      function: os.path:basename

    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator:
        type: python
        function: os.path:exists
""")

BAD_MODULE_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 1

    simulation:
      type: python
      function: no_such_module_xyz:fn

    training:
      type: python
      function: os.path:dirname

    active_learn:
      type: python
      function: os.path:basename

    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator:
        type: python
        function: os.path:exists
""")

BAD_ATTR_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 1

    simulation:
      type: python
      function: os.path:no_such_fn_xyz

    training:
      type: python
      function: os.path:dirname

    active_learn:
      type: python
      function: os.path:basename

    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator:
        type: python
        function: os.path:exists
""")

MULTI_BAD_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 1

    simulation:
      type: python
      function: no_such_module_xyz:fn

    training:
      type: python
      function: os.path:no_such_fn_xyz

    active_learn:
      type: python
      function: os.path:basename

    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator:
        type: python
        function: os.path:exists
""")


def test_validate_imports_succeeds_for_importable_functions(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(IMPORTABLE_YAML)
    spec = load_spec(p, validate_imports=True)
    assert spec is not None


def test_validate_imports_raises_on_bad_module(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(BAD_MODULE_YAML)
    with pytest.raises(ValueError, match="no_such_module_xyz:fn"):
        load_spec(p, validate_imports=True)


def test_validate_imports_raises_on_bad_attribute(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(BAD_ATTR_YAML)
    with pytest.raises(ValueError, match="no_such_fn_xyz"):
        load_spec(p, validate_imports=True)


def test_validate_imports_reports_all_failures(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(MULTI_BAD_YAML)
    with pytest.raises(ValueError) as exc_info:
        load_spec(p, validate_imports=True)
    msg = str(exc_info.value)
    assert "no_such_module_xyz:fn" in msg
    assert "no_such_fn_xyz" in msg


def test_validate_imports_default_false_skips_check(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(BAD_MODULE_YAML)
    spec = load_spec(p)  # validate_imports=False by default — should not raise
    assert spec is not None


# ── workflow: calls rhapsody.get_backend with broker_url, not bridge_url ──────
#
# Regression test for the orbit Bridge -> Broker rename: OrbitExecutionBackend
# (rhapsody) takes `broker_url=`, and this call site used to pass the stale
# `bridge_url=` kwarg (a TypeError against current rhapsody). `rhapsody` and
# `radical.asyncflow` are stubbed out and made to fail immediately inside
# get_backend, so this isolates the kwarg names without needing to simulate
# the rest of the execution stack (LearnerBuilder / learner.start / asyncflow).


class _Sentinel(Exception):
    pass


async def test_workflow_calls_get_backend_with_broker_url(tmp_path, monkeypatch):
    spec = _make_spec(tmp_path)
    captured = {}

    async def fake_get_backend(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        raise _Sentinel

    fake_rhapsody = types.ModuleType("rhapsody")
    fake_rhapsody.get_backend = fake_get_backend

    fake_asyncflow_mod = types.ModuleType("radical.asyncflow")
    fake_asyncflow_mod.WorkflowEngine = MagicMock()

    monkeypatch.setitem(sys.modules, "rhapsody", fake_rhapsody)
    monkeypatch.setitem(sys.modules, "radical.asyncflow", fake_asyncflow_mod)

    with pytest.raises(_Sentinel):
        await spec.workflow("https://broker.example:8000", "ep-1")

    assert captured["name"] == "orbit"
    assert "bridge_url" not in captured
    assert captured["broker_url"] == "https://broker.example:8000"
    assert captured["endpoint_name"] == "ep-1"
    assert captured["backends"] == spec.config.remote.backends
