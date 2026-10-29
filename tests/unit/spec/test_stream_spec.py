"""Unit tests for the YAML spec `stream:` block: schema validation, stream construction and client-
side import checks."""

import sys
import textwrap

import pytest

from rose.spec import WorkflowSpec, load_spec
from rose.spec.runner import _build_stream, resolve_client_callables
from rose.spec.schema import WorkflowConfig

BASE = """\
learner:
  type: {ltype}
simulation:
  type: {sim_type}
  {sim_field}
training:
  type: python
  function: mymod:train
active_learn:
  type: python
  function: mymod:select
"""

STREAM = """\
stream:
  batch_size: 2
  sources:
    - function: streammod:source
      kwargs: {n: 3}
  on_model_ready:
    function: streammod:publish
"""

CRITERION = """\
stop_criterion:
  metric: mse
  threshold: 0.1
  evaluator:
    type: python
    function: mymod:eval
"""


def spec(tmp_path, ltype="sequential_active_learner", shell=False, stream=True, criterion=False):
    sim_field = "command: 'sim.sh'" if shell else "function: mymod:sim"
    text = BASE.format(ltype=ltype, sim_type="shell" if shell else "python", sim_field=sim_field)
    text += (STREAM if stream else "") + (CRITERION if criterion else "")
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_stream_makes_stop_criterion_optional(tmp_path):
    cfg = WorkflowConfig.from_yaml(spec(tmp_path))
    assert cfg.stop_criterion is None
    assert cfg.stream.sources[0].kwargs == {"n": 3}


def test_stop_criterion_required_without_stream(tmp_path):
    with pytest.raises(ValueError, match="stop_criterion"):
        WorkflowConfig.from_yaml(spec(tmp_path, stream=False))


def test_shell_window_task_rejected(tmp_path):
    with pytest.raises(ValueError, match="python task"):
        WorkflowConfig.from_yaml(spec(tmp_path, shell=True))


def test_shell_task_fine_without_stream(tmp_path):
    WorkflowConfig.from_yaml(spec(tmp_path, shell=True, stream=False, criterion=True))


def test_stream_rejected_for_uq(tmp_path):
    text = (
        textwrap.dedent("""\
        learner:
          type: uq_active_learner
        simulation: {type: python, function: m:s}
        training: {type: python, function: m:t}
        prediction: {type: python, function: m:p}
        active_learn: {type: python, function: m:a}
        uncertainty: {type: python, function: m:u}
    """)
        + STREAM
    )
    p = tmp_path / "spec.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="not supported"):
        WorkflowConfig.from_yaml(p)


def test_learners_block_rejected_for_sequential(tmp_path):
    p = spec(tmp_path)
    p.write_text(
        p.read_text()
        + "learners:\n  - simulation: {type: python, function: m:s}\n"
        + "    training: {type: python, function: m:t}\n"
        + "    active_learn: {type: python, function: m:a}\n"
    )
    with pytest.raises(ValueError, match="only supported for parallel"):
        WorkflowConfig.from_yaml(p)


@pytest.mark.parametrize(
    "bad", ["batch_size: 0", "max_wait: -1", "sources: []", "sources: [{function: nocolon}]"]
)
def test_invalid_stream_fields(tmp_path, bad):
    p = tmp_path / "spec.yaml"
    head = BASE.format(
        ltype="sequential_active_learner", sim_type="python", sim_field="function: m:s"
    )
    body = "stream:\n  sources: [{function: 'm:src'}]\n"
    key = bad.split(":")[0]
    if key == "sources":
        body = f"stream:\n  {bad}\n"
    else:
        body += f"  {bad}\n"
    p.write_text(textwrap.dedent(head) + body)
    with pytest.raises(ValueError):
        WorkflowConfig.from_yaml(p)


@pytest.fixture
def streammod(tmp_path, monkeypatch):
    (tmp_path / "streammod.py").write_text(
        textwrap.dedent("""\
            async def source(n=3):
                for i in range(n):
                    yield i

            async def async_factory(n=3):
                return source(n)

            def not_an_iterator(n=3):
                return [1, 2, 3]

            def publish(state):
                pass
        """)
    )
    yield tmp_path
    sys.modules.pop("streammod", None)


def test_resolve_client_callables_uses_yaml_dir(tmp_path, streammod, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # restored after the test
    cfg = WorkflowConfig.from_yaml(spec(tmp_path))  # yaml next to streammod.py
    resolve_client_callables(cfg)
    assert str(streammod) in sys.path  # kept for lazy imports during the run


def test_resolve_client_callables_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    cfg = WorkflowConfig.from_yaml(spec(tmp_path))  # no streammod.py next to the yaml
    with pytest.raises(ValueError, match="streammod:source"):
        resolve_client_callables(cfg)


def test_empty_module_or_callable_rejected(tmp_path):
    p = spec(tmp_path)
    p.write_text(p.read_text().replace("streammod:publish", ":publish"))
    with pytest.raises(ValueError, match="module:callable"):
        WorkflowConfig.from_yaml(p)


def test_validate_imports_checks_stream_callables(tmp_path, streammod, monkeypatch):
    (tmp_path / "mymod.py").write_text("def sim(): pass\ndef train(): pass\ndef select(): pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))  # tasks importable as for a local run
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "mymod", raising=False)
    try:
        load_spec(spec(tmp_path), validate_imports=True)
    finally:
        sys.modules.pop("mymod", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", ["source", "async_factory"])
async def test_build_stream_from_factory(tmp_path, streammod, monkeypatch, factory):
    monkeypatch.syspath_prepend(str(streammod))
    cfg = WorkflowConfig.from_yaml(spec(tmp_path))
    cfg.stream.sources[0].function = f"streammod:{factory}"
    stream = await _build_stream(cfg.stream)
    sub = stream._subscribe()
    assert await sub.next_window() == [0, 1]
    await stream.aclose()


@pytest.mark.asyncio
async def test_build_stream_rejects_non_iterator(tmp_path, streammod, monkeypatch):
    monkeypatch.syspath_prepend(str(streammod))
    cfg = WorkflowConfig.from_yaml(spec(tmp_path))
    cfg.stream.sources[0].function = "streammod:not_an_iterator"
    with pytest.raises(TypeError, match="async iterator"):
        await _build_stream(cfg.stream)


def test_workflow_with_replaces_stream_block(tmp_path):
    ws = WorkflowSpec(WorkflowConfig.from_yaml(spec(tmp_path)))
    new = ws.workflow_with(stream={"sources": [{"function": "m:other"}]})
    assert [s.function for s in new.config.stream.sources] == ["m:other"]
    assert new.config.stream.on_model_ready is None  # whole block replaced
    with pytest.raises(ValueError, match="stop_criterion"):
        ws.workflow_with(stream=None)


@pytest.mark.asyncio
async def test_unstarted_sources_closed_on_aclose():
    from rose import Stream

    closed = []

    async def source():
        try:
            yield 1
        finally:
            closed.append(True)

    gen = source()
    await gen.__anext__()  # started, suspended at its yield
    stream = Stream(sources=gen)
    await stream.aclose()  # never pulled: the pump never ran
    assert closed == [True]
