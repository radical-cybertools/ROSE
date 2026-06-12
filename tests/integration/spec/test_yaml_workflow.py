"""Integration tests for the YAML spec layer — full AL loop, no HPC required."""
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec import load_spec
from rose.spec.builder import LearnerBuilder
from rose.spec.schema import SpecConfig


# ── Sequential AL via YAML ────────────────────────────────────────────────────

SEQ_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 3

    simulation:
      type: python
      function: tests.integration.spec.helpers:sim

    training:
      type: python
      function: tests.integration.spec.helpers:train

    active_learn:
      type: python
      function: tests.integration.spec.helpers:active_learn

    stop_criterion:
      metric: mse
      threshold: 0.01
      operator: "<"
      evaluator:
        type: python
        function: tests.integration.spec.helpers:criterion
""")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_sequential_workflow(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_YAML)

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    cfg = SpecConfig.from_yaml(p)
    builder = LearnerBuilder(cfg, asyncflow)
    learner = builder.build()

    states = []
    async for state in learner.start(max_iter=3):
        states.append(state)

    assert len(states) == 3
    await asyncflow.shutdown()


# ── load_spec convenience wrapper ─────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_spec_returns_workflow_spec(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_YAML)
    spec = load_spec(p)
    assert hasattr(spec, "workflow")
    assert callable(spec.workflow)
