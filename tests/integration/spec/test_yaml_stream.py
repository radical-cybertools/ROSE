"""Integration tests for the YAML spec `stream:` block — real asyncflow, thread pool backend."""

import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec.runner import run_learner
from rose.spec.schema import WorkflowConfig
from tests.integration.spec import helpers

AL_STREAM_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner

    simulation:
      type: python
      function: tests.integration.spec.helpers:sim_window
    training:
      type: python
      function: tests.integration.spec.helpers:train
    active_learn:
      type: python
      function: tests.integration.spec.helpers:active_learn

    stream:
      batch_size: 2
      sources:
        - function: tests.integration.spec.helpers:numbers
          kwargs: {n: 4}
      on_model_ready:
        function: tests.integration.spec.helpers:publish
""")

RL_STREAM_YAML = textwrap.dedent("""\
    learner:
      type: sequential_reinforcement_learner

    environment:
      type: python
      function: tests.integration.spec.helpers:env_window
    update:
      type: python
      function: tests.integration.spec.helpers:update

    stream:
      sources:
        - function: tests.integration.spec.helpers:numbers
          kwargs: {n: 3}
""")


async def _run(yaml_text, tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml_text)
    helpers.received_windows.clear()
    helpers.published.clear()
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    await run_learner(WorkflowConfig.from_yaml(p), asyncflow)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_stream_active_learner(tmp_path):
    await _run(AL_STREAM_YAML, tmp_path)
    assert helpers.received_windows == [[0, 1], [2, 3]]
    assert helpers.published == [0, 1]  # no criterion: every window publishes


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_stream_rl_without_criterion(tmp_path):
    await _run(RL_STREAM_YAML, tmp_path)
    assert helpers.received_windows == [[0], [1], [2]]
