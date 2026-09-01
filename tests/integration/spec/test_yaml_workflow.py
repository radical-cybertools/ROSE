"""Integration tests for the YAML spec layer — full AL loop, no HPC required."""

import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec import load_spec
from rose.spec.builder import LearnerBuilder
from rose.spec.schema import WorkflowConfig

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

    cfg = WorkflowConfig.from_yaml(p)
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


# ── Sequential AL with parameters: block ─────────────────────────────────────

SEQ_WITH_PARAMS_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 2

    simulation:
      type: python
      function: tests.integration.spec.helpers:sim_capture

    training:
      type: python
      function: tests.integration.spec.helpers:train_capture

    active_learn:
      type: python
      function: tests.integration.spec.helpers:active_learn_capture

    stop_criterion:
      metric: mse
      threshold: 0.01
      operator: "<"
      evaluator:
        type: python
        function: tests.integration.spec.helpers:criterion_capture

    parameters:
      dataset: test_ds
      scale: 1.5
""")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sequential_workflow_parameters_reach_tasks(tmp_path):
    import tests.integration.spec.helpers as helpers

    helpers.received_kwargs.clear()

    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_WITH_PARAMS_YAML)

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    cfg = WorkflowConfig.from_yaml(p)
    builder = LearnerBuilder(cfg, asyncflow)
    learner = builder.build()

    ic = builder.build_learner_config()
    start_kwargs = {"max_iter": 2}
    if ic is not None:
        start_kwargs["initial_config"] = ic

    async for _ in learner.start(**start_kwargs):
        pass

    await asyncflow.shutdown()

    assert len(helpers.received_kwargs) > 0, "sim_capture was never called"
    for kw in helpers.received_kwargs:
        assert kw.get("dataset") == "test_ds", f"expected dataset in kwargs, got {kw}"
        assert kw.get("scale") == 1.5, f"expected scale in kwargs, got {kw}"


# ── Parallel AL with parameters: block ───────────────────────────────────────

PAR_WITH_PARAMS_YAML = textwrap.dedent("""\
    learner:
      type: parallel_active_learner
      max_iter: 2

    learners:
      - label: rf
        simulation:
          type: python
          function: tests.integration.spec.helpers:sim_capture
        training:
          type: python
          function: tests.integration.spec.helpers:train_capture
        active_learn:
          type: python
          function: tests.integration.spec.helpers:active_learn_capture
      - label: mlp
        simulation:
          type: python
          function: tests.integration.spec.helpers:sim_capture
        training:
          type: python
          function: tests.integration.spec.helpers:train_capture
        active_learn:
          type: python
          function: tests.integration.spec.helpers:active_learn_capture

    stop_criterion:
      metric: mse
      threshold: 0.01
      operator: "<"
      evaluator:
        type: python
        function: tests.integration.spec.helpers:criterion_capture

    parameters:
      dataset: par_ds
      lr: 0.01
""")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_parallel_workflow_parameters_reach_tasks(tmp_path):
    import tests.integration.spec.helpers as helpers

    helpers.received_kwargs.clear()

    p = tmp_path / "spec.yaml"
    p.write_text(PAR_WITH_PARAMS_YAML)

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    cfg = WorkflowConfig.from_yaml(p)
    builder = LearnerBuilder(cfg, asyncflow)
    learner = builder.build()

    lcs = builder.build_learner_configs()
    start_kwargs = {"max_iter": 2, "parallel_learners": len(lcs), "learner_configs": lcs}

    async for _ in learner.start(**start_kwargs):
        pass

    await asyncflow.shutdown()

    assert len(helpers.received_kwargs) > 0, "sim_capture was never called"
    for kw in helpers.received_kwargs:
        assert kw.get("dataset") == "par_ds", f"expected dataset in kwargs, got {kw}"
        assert kw.get("lr") == 0.01, f"expected lr in kwargs, got {kw}"
        # learner_id is stripped by the dispatch closure; learner_label passes through
        assert "learner_id" not in kw, "learner_id should be stripped by dispatch"
        assert kw.get("learner_label") in {"rf", "mlp"}
