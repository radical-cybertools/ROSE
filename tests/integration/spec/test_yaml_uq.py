"""Integration tests for spec-driven UQ runs — real asyncflow, thread pool backend."""

import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec.runner import run_learner
from rose.spec.schema import WorkflowConfig
from tests.integration.spec import helpers

H = "tests.integration.spec.helpers"
UQ_YAML = textwrap.dedent(f"""\
    learner:
      type: uq_active_learner
      max_iter: 2
      model_names: [m1, m2]
      num_predictions: 2
    simulation: {{type: python, function: "{H}:uq_sim"}}
    training: {{type: python, function: "{H}:uq_train"}}
    prediction: {{type: python, function: "{H}:uq_predict"}}
    active_learn: {{type: python, function: "{H}:uq_active_learn"}}
    uncertainty_criterion:
      metric: predictive_entropy
      threshold: THRESHOLD
      operator: "<"
      evaluator: {{type: python, function: "{H}:uq_uncertainty"}}
    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator: {{type: python, function: "{H}:uq_criterion"}}
    parameters:
      dataset: ds
""")

STREAM = textwrap.dedent(f"""\
    stream:
      batch_size: 2
      sources: [{{function: "{H}:numbers", kwargs: {{n: 4}}}}]
      on_model_ready: {{function: "{H}:publish"}}
""")


async def _run(text, tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(text)
    helpers.uq_calls.clear()
    helpers.published.clear()
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    await run_learner(WorkflowConfig.from_yaml(p), asyncflow)


def calls(kind):
    return [c for c in helpers.uq_calls if c[0] == kind]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_uq_runs_with_parameters(tmp_path):
    await _run(UQ_YAML.replace("THRESHOLD", "0.1"), tmp_path)  # UQ threshold never met

    trains = calls("train")
    assert {c[1] for c in trains} == {"m1", "m2"}
    assert all(c[2]["dataset"] == "ds" and "iteration" in c[2] for c in trains)
    assert all(c[2]["dataset"] == "ds" for c in calls("crit"))  # criterion gets the config


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_uq_threshold_stops_early(tmp_path):
    await _run(UQ_YAML.replace("THRESHOLD", "0.5"), tmp_path)  # 0.3 < 0.5 at iteration 0
    assert calls("crit") == []  # stopped before active learning / criterion


@pytest.mark.asyncio
@pytest.mark.integration
async def test_yaml_uq_stream(tmp_path):
    text = UQ_YAML.replace("THRESHOLD", "0.1").replace("  max_iter: 2\n", "  max_iter: 0\n")
    text = text.replace("  num_predictions: 2\n", "  num_predictions: 2\n  publish_on: any\n")
    text = text.replace("  metric: mse\n  threshold: 0.1", "  metric: mse\n  threshold: 1.0")
    await _run(text + STREAM, tmp_path)

    windows = [c[1] for c in calls("sim")]
    assert sorted(map(tuple, windows)) == [(0, 1), (0, 1), (2, 3), (2, 3)]  # per model
    assert helpers.published == [0, 1]  # criterion met (0.5 < 1.0): published every window
