"""Unit tests for spec-driven UQ runs: schema, builder registration, configs and start kwargs."""

import textwrap
from unittest.mock import MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.spec import WorkflowSpec
from rose.spec.builder import LearnerBuilder
from rose.spec.runner import _start_kwargs
from rose.spec.schema import WorkflowConfig
from rose.uncertainty_quantification.uq_learner import UQLearnerConfig

UQ_YAML = textwrap.dedent("""\
    learner:
      type: uq_active_learner
      max_iter: 3
      model_names: [m1, m2]
      num_predictions: 2
    simulation: {type: python, function: m:s}
    training: {type: python, function: m:t}
    prediction: {type: python, function: m:p}
    active_learn: {type: python, function: m:a}
    uncertainty_criterion:
      metric: predictive_entropy
      threshold: 0.5
      operator: "<"
      evaluator: {type: python, function: m:u}
    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator: {type: python, function: m:c}
    parameters:
      dataset: ds
""")

SEQ_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 3
    simulation: {type: python, function: m:s}
    training: {type: python, function: m:t}
    active_learn: {type: python, function: m:a}
    stop_criterion:
      metric: mse
      threshold: 0.1
      evaluator: {type: python, function: m:c}
""")


def load(tmp_path, text):
    p = tmp_path / "spec.yaml"
    p.write_text(text)
    return WorkflowConfig.from_yaml(p)


def test_uq_spec_valid(tmp_path):
    cfg = load(tmp_path, UQ_YAML)
    assert cfg.learner.model_names == ["m1", "m2"]
    assert cfg.uncertainty_criterion.metric == "predictive_entropy"


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("  model_names: [m1, m2]\n", "", "model_names"),
        ("[m1, m2]", "[m1, m1]", "unique"),
        ("uncertainty_criterion:", "unused_block:", "Extra inputs|uncertainty_criterion"),
        ("  num_predictions: 2\n", "  num_predictions: 2\n  publish_on: any\n", "stream"),
        ("simulation: {", "uncertainty: {type: python, function: m:x}\nsimulation: {", "Extra"),
    ],
)
def test_uq_spec_invalid(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, UQ_YAML.replace(old, new))


def test_uq_only_fields_rejected_elsewhere(tmp_path):
    text = SEQ_YAML.replace("  max_iter: 3\n", "  max_iter: 3\n  model_names: [m1]\n")
    with pytest.raises(ValueError, match="only apply to uq_active_learner"):
        load(tmp_path, text)


def test_workflow_with_on_non_uq_spec(tmp_path):
    spec = WorkflowSpec(load(tmp_path, SEQ_YAML))
    assert spec.workflow_with(max_iter=5).config.learner.max_iter == 5


def test_builder_registers_uncertainty_and_uq_config(tmp_path):
    cfg = load(tmp_path, UQ_YAML)
    builder = LearnerBuilder(cfg, MagicMock(spec=WorkflowEngine))
    learner = builder.build()

    uq = learner.uncertainty_function
    assert (uq["uq_metric_name"], uq["threshold"], uq["operator"]) == (
        "predictive_entropy",
        0.5,
        "<",
    )

    lc = builder.build_learner_config()
    assert isinstance(lc, UQLearnerConfig)
    assert lc.get_task_config("uncertainty", 4).kwargs["iteration"] == 4
    assert lc.get_task_config("criterion", 1).kwargs["dataset"] == "ds"

    kwargs = _start_kwargs(cfg, builder)
    assert kwargs["model_names"] == ["m1", "m2"]
    assert kwargs["num_predictions"] == 2
    assert kwargs["learning_config"] is not None
    assert "initial_config" not in kwargs and "publish_on" not in kwargs


def test_uq_injected_kwargs_are_reserved(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        load(tmp_path, UQ_YAML.replace("  dataset: ds", '  "--model_name": x'))


def test_uncertainty_operator_defaults_to_metric_direction(tmp_path):
    cfg = load(
        tmp_path,
        UQ_YAML.replace(
            '  operator: "<"\n  evaluator: {type: python, function: m:u}',
            "  evaluator: {type: python, function: m:u}",
        ),
    )
    from rose.metrics import LearningMetrics

    builtin = LearningMetrics.get_operator("predictive_entropy")
    assert cfg.uncertainty_criterion.operator == builtin  # resolved: the operator actually used
    learner = LearnerBuilder(cfg, MagicMock(spec=WorkflowEngine)).build()
    assert learner.uncertainty_function["operator"] == builtin


@pytest.mark.parametrize(
    ("metric", "operator", "ok"),
    [
        ("my_spread", None, False),
        ("my_spread", ">", True),
        ("predictive_entropy", "<", True),
        ("predictive_entropy", ">", False),
    ],
)
def test_uncertainty_operator_rules(tmp_path, metric, operator, ok):
    text = UQ_YAML.replace("metric: predictive_entropy", f"metric: {metric}").replace(
        '  operator: "<"\n  evaluator: {type: python, function: m:u}',
        (f'  operator: "{operator}"\n' if operator else "")
        + "  evaluator: {type: python, function: m:u}",
    )
    if ok:
        load(tmp_path, text)
    else:
        with pytest.raises(ValueError, match="operator"):
            load(tmp_path, text)


def test_uq_tasks_get_iteration_without_parameters(tmp_path):
    cfg = load(tmp_path, UQ_YAML.replace("parameters:\n  dataset: ds\n", ""))
    lc = LearnerBuilder(cfg, MagicMock(spec=WorkflowEngine)).build_learner_config()
    assert lc.get_task_config("training", 2).kwargs["iteration"] == 2


@pytest.mark.parametrize(
    ("metric", "operator", "expect"),
    [
        ("mse", None, "<"),  # custom: default "<"
        ("mse", ">", ">"),
        ("model_accuracy", None, "builtin"),
        ("model_accuracy", "<", "error"),  # reverses the built-in direction (">=")
        ("model_accuracy", ">", "builtin"),  # same direction: accepted, resolved to ">="
    ],
)
def test_stop_criterion_operator_rules(tmp_path, metric, operator, expect):
    from rose.metrics import LearningMetrics

    text = SEQ_YAML.replace("metric: mse", f"metric: {metric}")
    if operator:
        text = text.replace("  threshold: 0.1\n", f'  threshold: 0.1\n  operator: "{operator}"\n')
    if expect == "error":
        with pytest.raises(ValueError, match="operator"):
            load(tmp_path, text)
        return
    op = load(tmp_path, text).stop_criterion.operator
    assert op == (LearningMetrics.get_operator(metric) if expect == "builtin" else expect)


def test_custom_uncertainty_metric_needs_operator(tmp_path):
    text = UQ_YAML.replace("metric: predictive_entropy", "metric: my_spread").replace(
        '  operator: "<"\n  evaluator: {type: python, function: m:u}',
        "  evaluator: {type: python, function: m:u}",
    )
    with pytest.raises(ValueError, match="uncertainty_criterion: custom metric"):
        load(tmp_path, text)


def test_uq_spec_without_stop_criterion(tmp_path):
    block = (
        "stop_criterion:\n  metric: mse\n  threshold: 0.1\n"
        "  evaluator: {type: python, function: m:c}\n"
    )
    assert block in UQ_YAML
    cfg = load(tmp_path, UQ_YAML.replace(block, ""))
    assert cfg.stop_criterion is None
    learner = LearnerBuilder(cfg, MagicMock(spec=WorkflowEngine)).build()
    assert not learner.criterion_function
    assert learner.uncertainty_function["threshold"] == 0.5
