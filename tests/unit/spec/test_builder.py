"""Unit tests for rose.spec.builder — LearnerConfig construction, no HPC needed."""
import textwrap
from unittest.mock import MagicMock

import pytest

from rose.spec.schema import WorkflowConfig


SEQ_WITH_PARAMS = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 3
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
      dataset: test_ds
      scale: 2.5
""")

SEQ_NO_PARAMS = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 3
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
""")

PAR_WITH_PARAMS = textwrap.dedent("""\
    learner:
      type: parallel_active_learner
      max_iter: 2
    learners:
      - label: rf
        simulation:
          type: python
          function: mymod:sim
        training:
          type: python
          function: mymod:train
        active_learn:
          type: python
          function: mymod:select
      - label: mlp
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
      metric: r2
      threshold: 0.9
      operator: ">"
      evaluator:
        type: python
        function: mymod:eval
    parameters:
      dataset: prod_ds
      lr: 0.001
""")


def _make_builder(yaml_text, tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml_text)
    cfg = WorkflowConfig.from_yaml(p)
    from rose.spec.builder import LearnerBuilder
    return LearnerBuilder(cfg, MagicMock())


# ── build_learner_config (sequential path) ────────────────────────────────────

def test_build_learner_config_no_parameters_returns_none(tmp_path):
    builder = _make_builder(SEQ_NO_PARAMS, tmp_path)
    assert builder.build_learner_config() is None


def test_build_learner_config_returns_learner_config(tmp_path):
    from rose.learner import LearnerConfig
    builder = _make_builder(SEQ_WITH_PARAMS, tmp_path)
    lc = builder.build_learner_config()
    assert lc is not None
    assert isinstance(lc, LearnerConfig)


def test_build_learner_config_per_iteration_kwargs(tmp_path):
    builder = _make_builder(SEQ_WITH_PARAMS, tmp_path)
    lc = builder.build_learner_config()
    # iteration 0: parameters + iteration=0
    kw0 = lc.simulation[0].kwargs
    assert kw0["dataset"] == "test_ds"
    assert kw0["scale"] == 2.5
    assert kw0["iteration"] == 0
    # iteration 3 (max_iter): parameters + iteration=3
    kw3 = lc.simulation[3].kwargs
    assert kw3["dataset"] == "test_ds"
    assert kw3["iteration"] == 3


def test_build_learner_config_criterion_schedule_included(tmp_path):
    builder = _make_builder(SEQ_WITH_PARAMS, tmp_path)
    lc = builder.build_learner_config()
    assert lc.criterion is not None
    assert lc.criterion[0].kwargs["dataset"] == "test_ds"


# ── build_learner_configs (parallel path) ────────────────────────────────────

def test_build_learner_configs_no_learners_returns_none(tmp_path):
    builder = _make_builder(SEQ_WITH_PARAMS, tmp_path)
    assert builder.build_learner_configs() is None


def test_build_learner_configs_with_parameters(tmp_path):
    from rose.learner import LearnerConfig
    builder = _make_builder(PAR_WITH_PARAMS, tmp_path)
    lcs = builder.build_learner_configs()
    assert lcs is not None
    assert len(lcs) == 2
    assert all(isinstance(lc, LearnerConfig) for lc in lcs)


def test_build_learner_configs_learner_0_kwargs(tmp_path):
    builder = _make_builder(PAR_WITH_PARAMS, tmp_path)
    lcs = builder.build_learner_configs()
    kw = lcs[0].simulation[0].kwargs
    assert kw["learner_id"] == 0
    assert kw["learner_label"] == "rf"
    assert kw["dataset"] == "prod_ds"
    assert kw["lr"] == 0.001
    assert kw["iteration"] == 0


def test_build_learner_configs_learner_1_kwargs(tmp_path):
    builder = _make_builder(PAR_WITH_PARAMS, tmp_path)
    lcs = builder.build_learner_configs()
    kw = lcs[1].simulation[1].kwargs
    assert kw["learner_id"] == 1
    assert kw["learner_label"] == "mlp"
    assert kw["dataset"] == "prod_ds"
    assert kw["iteration"] == 1


# ── pythonpath auto-injection ─────────────────────────────────────────────────

SEQ_WITH_PYTHONPATH = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 2
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
    remote:
      pythonpath:
        - /remote/path/a
        - /remote/path/b
""")

PAR_WITH_PYTHONPATH = textwrap.dedent("""\
    learner:
      type: parallel_active_learner
      max_iter: 2
    learners:
      - label: rf
        simulation:
          type: python
          function: mymod:sim
        training:
          type: python
          function: mymod:train
        active_learn:
          type: python
          function: mymod:select
      - label: mlp
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
      metric: r2
      threshold: 0.9
      operator: ">"
      evaluator:
        type: python
        function: mymod:eval
    remote:
      pythonpath:
        - /remote/path/a
""")


def test_build_learner_config_injects_pythonpath(tmp_path):
    builder = _make_builder(SEQ_WITH_PYTHONPATH, tmp_path)
    lc = builder.build_learner_config()
    assert lc is not None
    assert lc.simulation[0].kwargs["pythonpath"] == ["/remote/path/a", "/remote/path/b"]


def test_build_learner_config_pythonpath_empty_when_no_remote(tmp_path):
    builder = _make_builder(SEQ_WITH_PARAMS, tmp_path)
    lc = builder.build_learner_config()
    assert lc.simulation[0].kwargs["pythonpath"] == []


def test_build_learner_configs_injects_pythonpath(tmp_path):
    builder = _make_builder(PAR_WITH_PYTHONPATH, tmp_path)
    lcs = builder.build_learner_configs()
    assert lcs[0].simulation[0].kwargs["pythonpath"] == ["/remote/path/a"]
    assert lcs[1].simulation[0].kwargs["pythonpath"] == ["/remote/path/a"]


def test_build_learner_config_no_params_no_pythonpath_returns_none(tmp_path):
    builder = _make_builder(SEQ_NO_PARAMS, tmp_path)
    assert builder.build_learner_config() is None
