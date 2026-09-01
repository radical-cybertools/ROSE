"""Unit tests for rose.spec.schema — YAML validation without any HPC machinery."""

import textwrap

import pytest
from pydantic import ValidationError

from rose.spec.schema import RemoteConfig, TargetConfig, TaskDef, WorkflowConfig

# ── TaskDef ───────────────────────────────────────────────────────────────────


def test_taskdef_shell_valid():
    t = TaskDef(type="shell", command="python sim.py")
    assert t.type == "shell"
    assert t.command == "python sim.py"


def test_taskdef_python_valid():
    t = TaskDef(type="python", function="mymod.sub:fn")
    assert t.function == "mymod.sub:fn"


def test_taskdef_shell_missing_command():
    with pytest.raises(ValueError, match="shell task requires 'command'"):
        TaskDef(type="shell")


def test_taskdef_python_missing_function():
    with pytest.raises(ValueError, match="python task requires 'function'"):
        TaskDef(type="python")


def test_taskdef_python_bad_syntax():
    with pytest.raises(ValueError, match="module:callable"):
        TaskDef(type="python", function="mymod.fn")


def test_taskdef_extra_field_rejected():
    with pytest.raises(ValidationError):
        TaskDef(type="shell", command="x", unknown_field="y")


def test_taskdef_task_description_roundtrip():
    t = TaskDef(
        type="shell", command="python sim.py", task_description={"cpu_count": 4, "gpu_count": 1}
    )
    assert t.task_description == {"cpu_count": 4, "gpu_count": 1}


def test_taskdef_task_description_defaults_none():
    t = TaskDef(type="shell", command="x")
    assert t.task_description is None


def test_sequential_spec_task_description_from_yaml(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
        simulation:
          type: shell
          command: python sim.py
          task_description:
            cpu_count: 8
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
    p = tmp_path / "td.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.simulation.task_description == {"cpu_count": 8}
    assert cfg.training.task_description is None


# ── WorkflowConfig — sequential learner ──────────────────────────────────────────

SEQ_YAML = textwrap.dedent("""\
    learner:
      type: sequential_active_learner
      max_iter: 5

    simulation:
      type: shell
      command: python sim.py

    training:
      type: python
      function: mymod:train

    active_learn:
      type: python
      function: mymod:select

    stop_criterion:
      metric: mse
      threshold: 0.1
      operator: "<"
      evaluator:
        type: python
        function: mymod:eval_mse
""")


def test_sequential_spec_roundtrip(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_YAML)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.learner.type == "sequential_active_learner"
    assert cfg.learner.max_iter == 5
    assert cfg.simulation.command == "python sim.py"
    assert cfg.training.function == "mymod:train"
    assert cfg.stop_criterion.metric == "mse"
    assert cfg.tasks == {
        "simulation": cfg.simulation,
        "training": cfg.training,
        "active_learn": cfg.active_learn,
    }


def test_sequential_spec_missing_slot(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
        simulation:
          type: shell
          command: python sim.py
        training:
          type: python
          function: mymod:train
        stop_criterion:
          metric: mse
          threshold: 0.1
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="active_learn"):
        WorkflowConfig.from_yaml(p)


def test_sequential_spec_extra_slot(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
        simulation:
          type: shell
          command: python sim.py
        training:
          type: python
          function: mymod:train
        active_learn:
          type: python
          function: mymod:select
        environment:
          type: shell
          command: python env.py
        stop_criterion:
          metric: mse
          threshold: 0.1
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="Unexpected"):
        WorkflowConfig.from_yaml(p)


def test_unknown_learner_type(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: unknown_learner
        simulation:
          type: shell
          command: x
        stop_criterion:
          metric: mse
          threshold: 0.1
          evaluator:
            type: shell
            command: x
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="Unknown learner type"):
        WorkflowConfig.from_yaml(p)


# ── WorkflowConfig — parallel learner with learners ──────────────────────────────

PAR_YAML = textwrap.dedent("""\
    learner:
      type: parallel_active_learner
      max_iter: 3

    learners:
      - label: rf
        simulation:
          type: shell
          command: python sim.py --model rf
        training:
          type: python
          function: mymod:train_rf
        active_learn:
          type: python
          function: mymod:select
      - label: mlp
        simulation:
          type: shell
          command: python sim.py --model mlp
        training:
          type: python
          function: mymod:train_mlp
        active_learn:
          type: python
          function: mymod:select

    stop_criterion:
      metric: r2
      threshold: 0.9
      operator: ">"
      evaluator:
        type: python
        function: mymod:eval_r2
""")


def test_parallel_learners_roundtrip(tmp_path):
    p = tmp_path / "par.yaml"
    p.write_text(PAR_YAML)
    cfg = WorkflowConfig.from_yaml(p)
    assert len(cfg.learners) == 2
    assert cfg.learners[0].label == "rf"
    assert cfg.learners[1].tasks["training"].function == "mymod:train_mlp"


def test_parallel_learner_missing_slot(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: rf
            simulation:
              type: shell
              command: python sim.py
            training:
              type: python
              function: mymod:train
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="active_learn"):
        WorkflowConfig.from_yaml(p)


# ── WorkflowConfig — parameters block ────────────────────────────────────────────

# ── WorkflowConfig — reserved parameter keys ──────────────────────────────────────

# ── WorkflowConfig — task_description consistency across parallel learners ────────


def test_parallel_learners_identical_task_description_valid(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: a
            simulation:
              type: shell
              command: python sim.py
              task_description:
                cpu_count: 4
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
          - label: b
            simulation:
              type: shell
              command: python sim.py --model b
              task_description:
                cpu_count: 4
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "ok.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert len(cfg.learners) == 2


def test_parallel_learners_nested_task_description_different_key_order_valid(tmp_path):
    """Nested dicts with the same keys/values in a different insertion order must compare equal —
    dict equality, not string-sorted serialization."""
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: a
            simulation:
              type: shell
              command: python sim.py
              task_description:
                resources:
                  gpus: 1
                  cpus: 4
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
          - label: b
            simulation:
              type: shell
              command: python sim.py --model b
              task_description:
                resources:
                  cpus: 4
                  gpus: 1
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "ok_nested.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert len(cfg.learners) == 2


def test_parallel_learners_different_task_description_raises(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: a
            simulation:
              type: shell
              command: python sim.py
              task_description:
                cpu_count: 4
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
          - label: b
            simulation:
              type: shell
              command: python sim.py
              task_description:
                cpu_count: 8
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="task_description"):
        WorkflowConfig.from_yaml(p)


def test_parallel_learners_one_has_task_description_other_absent_raises(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: a
            simulation:
              type: shell
              command: python sim.py
              task_description:
                cpu_count: 4
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
          - label: b
            simulation:
              type: shell
              command: python sim.py
            training:
              type: python
              function: mymod:train
            active_learn:
              type: python
              function: mymod:select
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="task_description"):
        WorkflowConfig.from_yaml(p)


def test_remote_backends_custom(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          backends: [concurrent]
    """)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.remote.backends == ["concurrent"]


def test_remote_backends_default():
    from rose.spec.schema import RemoteConfig

    assert RemoteConfig().backends == ["dragon"]


def test_remote_extra_field_rejected(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          foo: bar
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValidationError):
        WorkflowConfig.from_yaml(p)


def test_parameters_reserved_key_pythonpath(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          pythonpath: /some/path
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="reserved keys"):
        WorkflowConfig.from_yaml(p)


def test_parameters_reserved_key_iteration(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          iteration: 5
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="reserved keys"):
        WorkflowConfig.from_yaml(p)


def test_parameters_reserved_key_learner_id(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          learner_id: 0
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="reserved keys"):
        WorkflowConfig.from_yaml(p)


def test_parameters_non_reserved_key_allowed(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          dataset: my_ds
    """)
    p = tmp_path / "ok.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.parameters == {"dataset": "my_ds"}


def test_parameters_roundtrip(tmp_path):
    yaml = textwrap.dedent("""\
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
          dataset: my_ds
          scale: 1.5
          growing_pool: false
    """)
    p = tmp_path / "params.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.parameters == {"dataset": "my_ds", "scale": 1.5, "growing_pool": False}


def test_parameters_defaults_empty(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_YAML)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.parameters == {}


# ── WorkflowConfig — parallel learners (mixed types) ─────────────────────────────


# ── RemoteConfig.target (TargetConfig) ───────────────────────────────────────


def test_remote_target_default_none():
    assert RemoteConfig().target is None


def test_remote_embedded_default_false():
    assert RemoteConfig().embedded is False


def test_remote_embedded_and_broker_url_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RemoteConfig(embedded=True, broker_url="https://broker:8000")


def test_remote_embedded_without_broker_url_valid():
    cfg = RemoteConfig(embedded=True)
    assert cfg.embedded is True
    assert cfg.broker_url is None


def test_remote_psij_edge_name_required_unless_embedded():
    with pytest.raises(ValueError, match="requires 'edge_name'"):
        RemoteConfig(
            embedded=False,
            target=TargetConfig(kind="psij", account="amsc007"),
        )


def test_remote_psij_edge_name_not_required_when_embedded():
    cfg = RemoteConfig(
        embedded=True,
        target=TargetConfig(kind="psij", account="amsc007"),
    )
    assert cfg.target.edge_name is None


def test_remote_psij_edge_name_still_valid_when_embedded():
    """A user may still set edge_name when embedded — harmless, ignored by the RemoteConfig-level
    requirement check."""
    cfg = RemoteConfig(
        embedded=True,
        target=TargetConfig(kind="psij", edge_name="login1", account="amsc007"),
    )
    assert cfg.target.edge_name == "login1"


def test_remote_iri_target_unaffected_by_embedded_validator():
    cfg = RemoteConfig(
        embedded=True,
        target=TargetConfig(
            kind="iri",
            endpoint="olcf",
            resource_id="odo",
            account="fus183",
            home_dir="/home/x",
        ),
    )
    assert cfg.target.kind == "iri"


def test_target_config_sfapi_valid():
    t = TargetConfig(
        kind="sfapi",
        endpoint="nersc",
        resource_id="perlmutter",
        account="amsc007",
        home_dir="/global/u2/m/merzky",
    )
    assert t.kind == "sfapi"
    assert t.tunnel == "none"
    assert t.walltime_min == 30
    assert t.endpoint_timeout_min == 30


def test_target_config_endpoint_timeout_min_custom():
    t = TargetConfig(
        kind="sfapi",
        endpoint="nersc",
        resource_id="perlmutter",
        account="amsc007",
        home_dir="/global/u2/m/merzky",
        endpoint_timeout_min=90,
    )
    assert t.endpoint_timeout_min == 90
    assert "endpoint_timeout_min" in t.model_fields_set


def test_target_config_endpoint_timeout_min_not_in_fields_set_by_default():
    t = TargetConfig(
        kind="sfapi",
        endpoint="nersc",
        resource_id="perlmutter",
        account="amsc007",
        home_dir="/global/u2/m/merzky",
    )
    assert "endpoint_timeout_min" not in t.model_fields_set


def test_target_config_iri_missing_endpoint_raises():
    with pytest.raises(ValueError, match="requires 'endpoint'"):
        TargetConfig(kind="iri", resource_id="odo", account="fus183", home_dir="/home/x")


def test_target_config_iri_missing_resource_id_raises():
    with pytest.raises(ValueError, match="requires 'resource_id'"):
        TargetConfig(kind="iri", endpoint="olcf", account="fus183", home_dir="/home/x")


def test_target_config_iri_missing_home_dir_raises():
    with pytest.raises(ValueError, match="requires 'home_dir'"):
        TargetConfig(kind="iri", endpoint="olcf", resource_id="odo", account="fus183")


def test_target_config_psij_missing_edge_name_allowed_standalone():
    """TargetConfig alone no longer enforces 'edge_name' for kind='psij' — whether it's required
    depends on RemoteConfig.embedded, a sibling field this model can't see.

    See RemoteConfig-level tests below for the actual enforcement (edge_name required unless
    embedded=True).
    """
    t = TargetConfig(kind="psij", account="amsc007")
    assert t.edge_name is None


def test_target_config_psij_valid():
    t = TargetConfig(kind="psij", edge_name="perlmutter-login", account="amsc007")
    assert t.edge_name == "perlmutter-login"


def test_target_config_missing_account_raises():
    with pytest.raises(ValueError, match="requires 'account'"):
        TargetConfig(kind="psij", edge_name="login1")


def test_target_config_extra_field_rejected():
    with pytest.raises(ValidationError):
        TargetConfig(kind="psij", edge_name="login1", account="a", bogus_field="x")


def test_target_config_bad_tunnel_value_rejected():
    with pytest.raises(ValidationError):
        TargetConfig(kind="psij", edge_name="login1", account="a", tunnel="yes")


def test_remote_target_roundtrip_from_yaml(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          backends: [dragon]
          target:
            kind: sfapi
            endpoint: nersc
            resource_id: perlmutter
            account: amsc007
            queue_name: debug
            walltime_min: 30
            n_nodes: 1
            constraint: cpu
            home_dir: /global/u2/m/merzky
    """)
    p = tmp_path / "remote.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.remote.target is not None
    assert cfg.remote.target.kind == "sfapi"
    assert cfg.remote.target.endpoint == "nersc"
    assert cfg.remote.target.constraint == "cpu"


def test_remote_embedded_psij_roundtrip_from_yaml(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: sequential_active_learner
          max_iter: 1
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
          embedded: true
          target:
            kind: psij
            account: amsc007
            queue_name: debug
    """)
    p = tmp_path / "embedded.yaml"
    p.write_text(yaml)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.remote.embedded is True
    assert cfg.remote.target.kind == "psij"
    assert cfg.remote.target.edge_name is None


def test_remote_without_target_still_valid(tmp_path):
    """Specs written before remote.target existed (e.g. the m3dc1 use case) must keep validating
    unchanged."""
    p = tmp_path / "spec.yaml"
    p.write_text(SEQ_YAML)
    cfg = WorkflowConfig.from_yaml(p)
    assert cfg.remote.target is None


def test_parallel_learners_mixed_types(tmp_path):
    yaml = textwrap.dedent("""\
        learner:
          type: parallel_active_learner
        learners:
          - label: a
            simulation:
              type: shell
              command: python sim.py
            training:
              type: python
              function: mymod:train_a
            active_learn:
              type: python
              function: mymod:select
          - label: b
            simulation:
              type: python
              function: mymod:sim_b
            training:
              type: python
              function: mymod:train_b
            active_learn:
              type: python
              function: mymod:select
        stop_criterion:
          metric: r2
          threshold: 0.9
          evaluator:
            type: python
            function: mymod:eval
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="mixed types"):
        WorkflowConfig.from_yaml(p)
