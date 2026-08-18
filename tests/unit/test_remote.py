"""Unit tests for rose.remote — bootstrap/execute/teardown orchestration for
`rose run --remote`. radical.orbit's EndpointRuntime is mocked throughout;
these tests never touch a real broker or HPC system.
"""

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

import rose.remote as remote
from rose.spec import load_spec

SFAPI_TARGET_YAML = textwrap.dedent("""\
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
      target:
        kind: sfapi
        endpoint: nersc
        resource_id: perlmutter
        account: amsc007
        home_dir: /global/u2/m/merzky
""")

EMBEDDED_PSIJ_TARGET_YAML = textwrap.dedent("""\
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
""")

NO_TARGET_YAML = textwrap.dedent("""\
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
""")


def _make_spec(tmp_path, yaml):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml)
    return load_spec(p)


# ── _wait_for_endpoint ─────────────────────────────────────────────────────


def test_wait_for_endpoint_succeeds_when_name_appears():
    rt = MagicMock()
    rt.topology.side_effect = [{}, {}, {"ep-1": {}}]
    remote._wait_for_endpoint(rt, "ep-1", timeout=10, poll=0)


def test_wait_for_endpoint_times_out():
    rt = MagicMock()
    rt.topology.return_value = {}
    with pytest.raises(TimeoutError, match="ep-1"):
        remote._wait_for_endpoint(rt, "ep-1", timeout=0.05, poll=0.01)


# ── credential reading ───────────────────────────────────────────────────────


def test_read_iri_token_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="token file missing"):
        remote._read_iri_token("olcf")


def test_read_iri_token_reads_stripped_content(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    (tmp_path / "token_olcf").write_text("  secret-token  \n")
    assert remote._read_iri_token("olcf") == "secret-token"


def test_read_sfapi_credentials_missing_client_id_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    monkeypatch.delenv("SFAPI_CLIENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="SFAPI_CLIENT_ID"):
        remote._read_sfapi_credentials("nersc")


def test_read_sfapi_credentials_missing_key_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    monkeypatch.setenv("SFAPI_CLIENT_ID", "abc123")
    with pytest.raises(RuntimeError, match="private key not found"):
        remote._read_sfapi_credentials("nersc")


def test_read_sfapi_credentials_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    monkeypatch.setenv("SFAPI_CLIENT_ID", "abc123")
    (tmp_path / "sfapi_key_nersc.pem").write_text("-----BEGIN KEY-----\n...")
    client_id, key = remote._read_sfapi_credentials("nersc")
    assert client_id == "abc123"
    assert key.startswith("-----BEGIN KEY-----")


# ── teardown ──────────────────────────────────────────────────────────────


def test_teardown_none_is_noop():
    remote._teardown(None)  # must not raise


def test_teardown_iri_cancels_and_disconnects():
    instance = MagicMock()
    connect_client = MagicMock()
    created = {
        "kind": "sfapi", "client": instance, "connect_client": connect_client,
        "endpoint_key": "nersc", "resource_id": "perlmutter", "job_id": "j1",
    }
    remote._teardown(created)
    instance.cancel_job.assert_called_once_with("perlmutter", "j1")
    connect_client.disconnect.assert_called_once_with("nersc")


def test_teardown_psij_cancels_only():
    psij = MagicMock()
    created = {"kind": "psij", "client": psij, "job_id": "j1",
              "parent_edge": "login1", "endpoint_name": "rose-login1-abc"}
    remote._teardown(created)
    psij.cancel_job.assert_called_once_with("j1")


def test_teardown_swallows_cancel_errors():
    instance = MagicMock()
    instance.cancel_job.side_effect = RuntimeError("already gone")
    connect_client = MagicMock()
    created = {
        "kind": "iri", "client": instance, "connect_client": connect_client,
        "endpoint_key": "olcf", "resource_id": "odo", "job_id": "j1",
    }
    remote._teardown(created)  # must not raise
    connect_client.disconnect.assert_called_once_with("olcf")


# ── run_remote ────────────────────────────────────────────────────────────


async def test_run_remote_requires_target(tmp_path):
    spec = _make_spec(tmp_path, NO_TARGET_YAML)
    with pytest.raises(RuntimeError, match="remote.target"):
        await remote.run_remote(spec)


async def test_run_remote_psij_edge_not_in_topology_raises(tmp_path, monkeypatch):
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
          target:
            kind: psij
            edge_name: login1
            account: amsc007
    """)
    spec = _make_spec(tmp_path, yaml)

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://broker:8000"
    fake_rt.topology.return_value = {}
    monkeypatch.setattr(remote, "_teardown", MagicMock())

    import radical.orbit
    monkeypatch.setattr(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt))

    with pytest.raises(RuntimeError, match="login1"):
        await remote.run_remote(spec)
    fake_rt.stop.assert_called_once()


async def test_run_remote_sfapi_end_to_end(tmp_path, monkeypatch):
    spec = _make_spec(tmp_path, SFAPI_TARGET_YAML)
    monkeypatch.setattr(remote, "AMSC_DIR", tmp_path)
    monkeypatch.setenv("SFAPI_CLIENT_ID", "abc123")
    (tmp_path / "sfapi_key_nersc.pem").write_text("-----BEGIN KEY-----\n...")

    instance = MagicMock()
    instance.submit_job.return_value = {"job_id": "job-42"}

    connect_client = MagicMock()
    connect_client.connect.return_value = instance

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://broker:8000"
    fake_rt.get_plugin.return_value = connect_client
    # topology never contains the bootstrapped endpoint (loop exits fast via patched wait)

    monkeypatch.setattr(remote, "_wait_for_endpoint", MagicMock())
    fake_workflow = AsyncMock()
    monkeypatch.setattr(type(spec), "workflow", property(lambda self: fake_workflow))

    import radical.orbit
    monkeypatch.setattr(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt))

    await remote.run_remote(spec)

    fake_rt.get_plugin.assert_called_once_with("broker", "sfapi_connect")
    connect_client.connect.assert_called_once_with("nersc", "abc123",
                                                    "-----BEGIN KEY-----\n...")
    instance.submit_job.assert_called_once()
    resource_id, job_spec = instance.submit_job.call_args[0]
    assert resource_id == "perlmutter"
    assert job_spec["attributes"]["account"] == "amsc007"

    fake_workflow.assert_awaited_once()
    call_args = fake_workflow.await_args[0]
    assert call_args[0] == "https://broker:8000"

    instance.cancel_job.assert_called_once()
    connect_client.disconnect.assert_called_once_with("nersc")
    fake_rt.stop.assert_called_once()


# ── _launch_psij: embedded vs standalone plugin/home_dir source ───────────


def test_launch_psij_embedded_uses_broker_and_local_home(monkeypatch):
    psij = MagicMock()
    psij.submit_tunneled.return_value = {"job_id": "j1", "endpoint_name": "rose-broker-abc"}

    rt = MagicMock()
    rt.get_plugin.return_value = psij

    from rose.spec.schema import TargetConfig
    target = TargetConfig(kind="psij", account="amsc007")

    class _FakeHomePath:
        @staticmethod
        def home():
            from pathlib import Path as _RealPath
            return _RealPath("/home/local")

    # Rebind the name `Path` inside rose.remote only — must not touch the
    # real pathlib.Path class, which other code/tests still rely on.
    monkeypatch.setattr(remote, "Path", _FakeHomePath)

    created = remote._launch_psij(rt, target, "https://broker:8000", embedded=True)

    rt.get_plugin.assert_called_once_with("broker", "psij")
    job_spec = psij.submit_tunneled.call_args[0][0]
    assert job_spec["executable"] == "/home/local/.amsc/ve/bin/radical-orbit-endpoint-wrapper.sh"
    assert created["endpoint_name"] == "rose-broker-abc"


def test_launch_psij_embedded_prefers_explicit_home_dir():
    psij = MagicMock()
    psij.submit_tunneled.return_value = {"job_id": "j1", "endpoint_name": "ep"}
    rt = MagicMock()
    rt.get_plugin.return_value = psij

    from rose.spec.schema import TargetConfig
    target = TargetConfig(kind="psij", account="amsc007", home_dir="/custom/home")

    remote._launch_psij(rt, target, "https://broker:8000", embedded=True)

    job_spec = psij.submit_tunneled.call_args[0][0]
    assert job_spec["executable"] == "/custom/home/.amsc/ve/bin/radical-orbit-endpoint-wrapper.sh"


def test_launch_psij_non_embedded_uses_edge_and_remote_sysinfo():
    psij = MagicMock()
    psij.submit_tunneled.return_value = {"job_id": "j1", "endpoint_name": "ep"}
    sysinfo = MagicMock()
    sysinfo.homedir.return_value = "/remote/home"

    rt = MagicMock()
    rt.get_plugin.side_effect = lambda edge, name: {"psij": psij, "sysinfo": sysinfo}[name]

    from rose.spec.schema import TargetConfig
    target = TargetConfig(kind="psij", edge_name="login1", account="amsc007")

    remote._launch_psij(rt, target, "https://broker:8000", embedded=False)

    rt.get_plugin.assert_any_call("login1", "psij")
    rt.get_plugin.assert_any_call("login1", "sysinfo")
    job_spec = psij.submit_tunneled.call_args[0][0]
    assert job_spec["executable"] == "/remote/home/.amsc/ve/bin/radical-orbit-endpoint-wrapper.sh"


# ── run_remote: embedded broker ─────────────────────────────────────────────


async def test_run_remote_embedded_constructs_and_tears_down_broker(tmp_path, monkeypatch):
    spec = _make_spec(tmp_path, EMBEDDED_PSIJ_TARGET_YAML)

    psij = MagicMock()
    psij.submit_tunneled.return_value = {"job_id": "j1", "endpoint_name": "ep-1"}

    stop_order = []

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://embedded:8000"
    fake_rt.get_plugin.return_value = psij
    fake_rt.stop.side_effect = lambda: stop_order.append("rt")

    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"
    fake_eb.stop.side_effect = lambda: stop_order.append("eb")

    monkeypatch.setattr(remote, "_wait_for_endpoint", MagicMock())
    fake_workflow = AsyncMock()
    monkeypatch.setattr(type(spec), "workflow", property(lambda self: fake_workflow))

    import radical.orbit
    import radical.orbit.embedded
    monkeypatch.setattr(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt))
    fake_embedded_broker_cls = MagicMock(return_value=fake_eb)
    monkeypatch.setattr(radical.orbit.embedded, "EmbeddedBroker", fake_embedded_broker_cls)

    await remote.run_remote(spec)

    fake_embedded_broker_cls.assert_called_once_with(plugins="psij")
    fake_eb.start.assert_called_once()
    radical.orbit.EndpointRuntime.assert_called_once_with(broker_url="https://embedded:8000")
    fake_rt.get_plugin.assert_called_once_with("broker", "psij")

    # teardown order: runtime stops before the embedded broker
    assert stop_order == ["rt", "eb"]


async def test_run_remote_embedded_stops_broker_if_runtime_start_fails(tmp_path, monkeypatch):
    spec = _make_spec(tmp_path, EMBEDDED_PSIJ_TARGET_YAML)

    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    fake_rt = MagicMock()
    fake_rt.start.side_effect = RuntimeError("registration failed")

    import radical.orbit
    import radical.orbit.embedded
    monkeypatch.setattr(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt))
    monkeypatch.setattr(radical.orbit.embedded, "EmbeddedBroker",
                        MagicMock(return_value=fake_eb))

    with pytest.raises(RuntimeError, match="registration failed"):
        await remote.run_remote(spec)

    fake_eb.start.assert_called_once()
    fake_eb.stop.assert_called_once()
