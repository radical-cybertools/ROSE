"""Unit tests for rose.bootstrap — the `rose setup` interactive wizard.

All interactive input, radical.orbit connectivity, and subprocess calls are mocked; these tests
never touch a real broker, HPC system, or terminal.
"""

from unittest.mock import MagicMock, patch

import rose.remote.bootstrap as bootstrap

# ── prompt helpers ──────────────────────────────────────────────────────────


def test_ask_returns_default_on_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert bootstrap.ask("prompt", "default") == "default"


def test_ask_returns_typed_value(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "  typed  ")
    assert bootstrap.ask("prompt") == "typed"


def test_ask_int_retries_on_non_numeric(monkeypatch, capsys):
    answers = iter(["not-a-number", "42"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert bootstrap.ask_int("prompt", 1) == 42
    assert "not a number" in capsys.readouterr().out


def test_confirm_default_true_on_empty(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert bootstrap.confirm("prompt", default=True) is True


def test_confirm_parses_yes_no(monkeypatch):
    for raw, expected in [("y", True), ("yes", True), ("n", False), ("no", False)]:
        monkeypatch.setattr("builtins.input", lambda _, r=raw: r)
        assert bootstrap.confirm("prompt", default=True) == expected


def test_confirm_retries_on_garbage(monkeypatch, capsys):
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert bootstrap.confirm("prompt") is True
    assert "please answer y or n" in capsys.readouterr().out


def test_choose_returns_selected_value(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    result = bootstrap.choose("pick one", [("first", "a"), ("second", "b")])
    assert result == "b"


def test_choose_retries_out_of_range(monkeypatch, capsys):
    answers = iter(["9", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    result = bootstrap.choose("pick one", [("first", "a"), ("second", "b")])
    assert result == "a"
    assert "please enter a number" in capsys.readouterr().out


# ── broker credentials ──────────────────────────────────────────────────────


def test_ensure_broker_credentials_skips_when_present(tmp_path, monkeypatch, capsys):
    for name in ("broker_cert.pem", "broker_key.pem", "broker.token"):
        (tmp_path / name).write_text("x")
    monkeypatch.setattr(bootstrap, "ORBIT_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "CERT_PATH", tmp_path / "broker_cert.pem")
    monkeypatch.setattr(bootstrap, "KEY_PATH", tmp_path / "broker_key.pem")
    monkeypatch.setattr(bootstrap, "TOKEN_PATH", tmp_path / "broker.token")

    assert bootstrap.ensure_broker_credentials() is True
    assert "already present" in capsys.readouterr().out


def test_ensure_broker_credentials_auto_generates(tmp_path, monkeypatch):
    cert, key, token = (
        tmp_path / "broker_cert.pem",
        tmp_path / "broker_key.pem",
        tmp_path / "broker.token",
    )
    monkeypatch.setattr(bootstrap, "ORBIT_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "CERT_PATH", cert)
    monkeypatch.setattr(bootstrap, "KEY_PATH", key)
    monkeypatch.setattr(bootstrap, "TOKEN_PATH", token)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    def fake_run(args, **kwargs):
        # Simulate `openssl req ... -keyout key -out cert`
        cert.write_text("cert")
        key.write_text("key")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        assert bootstrap.ensure_broker_credentials() is True

    assert cert.exists() and key.exists() and token.exists()


def test_ensure_broker_credentials_falls_back_on_subprocess_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "ORBIT_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "CERT_PATH", tmp_path / "broker_cert.pem")
    monkeypatch.setattr(bootstrap, "KEY_PATH", tmp_path / "broker_key.pem")
    monkeypatch.setattr(bootstrap, "TOKEN_PATH", tmp_path / "broker.token")

    answers = iter(["y", "n"])  # "generate automatically?" -> yes; "completed steps?" -> no
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    with patch("subprocess.run", side_effect=FileNotFoundError("no openssl")):
        assert bootstrap.ensure_broker_credentials() is False

    out = capsys.readouterr().out
    assert "didn't work" in out
    assert "openssl req" in out  # TLS_RECIPE printed
    assert "secrets.token_urlsafe" in out  # TOKEN_RECIPE printed


def test_ensure_broker_credentials_user_declines_automation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "ORBIT_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "CERT_PATH", tmp_path / "broker_cert.pem")
    monkeypatch.setattr(bootstrap, "KEY_PATH", tmp_path / "broker_key.pem")
    monkeypatch.setattr(bootstrap, "TOKEN_PATH", tmp_path / "broker.token")

    answers = iter(["n", "n"])  # decline automation, then decline "have you done it"
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap.ensure_broker_credentials() is False
    assert "openssl req" in capsys.readouterr().out


# ── PsiJ branch: scheduler detection ────────────────────────────────────────


def test_run_psij_setup_no_scheduler_prints_instructions_and_returns_false(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "")  # blank login host

    assert bootstrap.run_psij_setup() is False
    out = capsys.readouterr().out
    assert "doesn't have SLURM or PBS" in out
    assert "rose setup" in out


def test_has_scheduler_true_when_sbatch_present(monkeypatch):
    monkeypatch.setattr(
        bootstrap.shutil, "which", lambda name: "/usr/bin/sbatch" if name == "sbatch" else None
    )
    assert bootstrap._has_scheduler() is True


def test_has_scheduler_false_when_neither_present(monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
    assert bootstrap._has_scheduler() is False


def test_run_psij_setup_verifies_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_has_scheduler", lambda: True)
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)

    answers = iter(["amsc007", "debug"])  # account, queue
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    psij = MagicMock()
    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.get_plugin.return_value = psij
    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    import radical.orbit
    import radical.orbit.embedded

    with (
        patch.object(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt)),
        patch.object(radical.orbit.embedded, "EmbeddedBroker", MagicMock(return_value=fake_eb)),
    ):
        assert bootstrap.run_psij_setup() is True

    fake_rt.get_plugin.assert_called_once_with("broker", "psij")
    fake_rt.stop.assert_called_once()
    fake_eb.stop.assert_called_once()
    out = capsys.readouterr().out
    assert "Setup verified" in out
    assert "embedded: true" in out
    assert "kind: psij" in out


def test_run_psij_setup_reports_failure_on_connect_error(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_has_scheduler", lambda: True)
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    answers = iter(["amsc007", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    import radical.orbit.embedded

    with patch.object(
        radical.orbit.embedded, "EmbeddedBroker", MagicMock(side_effect=RuntimeError("port in use"))
    ):
        assert bootstrap.run_psij_setup() is False

    out = capsys.readouterr().out
    assert "Could not verify PsiJ" in out
    assert "did NOT succeed" in out


# ── IRI/SFAPI branch ─────────────────────────────────────────────────────────


def _iri_answers(
    facility_choice="1", account="amsc007", home="/home/x", queue="", walltime="30", token="tok123"
):
    return iter([facility_choice, account, home, queue, walltime, token])


def test_run_iri_sfapi_setup_end_to_end_olcf(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    # facility=2 (OLCF -> iri), account, home_dir, queue(blank), walltime(default),
    # token entered directly
    answers = iter(["2", "fus183", "/ccs/home/x", "", "30", "tok-olcf"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    instance = MagicMock()
    instance.submit_job.return_value = {"job_id": "env-check-1", "state": "completed"}
    instance.get_job_status.return_value = {"state": "completed"}
    instance.cancel_job.return_value = {}

    connect_client = MagicMock()
    connect_client.connect.return_value = instance

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://embedded:8000"
    fake_rt.get_plugin.return_value = connect_client
    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    monkeypatch.setattr(bootstrap._remote, "_wait_for_endpoint", MagicMock())

    import radical.orbit
    import radical.orbit.embedded

    with (
        patch.object(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt)),
        patch.object(radical.orbit.embedded, "EmbeddedBroker", MagicMock(return_value=fake_eb)),
    ):
        assert bootstrap.run_iri_sfapi_setup() is True

    # connect() is called twice (an initial auth check, then again inside
    # _launch_iri_or_sfapi for the real bootstrap job) — idempotent on the
    # server side (IRIConnectClient.connect() just refreshes the token in
    # place on reconnect), so this is harmless, expected duplication, not a
    # bug — assert both calls used the right credential rather than "once".
    for call in connect_client.connect.call_args_list:
        assert call.args == ("olcf", "tok-olcf")
    assert connect_client.connect.call_count == 2
    assert instance.submit_job.call_count == 2  # env-check job + real bootstrap job
    instance.cancel_job.assert_called_once()
    fake_rt.stop.assert_called_once()
    fake_eb.stop.assert_called_once()

    out = capsys.readouterr().out
    assert "Setup verified" in out
    assert "kind: iri" in out
    assert "endpoint: olcf" in out

    # default wait_timeout (300s) reaches both HPC-wait points
    bootstrap._remote._wait_for_endpoint.assert_called_once()
    assert bootstrap._remote._wait_for_endpoint.call_args.kwargs["timeout"] == 300.0
    assert instance.get_job_status.call_count >= 1  # _poll_job actually polled


def test_run_iri_sfapi_setup_custom_wait_timeout_reaches_both_hpc_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    answers = iter(["2", "fus183", "/ccs/home/x", "", "30", "tok-olcf"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    instance = MagicMock()
    instance.submit_job.return_value = {"job_id": "env-check-1", "state": "completed"}
    instance.get_job_status.return_value = {"state": "completed"}

    connect_client = MagicMock()
    connect_client.connect.return_value = instance

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://embedded:8000"
    fake_rt.get_plugin.return_value = connect_client
    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    monkeypatch.setattr(bootstrap._remote, "_wait_for_endpoint", MagicMock())
    fake_poll_job = MagicMock(return_value="completed")
    monkeypatch.setattr(bootstrap, "_poll_job", fake_poll_job)

    import radical.orbit
    import radical.orbit.embedded

    with (
        patch.object(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt)),
        patch.object(radical.orbit.embedded, "EmbeddedBroker", MagicMock(return_value=fake_eb)),
    ):
        assert bootstrap.run_iri_sfapi_setup(wait_timeout=45.0) is True

    assert bootstrap._remote._wait_for_endpoint.call_args.kwargs["timeout"] == 45.0
    assert fake_poll_job.call_args.kwargs["timeout"] == 45.0


def test_run_iri_sfapi_setup_nersc_classic_iri_via_globus(monkeypatch, tmp_path, capsys):
    """Selecting NERSC then 'Classic IRI via Globus login' must route through kind='iri' (not
    sfapi), reusing the same generic iri code path OLCF uses."""
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    # Pre-seed the token file so _ensure_iri_token short-circuits without
    # needing to exercise the Globus auto-fetch flow (covered separately).
    (tmp_path / "token_nersc").write_text("tok-nersc-globus\n")

    # facility=1 (NERSC), auth sub-choice=2 (classic IRI via Globus),
    # account, home_dir, queue(blank), walltime(default)
    answers = iter(["1", "2", "amsc007", "/global/u2/m/x", "", "30"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    instance = MagicMock()
    instance.submit_job.return_value = {"job_id": "env-check-1", "state": "completed"}
    instance.get_job_status.return_value = {"state": "completed"}
    instance.cancel_job.return_value = {}

    connect_client = MagicMock()
    connect_client.connect.return_value = instance

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.broker_url = "https://embedded:8000"
    fake_rt.get_plugin.return_value = connect_client
    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    monkeypatch.setattr(bootstrap._remote, "_wait_for_endpoint", MagicMock())

    import radical.orbit
    import radical.orbit.embedded

    with (
        patch.object(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt)),
        patch.object(radical.orbit.embedded, "EmbeddedBroker", MagicMock(return_value=fake_eb)),
    ):
        assert bootstrap.run_iri_sfapi_setup() is True

    for call in connect_client.connect.call_args_list:
        assert call.args == ("nersc", "tok-nersc-globus")
    fake_rt.stop.assert_called_once()
    fake_eb.stop.assert_called_once()

    out = capsys.readouterr().out
    assert "Setup verified" in out
    assert "kind: iri" in out
    assert "endpoint: nersc" in out


def test_run_iri_sfapi_setup_nersc_default_sfapi_path_unaffected(monkeypatch, tmp_path):
    """Selecting NERSC then SFAPI (the default/recommended option) must still call the SFAPI
    credential path, never the IRI token path."""
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    sfapi_calls = []
    iri_calls = []
    monkeypatch.setattr(
        bootstrap, "_ensure_sfapi_credentials", lambda f: sfapi_calls.append(f) or False
    )
    monkeypatch.setattr(bootstrap, "_ensure_iri_token", lambda f: iri_calls.append(f) or False)

    # facility=1 (NERSC), auth sub-choice=1 (SFAPI), account, home_dir,
    # queue(blank), walltime(default) — credential collection then fails
    # (mocked False above) so the test stays short: only the routing matters.
    answers = iter(["1", "1", "amsc007", "/global/u2/m/x", "", "30"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap.run_iri_sfapi_setup() is False
    assert sfapi_calls == ["nersc"]
    assert iri_calls == []


def test_run_iri_sfapi_setup_env_check_failure_offers_fallback_and_aborts(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    # facility=2 (OLCF), account, home, queue(blank), walltime(default), token,
    # then "n" to decline the manual-fallback confirmation
    answers = iter(["2", "fus183", "/ccs/home/x", "", "30", "tok-olcf", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    instance = MagicMock()
    instance.submit_job.return_value = {"job_id": "env-check-1"}
    instance.get_job_status.return_value = {"state": "failed"}

    connect_client = MagicMock()
    connect_client.connect.return_value = instance

    fake_rt = MagicMock()
    fake_rt.start.return_value = fake_rt
    fake_rt.get_plugin.return_value = connect_client
    fake_eb = MagicMock()
    fake_eb.url = "https://embedded:8000"

    import radical.orbit
    import radical.orbit.embedded

    with (
        patch.object(radical.orbit, "EndpointRuntime", MagicMock(return_value=fake_rt)),
        patch.object(radical.orbit.embedded, "EmbeddedBroker", MagicMock(return_value=fake_eb)),
    ):
        assert bootstrap.run_iri_sfapi_setup() is False

    out = capsys.readouterr().out
    assert "not success" in out
    assert "ssh " in out  # manual fallback printed
    assert "did NOT succeed" in out
    fake_rt.stop.assert_called_once()
    fake_eb.stop.assert_called_once()


def test_run_iri_sfapi_setup_connect_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bootstrap, "ensure_broker_credentials", lambda: True)
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    answers = iter(["2", "fus183", "/ccs/home/x", "", "30", "tok-olcf"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    import radical.orbit
    import radical.orbit.embedded

    with patch.object(
        radical.orbit.embedded, "EmbeddedBroker", MagicMock(side_effect=RuntimeError("no cert"))
    ):
        assert bootstrap.run_iri_sfapi_setup() is False

    out = capsys.readouterr().out
    assert "Could not connect" in out
    assert "did NOT succeed" in out


# ── SFAPI credential collection ──────────────────────────────────────────────


def test_ensure_sfapi_credentials_uses_env_var(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    monkeypatch.setenv("SFAPI_CLIENT_ID", "abc123")
    (tmp_path / "sfapi_key_nersc.pem").write_text("-----BEGIN KEY-----")

    assert bootstrap._ensure_sfapi_credentials("nersc") is True
    assert "using the SFAPI credentials" in capsys.readouterr().out


def test_ensure_sfapi_credentials_writes_client_id_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    monkeypatch.delenv("SFAPI_CLIENT_ID", raising=False)
    (tmp_path / "sfapi_key_nersc.pem").write_text("-----BEGIN KEY-----")

    answers = iter(["abc123"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_sfapi_credentials("nersc") is True
    assert (tmp_path / "sfapi_client_id_nersc").read_text().strip() == "abc123"


def test_ensure_sfapi_credentials_gives_up_when_declined(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    monkeypatch.delenv("SFAPI_CLIENT_ID", raising=False)

    answers = iter(["", "n"])  # no client id pasted, decline retry
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_sfapi_credentials("nersc") is False


# ── IRI token collection ─────────────────────────────────────────────────────


def test_ensure_iri_token_existing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    (tmp_path / "token_olcf").write_text("existing-token")

    assert bootstrap._ensure_iri_token("olcf") is True
    assert "using the IRI token" in capsys.readouterr().out


def test_ensure_iri_token_pasted_value_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    answers = iter(["pasted-token"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_iri_token("olcf") is True
    assert (tmp_path / "token_olcf").read_text().strip() == "pasted-token"


def test_ensure_iri_token_gives_up_when_declined(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)

    answers = iter(["", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_iri_token("olcf") is False


def test_ensure_iri_token_olcf_never_offers_globus(tmp_path, monkeypatch):
    """OLCF isn't a Globus-authenticated facility — the automatic-fetch offer must never appear for
    it, only the manual-paste flow."""
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "_try_automatic_nersc_iri_token",
        MagicMock(side_effect=AssertionError("must not be called for olcf")),
    )
    answers = iter(["pasted-token"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_iri_token("olcf") is True


def test_ensure_iri_token_nersc_uses_automatic_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    path = tmp_path / "token_nersc"

    def fake_auto(p):
        p.write_text("auto-fetched\n")
        return True

    monkeypatch.setattr(bootstrap, "_try_automatic_nersc_iri_token", fake_auto)

    assert bootstrap._ensure_iri_token("nersc") is True
    assert path.read_text().strip() == "auto-fetched"
    assert "using the IRI token" in capsys.readouterr().out


def test_ensure_iri_token_nersc_falls_back_to_manual_when_auto_declined(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap._remote, "AMSC_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "_try_automatic_nersc_iri_token", lambda _p: False)

    answers = iter(["pasted-manually"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert bootstrap._ensure_iri_token("nersc") is True
    assert (tmp_path / "token_nersc").read_text().strip() == "pasted-manually"


# ── job polling ──────────────────────────────────────────────────────────────


def test_poll_job_returns_terminal_state():
    instance = MagicMock()
    instance.get_job_status.return_value = {"state": "COMPLETED"}
    assert bootstrap._poll_job(instance, "perlmutter", "j1", timeout=5, poll=0) == "completed"


def test_poll_job_times_out():
    instance = MagicMock()
    instance.get_job_status.return_value = {"state": "running"}
    assert bootstrap._poll_job(instance, "perlmutter", "j1", timeout=0.05, poll=0.01) == "timeout"


def test_job_state_handles_nested_status_dict():
    assert bootstrap._job_state({"status": {"state": "FAILED"}}) == "failed"


def test_job_state_handles_flat_state():
    assert bootstrap._job_state({"state": "Running"}) == "running"


# ── wizard entry point branch selection ──────────────────────────────────────


def test_run_setup_wizard_dispatches_to_iri(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    called = {}

    def fake_iri(wait_timeout=300.0):
        called["iri"] = wait_timeout
        return True

    monkeypatch.setattr(bootstrap, "run_iri_sfapi_setup", fake_iri)
    monkeypatch.setattr(
        bootstrap, "run_psij_setup", lambda: called.setdefault("psij", True) or True
    )

    assert bootstrap.run_setup_wizard() is True
    assert called == {"iri": 300.0}


def test_run_setup_wizard_dispatches_to_psij(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    called = {}

    def fake_iri(wait_timeout=300.0):
        called["iri"] = wait_timeout
        return True

    monkeypatch.setattr(bootstrap, "run_iri_sfapi_setup", fake_iri)
    monkeypatch.setattr(
        bootstrap, "run_psij_setup", lambda: called.setdefault("psij", True) or True
    )

    assert bootstrap.run_setup_wizard() is True
    assert called == {"psij": True}


def test_run_setup_wizard_threads_custom_wait_timeout_to_iri(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    called = {}

    def fake_iri(wait_timeout=300.0):
        called["iri"] = wait_timeout
        return True

    monkeypatch.setattr(bootstrap, "run_iri_sfapi_setup", fake_iri)

    assert bootstrap.run_setup_wizard(wait_timeout=45.0) is True
    assert called == {"iri": 45.0}
