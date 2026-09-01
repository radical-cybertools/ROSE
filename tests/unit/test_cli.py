"""Unit tests for rose.cli — currently just `_run_setup`'s --wait-timeout default-vs-explicit
warning logic.

`run_setup_wizard` itself is mocked out;
see test_bootstrap.py for its own behavior.
"""

from unittest.mock import MagicMock, patch

import pytest

import rose.cli as cli


def test_run_setup_warns_and_uses_default_when_timeout_omitted(capsys):
    with patch("rose.remote.run_setup_wizard", MagicMock(return_value=True)) as wizard:
        with pytest.raises(SystemExit) as exc:
            cli._run_setup(None)

    assert exc.value.code == 0
    wizard.assert_called_once_with(wait_timeout=300.0)
    out = capsys.readouterr().out
    assert "default wait timeout of 300s" in out
    assert "--wait-timeout SECONDS" in out


def test_run_setup_silent_when_timeout_explicit(capsys):
    with patch("rose.remote.run_setup_wizard", MagicMock(return_value=True)) as wizard:
        with pytest.raises(SystemExit) as exc:
            cli._run_setup(600.0)

    assert exc.value.code == 0
    wizard.assert_called_once_with(wait_timeout=600.0)
    assert "default wait timeout" not in capsys.readouterr().out


def test_run_setup_exit_code_reflects_failure():
    with patch("rose.remote.run_setup_wizard", MagicMock(return_value=False)):
        with pytest.raises(SystemExit) as exc:
            cli._run_setup(600.0)

    assert exc.value.code == 1
