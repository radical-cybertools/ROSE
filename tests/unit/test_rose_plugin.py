"""Unit tests for the ROSE Edge Plugin.

Tests the RoseSession, RoseClient, and WorkflowLoader classes.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from rose.service.api.rest import PluginRose, RoseClient, RoseSession
from rose.service.manager import WorkflowLoader
from rose.service.models import Workflow, WorkflowState

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Create a mock WorkflowEngine."""
    engine = AsyncMock()
    engine.shutdown = AsyncMock()
    return engine


@pytest.fixture
def rose_session():
    """Create a RoseSession for testing."""
    session = RoseSession(sid="test-session-001")
    return session


@pytest.fixture
def sample_workflow_yaml(tmp_path):
    """Create a sample workflow YAML file."""
    wf_content = """
learner:
  type: SequentialActiveLearner

components:
  simulation:
    type: script
    path: /bin/echo
    config:
      args: ["sim"]
  training:
    type: script
    path: /bin/echo
    config:
      args: ["train"]
  active_learn:
    type: script
    path: /bin/echo
    config:
      args: ["al"]

config:
  max_iterations: 2
  work_dir: /tmp/rose_test
"""
    wf_file = tmp_path / "test_workflow.yaml"
    wf_file.write_text(wf_content)
    return str(wf_file)


# -----------------------------------------------------------------------------
# WorkflowLoader Tests
# -----------------------------------------------------------------------------


class TestWorkflowLoader:
    """Tests for WorkflowLoader class."""

    def test_load_yaml_valid(self, sample_workflow_yaml):
        """Test loading a valid YAML workflow file."""
        wf_def = WorkflowLoader.load_yaml(sample_workflow_yaml)

        assert "learner" in wf_def
        assert wf_def["learner"]["type"] == "SequentialActiveLearner"
        assert "components" in wf_def
        assert "simulation" in wf_def["components"]
        assert "config" in wf_def
        assert wf_def["config"]["max_iterations"] == 2

    def test_load_yaml_file_not_found(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            WorkflowLoader.load_yaml("/nonexistent/path/workflow.yaml")

    def test_create_learner_sequential(self, sample_workflow_yaml, mock_engine):
        """Test creating a SequentialActiveLearner from YAML."""
        wf_def = WorkflowLoader.load_yaml(sample_workflow_yaml)
        learner, config = WorkflowLoader.create_learner("wf.test001", wf_def, mock_engine)

        from rose.al.active_learner import SequentialActiveLearner

        assert isinstance(learner, SequentialActiveLearner)
        assert learner.learner_id == "wf.test001"

    def test_import_function_valid(self):
        """Test importing a valid function path."""
        func = WorkflowLoader._import_function("os.path.exists")
        import os

        assert func == os.path.exists

    def test_import_function_invalid(self):
        """Test importing an invalid function path raises error."""
        with pytest.raises(ImportError):
            WorkflowLoader._import_function("nonexistent.module.func")

    def test_create_script_task_factory(self):
        """Test creating a script task factory."""
        factory = WorkflowLoader._create_script_task_factory("/bin/echo")

        # The factory should be an async function
        assert asyncio.iscoroutinefunction(factory)


# -----------------------------------------------------------------------------
# RoseSession Tests
# -----------------------------------------------------------------------------


class TestRoseSession:
    """Tests for RoseSession class."""

    def test_init(self, rose_session):
        """Test RoseSession initialization."""
        assert rose_session.sid == "test-session-001"
        assert rose_session.is_active
        assert rose_session._workflows == {}
        assert rose_session._engine is None

    @pytest.mark.asyncio
    async def test_ensure_engine(self, rose_session):
        """Test lazy engine initialization."""
        with (
            patch("rose.service.api.rest.LocalExecutionBackend") as mock_backend,
            patch("rose.service.api.rest.WorkflowEngine") as mock_engine_cls,
        ):
            mock_backend.return_value = Mock()
            mock_engine_cls.create = AsyncMock(return_value=Mock())

            await rose_session._ensure_engine()

            assert rose_session._engine is not None
            mock_engine_cls.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_workflow(self, rose_session, sample_workflow_yaml):
        """Test workflow submission."""
        # Mock the engine and workflow execution
        with (
            patch.object(rose_session, "_ensure_engine", new_callable=AsyncMock),
            patch.object(rose_session, "_run_workflow", new_callable=AsyncMock),
        ):
            rose_session._engine = Mock()

            result = await rose_session.submit_workflow(sample_workflow_yaml)

            assert "wf_id" in result
            assert result["wf_id"].startswith("wf.")
            assert result["wf_id"] in rose_session._workflows
            assert rose_session._workflows[result["wf_id"]].state == WorkflowState.SUBMITTED

    @pytest.mark.asyncio
    async def test_get_workflow_status_found(self, rose_session):
        """Test getting status of an existing workflow."""
        # Add a workflow manually
        wf = Workflow(wf_id="wf.test123", state=WorkflowState.RUNNING)
        rose_session._workflows["wf.test123"] = wf

        status = await rose_session.get_workflow_status("wf.test123")

        assert status["wf_id"] == "wf.test123"
        assert status["state"] == "RUNNING"

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self, rose_session):
        """Test getting status of non-existent workflow raises 404."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await rose_session.get_workflow_status("wf.nonexistent")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_workflows(self, rose_session):
        """Test listing all workflows."""
        # Add some workflows
        rose_session._workflows["wf.001"] = Workflow(wf_id="wf.001", state=WorkflowState.COMPLETED)
        rose_session._workflows["wf.002"] = Workflow(wf_id="wf.002", state=WorkflowState.RUNNING)

        result = await rose_session.list_workflows()

        assert len(result) == 2
        assert "wf.001" in result
        assert "wf.002" in result
        assert result["wf.001"]["state"] == "COMPLETED"
        assert result["wf.002"]["state"] == "RUNNING"

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, rose_session):
        """Test canceling a running workflow."""
        # Add a running workflow with mock learner
        mock_learner = Mock()
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False

        wf = Workflow(wf_id="wf.cancel", state=WorkflowState.RUNNING)
        wf.learner_instance = mock_learner
        rose_session._workflows["wf.cancel"] = wf
        rose_session._learner_tasks["wf.cancel"] = mock_task

        result = await rose_session.cancel_workflow("wf.cancel")

        assert result["wf_id"] == "wf.cancel"
        mock_learner.stop.assert_called_once()
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_running(self, rose_session):
        """Test canceling a completed workflow raises 400."""
        from fastapi import HTTPException

        wf = Workflow(wf_id="wf.done", state=WorkflowState.COMPLETED)
        rose_session._workflows["wf.done"] = wf

        with pytest.raises(HTTPException) as exc_info:
            await rose_session.cancel_workflow("wf.done")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_close_session(self, rose_session):
        """Test closing session stops all workflows."""
        # Add a running workflow with a real (cancelled) asyncio.Task
        mock_learner = Mock()

        async def _noop():
            await asyncio.sleep(10)

        real_task = asyncio.create_task(_noop())

        wf = Workflow(wf_id="wf.close", state=WorkflowState.RUNNING)
        wf.learner_instance = mock_learner
        rose_session._workflows["wf.close"] = wf
        rose_session._learner_tasks["wf.close"] = real_task

        mock_engine = AsyncMock()
        mock_engine.shutdown = AsyncMock()
        rose_session._engine = mock_engine

        result = await rose_session.close()

        assert result == {}
        assert not rose_session.is_active
        assert real_task.cancelled()
        mock_learner.stop.assert_called_once()
        mock_engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_closed_check(self, rose_session):
        """Test operations on closed session raise error."""
        await rose_session.close()

        with pytest.raises(RuntimeError, match="session is closed"):
            await rose_session.list_workflows()

    # ------------------------------------------------------------------
    # Notification (_dispatch_notify) tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_submit_workflow_dispatches_submitted_notification(
            self, rose_session, sample_workflow_yaml):
        """submit_workflow fires a SUBMITTED workflow_state notification."""
        mock_plugin = MagicMock()
        rose_session._plugin = mock_plugin

        with (
            patch.object(rose_session, "_ensure_engine", new_callable=AsyncMock),
            patch.object(rose_session, "_run_workflow", new_callable=AsyncMock),
        ):
            rose_session._engine = Mock()
            result = await rose_session.submit_workflow(sample_workflow_yaml)

        wf_id = result["wf_id"]
        mock_plugin._dispatch_notify.assert_called_once_with(
            "workflow_state",
            {"wf_id": wf_id, "state": "SUBMITTED", "workflow_file": sample_workflow_yaml},
        )

    @pytest.mark.asyncio
    async def test_notify_state_calls_dispatch_notify(self, rose_session):
        """_notify_state sends the current workflow state via _dispatch_notify."""
        mock_plugin = MagicMock()
        rose_session._plugin = mock_plugin

        wf = Workflow(wf_id="wf.ns01", state=WorkflowState.RUNNING)
        wf.stats = {"iteration": 3}
        rose_session._workflows["wf.ns01"] = wf

        rose_session._notify_state(wf)

        mock_plugin._dispatch_notify.assert_called_once_with(
            "workflow_state",
            {"wf_id": "wf.ns01", "state": "RUNNING", "stats": {"iteration": 3}, "error": None},
        )

    @pytest.mark.asyncio
    async def test_notify_state_no_plugin_does_not_raise(self, rose_session):
        """_notify_state is a no-op when _plugin is None (e.g. in bare unit tests)."""
        wf = Workflow(wf_id="wf.nop", state=WorkflowState.SUBMITTED)
        # _plugin is None by default — must not raise AttributeError
        rose_session._notify_state(wf)

    @pytest.mark.asyncio
    async def test_task_event_dispatched_via_plugin(self, rose_session):
        """_run_workflow wraps learner tasks and fires task_event notifications."""
        mock_plugin = MagicMock()
        rose_session._plugin = mock_plugin

        # Minimal fake learner whose _register_task calls the callback synchronously
        import concurrent.futures
        fut = concurrent.futures.Future()
        fut.set_result("ok output")

        orig_calls = []

        def fake_register(task_obj, deps=None):
            orig_calls.append(task_obj)
            return fut

        mock_learner = Mock()
        mock_learner._register_task = fake_register

        with (
            patch.object(rose_session, "_ensure_engine", new_callable=AsyncMock),
            patch("rose.service.api.rest.WorkflowLoader.load_yaml", return_value={}),
            patch("rose.service.api.rest.WorkflowLoader.create_learner",
                  return_value=(mock_learner, {})),
            patch("rose.service.api.rest.WorkflowLoader.run_learner",
                  new_callable=AsyncMock) as mock_run,
        ):
            rose_session._engine = Mock()

            # Trigger the patched _register_task wrapper by simulating run_learner
            async def _side_effect(learner, wf_def, cfg, on_iter):
                learner._register_task("dummy_task")

            mock_run.side_effect = _side_effect

            wf = Workflow(wf_id="wf.te01", state=WorkflowState.SUBMITTED)
            rose_session._workflows["wf.te01"] = wf
            await rose_session._run_workflow(wf)

        # The done-callback fires synchronously on a resolved Future, so
        # _dispatch_notify should have been called with "task_event"
        calls = [c for c in mock_plugin._dispatch_notify.call_args_list
                 if c[0][0] == "task_event"]
        assert calls, "Expected at least one task_event notification"
        payload = calls[0][0][1]
        assert payload["wf_id"] == "wf.te01"
        assert payload["ok"] is True


# -----------------------------------------------------------------------------
# RoseClient Tests
# -----------------------------------------------------------------------------


class TestRoseClient:
    """Tests for RoseClient class."""

    @pytest.fixture
    def mock_http(self):
        """Create a mock HTTP client."""
        http = Mock()
        response = Mock()
        response.json.return_value = {"sid": "session.abc123"}
        response.status_code = 200
        response.is_error = False
        http.post.return_value = response
        http.get.return_value = response
        return http

    @pytest.fixture
    def rose_client(self, mock_http):
        """Create a RoseClient with mocked HTTP."""
        client = RoseClient(mock_http, "/test/rose")
        client._sid = "session.test"
        return client

    def test_submit_workflow(self, rose_client, mock_http):
        """Test submitting a workflow via client."""
        mock_http.post.return_value.json.return_value = {"wf_id": "wf.new"}

        result = rose_client.submit_workflow("/path/to/wf.yaml")

        assert result == {"wf_id": "wf.new"}
        mock_http.post.assert_called()

    def test_submit_workflow_no_session(self, mock_http):
        """Test submit without session raises error."""
        client = RoseClient(mock_http, "/test/rose")
        # No session registered

        with pytest.raises(RuntimeError, match="No active session"):
            client.submit_workflow("/path/to/wf.yaml")

    def test_get_workflow_status(self, rose_client, mock_http):
        """Test getting workflow status via client."""
        mock_http.get.return_value.json.return_value = {"wf_id": "wf.123", "state": "RUNNING"}

        result = rose_client.get_workflow_status("wf.123")

        assert result["wf_id"] == "wf.123"
        assert result["state"] == "RUNNING"

    def test_list_workflows(self, rose_client, mock_http):
        """Test listing workflows via client."""
        mock_http.get.return_value.json.return_value = {
            "wf.001": {"state": "COMPLETED"},
            "wf.002": {"state": "RUNNING"},
        }

        result = rose_client.list_workflows()

        assert len(result) == 2

    def test_cancel_workflow(self, rose_client, mock_http):
        """Test canceling workflow via client."""
        mock_http.post.return_value.json.return_value = {"wf_id": "wf.cancel"}

        result = rose_client.cancel_workflow("wf.cancel")

        assert result["wf_id"] == "wf.cancel"

    def test_notification_callbacks(self, rose_client):
        """Test registering notification callbacks."""
        callback = Mock()

        # Should not raise
        with patch.object(rose_client, "register_notification_callback"):
            rose_client.on_workflow_state(callback)
            rose_client.register_notification_callback.assert_called_with(callback)

        with patch.object(rose_client, "unregister_notification_callback"):
            rose_client.off_workflow_state(callback)
            rose_client.unregister_notification_callback.assert_called_with(callback)


# -----------------------------------------------------------------------------
# PluginRose Tests
# -----------------------------------------------------------------------------


class TestPluginRose:
    """Tests for PluginRose class."""

    def test_plugin_attributes(self):
        """Test plugin class attributes."""
        assert PluginRose.plugin_name == "rose"
        assert PluginRose.session_class == RoseSession
        assert PluginRose.client_class == RoseClient
        assert PluginRose.version == "0.2.0"
        assert PluginRose.session_ttl == 0

    def test_ui_config(self):
        """Test UI configuration is defined."""
        assert PluginRose.ui_config is not None
        assert PluginRose.ui_config.title == "ROSE Active Learning"
        assert len(PluginRose.ui_config.forms) == 1
        assert len(PluginRose.ui_config.monitors) == 1
        assert PluginRose.ui_config.notifications is not None
