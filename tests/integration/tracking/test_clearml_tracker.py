"""Unit tests for ClearMLTracker.

clearml is never actually imported — every test injects a MagicMock into sys.modules before
importing the tracker class.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from rose.learner import IterationState
from rose.tracking import CriterionManifest, PipelineManifest, TaskManifest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_simple_manifest():
    return PipelineManifest(
        learner_type="SequentialActiveLearner",
        tasks={
            "simulation": TaskManifest(
                func_name="sim",
                func_module="mod",
                as_executable=False,
            ),
            "training": TaskManifest(
                func_name="train",
                func_module="mod",
                as_executable=False,
                log_params={"num_gpus": 2},
            ),
        },
        criterion=CriterionManifest(
            func_name="check",
            func_module="mod",
            as_executable=False,
            metric_name="mean_squared_error_mse",
            threshold=0.01,
            operator="<",
        ),
    )


def _make_mock_clearml():
    """Return a dict mirroring the fixture structure for use inside with blocks."""
    mock_task_instance = MagicMock()
    mock_logger = MagicMock()
    mock_task_instance.get_logger.return_value = mock_logger

    mock_task_cls = MagicMock()
    mock_task_cls.init.return_value = mock_task_instance

    mock_clearml_module = MagicMock()
    mock_clearml_module.Task = mock_task_cls

    return {
        "module": mock_clearml_module,
        "Task": mock_task_cls,
        "task": mock_task_instance,
        "logger": mock_logger,
    }


# ---------------------------------------------------------------------------
# TestClearMLTrackerInit
# ---------------------------------------------------------------------------


class TestClearMLTrackerInit:
    def test_raises_import_error_when_clearml_missing(self):
        with patch.dict(sys.modules, {"clearml": None}):
            with pytest.raises(ImportError):
                from rose.integrations.clearml_tracker import ClearMLTracker  # noqa: F401

                ClearMLTracker(project_name="p", task_name="t")

    def test_init_succeeds_when_clearml_available(self):
        mocks = _make_mock_clearml()
        with patch.dict(sys.modules, {"clearml": mocks["module"]}):
            from rose.integrations.clearml_tracker import ClearMLTracker

            tracker = ClearMLTracker(project_name="test-project", task_name="test-task")
            assert tracker is not None


# ---------------------------------------------------------------------------
# TestClearMLTrackerOnStart
# ---------------------------------------------------------------------------


class TestClearMLTrackerOnStart:
    @pytest.fixture
    def mock_clearml(self):
        mocks = _make_mock_clearml()
        with patch.dict(sys.modules, {"clearml": mocks["module"]}):
            yield mocks

    @pytest.fixture
    def tracker(self, mock_clearml):
        from rose.integrations.clearml_tracker import ClearMLTracker

        return ClearMLTracker(project_name="test-project", task_name="test-task")

    @pytest.fixture
    def simple_manifest(self):
        return _make_simple_manifest()

    def test_inits_clearml_task(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        mock_clearml["Task"].init.assert_called_once_with(
            project_name="test-project",
            task_name="test-task",
        )

    def test_gets_logger(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        mock_clearml["task"].get_logger.assert_called_once()

    def test_connects_learner_type(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        connected_params = mock_clearml["task"].connect.call_args[0][0]
        assert "learner_type" in connected_params
        assert connected_params["learner_type"] == "SequentialActiveLearner"

    def test_connects_criterion_params(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        connected_params = mock_clearml["task"].connect.call_args[0][0]
        assert connected_params["criterion/metric_name"] == "mean_squared_error_mse"
        assert connected_params["criterion/threshold"] == 0.01

    def test_connects_criterion_operator(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        connected_params = mock_clearml["task"].connect.call_args[0][0]
        assert connected_params["criterion/operator"] == "<"

    def test_connects_task_params(self, tracker, mock_clearml, simple_manifest):
        tracker.on_start(simple_manifest)
        connected_params = mock_clearml["task"].connect.call_args[0][0]
        assert "task/simulation/as_executable" in connected_params

    def test_no_criterion_connects_none_values(self, tracker, mock_clearml):
        manifest = PipelineManifest(
            learner_type="SequentialActiveLearner",
            tasks={},
            criterion=None,
        )
        tracker.on_start(manifest)
        connected_params = mock_clearml["task"].connect.call_args[0][0]
        assert connected_params["criterion/metric_name"] is None
        assert connected_params["criterion/threshold"] is None


# ---------------------------------------------------------------------------
# TestClearMLTrackerOnIteration
# ---------------------------------------------------------------------------


class TestClearMLTrackerOnIteration:
    @pytest.fixture
    def mock_clearml(self):
        mocks = _make_mock_clearml()
        with patch.dict(sys.modules, {"clearml": mocks["module"]}):
            yield mocks

    @pytest.fixture
    def tracker(self, mock_clearml):
        from rose.integrations.clearml_tracker import ClearMLTracker

        t = ClearMLTracker(project_name="test-project", task_name="test-task")
        # Simulate on_start having been called
        t._task = mock_clearml["task"]
        t._logger = mock_clearml["logger"]
        return t

    @pytest.fixture
    def iteration_state(self):
        return IterationState(
            iteration=3,
            metric_value=0.05,
            metric_name="mse",
            state={"n_labeled": 25, "train_loss": 0.03, "label": "not_numeric"},
        )

    def test_reports_metric_scalar(self, tracker, mock_clearml, iteration_state):
        tracker.on_iteration(iteration_state)
        mock_clearml["logger"].report_scalar.assert_any_call(
            title="mse",
            series="value",
            value=0.05,
            iteration=3,
        )

    def test_reports_state_scalars(self, tracker, mock_clearml, iteration_state):
        tracker.on_iteration(iteration_state)
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert "n_labeled" in reported_titles
        assert "train_loss" in reported_titles

    def test_skips_non_numeric_state(self, tracker, mock_clearml, iteration_state):
        tracker.on_iteration(iteration_state)
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert "label" not in reported_titles

    def test_skips_when_no_logger(self, tracker, mock_clearml, iteration_state):
        tracker._logger = None
        # Should not raise
        tracker.on_iteration(iteration_state)
        mock_clearml["logger"].report_scalar.assert_not_called()

    def test_parallel_learner_uses_learner_id_as_series(self, tracker, mock_clearml):
        state = IterationState(
            iteration=2,
            metric_value=0.05,
            metric_name="mse",
            learner_id="B",
            state={},
        )
        tracker.on_iteration(state)
        mock_clearml["logger"].report_scalar.assert_called_once_with(
            title="mse",
            series="B",
            value=0.05,
            iteration=2,
        )

    def test_sequential_learner_uses_value_as_series(self, tracker, mock_clearml):
        state = IterationState(
            iteration=0,
            metric_value=0.1,
            metric_name="mse",
            learner_id=None,
            state={},
        )
        tracker.on_iteration(state)
        call_kwargs = mock_clearml["logger"].report_scalar.call_args[1]
        assert call_kwargs["series"] == "value"

    def test_logs_config_kwargs_as_scalars(self, tracker, mock_clearml):
        from rose.learner import LearnerConfig, TaskConfig

        state = IterationState(
            iteration=10,
            metric_value=0.01,
            metric_name="mse",
            state={},
            current_config=LearnerConfig(
                training=TaskConfig(kwargs={"length_scale": 0.2, "noise_level": 0.01})
            ),
        )
        tracker.on_iteration(state)
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert "config/training/length_scale" in reported_titles
        assert "config/training/noise_level" in reported_titles
        # Verify values
        for call in mock_clearml["logger"].report_scalar.call_args_list:
            if call[1]["title"] == "config/training/length_scale":
                assert call[1]["value"] == 0.2
                assert call[1]["iteration"] == 10

    def test_skips_cli_style_config_kwargs(self, tracker, mock_clearml):
        from rose.learner import LearnerConfig, TaskConfig

        state = IterationState(
            iteration=0,
            metric_value=None,
            metric_name=None,
            state={},
            current_config=LearnerConfig(
                training=TaskConfig(kwargs={"--learner_name": "ensemble-A", "lr": 0.001})
            ),
        )
        tracker.on_iteration(state)
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert not any("learner_name" in t for t in reported_titles)
        assert any("lr" in t for t in reported_titles)

    def test_no_config_skips_config_scalars(self, tracker, mock_clearml):
        state = IterationState(
            iteration=0,
            metric_value=None,
            metric_name=None,
            state={},
            current_config=None,
        )
        tracker.on_iteration(state)
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert not any("config/" in t for t in reported_titles)

    def test_logs_string_config_kwargs_to_current_config_section(self, tracker, mock_clearml):
        from rose.learner import LearnerConfig, TaskConfig

        state = IterationState(
            iteration=5,
            metric_value=None,
            metric_name=None,
            state={},
            current_config=LearnerConfig(
                training=TaskConfig(kwargs={"kernel_type": "RBF+WhiteKernel", "length_scale": 0.2})
            ),
        )
        tracker.on_iteration(state)
        mock_clearml["task"].connect.assert_called_once_with(
            {"training/kernel_type": "RBF+WhiteKernel"}, name="current_config"
        )
        # numeric value goes to report_scalar, not connect
        reported_titles = [
            c[1]["title"] for c in mock_clearml["logger"].report_scalar.call_args_list
        ]
        assert "config/training/length_scale" in reported_titles

    def test_learner_names_maps_int_id_to_name(self, mock_clearml):
        from rose.integrations.clearml_tracker import ClearMLTracker

        t = ClearMLTracker(
            project_name="p", task_name="t", learner_names=["ensemble-A", "ensemble-B"]
        )
        t._task = mock_clearml["task"]
        t._logger = mock_clearml["logger"]

        state = IterationState(
            iteration=1, metric_value=0.05, metric_name="mse", learner_id=1, state={}
        )
        t.on_iteration(state)
        mock_clearml["logger"].report_scalar.assert_called_once_with(
            title="mse", series="ensemble-B", value=0.05, iteration=1
        )

    def test_integer_learner_id_zero_uses_string_zero_not_value(self, tracker, mock_clearml):
        state = IterationState(
            iteration=0, metric_value=0.1, metric_name="mse", learner_id=0, state={}
        )
        tracker.on_iteration(state)
        call_kwargs = mock_clearml["logger"].report_scalar.call_args[1]
        assert call_kwargs["series"] == "0"
        assert call_kwargs["series"] != "value"


# ---------------------------------------------------------------------------
# TestClearMLTrackerOnStop
# ---------------------------------------------------------------------------


class TestClearMLTrackerOnStop:
    @pytest.fixture
    def mock_clearml(self):
        mocks = _make_mock_clearml()
        with patch.dict(sys.modules, {"clearml": mocks["module"]}):
            yield mocks

    @pytest.fixture
    def tracker(self, mock_clearml):
        from rose.integrations.clearml_tracker import ClearMLTracker

        t = ClearMLTracker(project_name="test-project", task_name="test-task")
        t._task = mock_clearml["task"]
        t._logger = mock_clearml["logger"]
        return t

    def test_adds_stop_reason_tag(self, tracker, mock_clearml):
        tracker.on_stop(IterationState(iteration=5), "criterion_met")
        mock_clearml["task"].add_tags.assert_any_call(["stop:criterion_met"])

    def test_adds_final_iter_tag(self, tracker, mock_clearml):
        tracker.on_stop(IterationState(iteration=5), "criterion_met")
        mock_clearml["task"].add_tags.assert_any_call(["final_iter:5"])

    def test_closes_task(self, tracker, mock_clearml):
        tracker.on_stop(IterationState(iteration=1), "max_iter_reached")
        mock_clearml["task"].close.assert_called_once()

    def test_marks_failed_on_error(self, tracker, mock_clearml):
        tracker.on_stop(None, "error")
        mock_clearml["task"].mark_failed.assert_called_once_with(status_reason="error during run")
        mock_clearml["task"].close.assert_called_once()

    def test_does_not_mark_failed_on_normal_stop(self, tracker, mock_clearml):
        tracker.on_stop(IterationState(iteration=3), "criterion_met")
        mock_clearml["task"].mark_failed.assert_not_called()

    def test_no_final_state_skips_iter_tag(self, tracker, mock_clearml):
        tracker.on_stop(None, "stopped")
        all_tag_calls = mock_clearml["task"].add_tags.call_args_list
        # Only one add_tags call (for stop reason), no final_iter tag
        assert len(all_tag_calls) == 1
        assert all_tag_calls[0][0][0] == ["stop:stopped"]

    def test_skips_when_no_task(self, tracker, mock_clearml):
        tracker._task = None
        # Should not raise
        tracker.on_stop(IterationState(iteration=1), "error")
        mock_clearml["task"].add_tags.assert_not_called()
        mock_clearml["task"].close.assert_not_called()
