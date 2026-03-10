"""Unit tests for MLflowTracker.

mlflow is never actually imported — every test injects a MagicMock into
sys.modules before importing the tracker class.
"""

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from rose.learner import IterationState
from rose.tracking import CriterionManifest, PipelineManifest, TaskManifest


# ---------------------------------------------------------------------------
# Shared fixtures (module-level helpers used by multiple classes)
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
                decor_kwargs={"num_gpus": 2},
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


# ---------------------------------------------------------------------------
# TestMLflowTrackerInit
# ---------------------------------------------------------------------------


class TestMLflowTrackerInit:
    def test_raises_import_error_when_mlflow_missing(self):
        with patch.dict(sys.modules, {"mlflow": None, "mlflow.models": None}):
            with pytest.raises(ImportError, match="pip install rose\\[mlflow\\]"):
                from rose.integrations.mlflow_tracker import MLflowTracker  # noqa: F401

                MLflowTracker(experiment_name="x")

    def test_init_succeeds_when_mlflow_available(self):
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.models": MagicMock()}):
            from rose.integrations.mlflow_tracker import MLflowTracker

            tracker = MLflowTracker(experiment_name="test-exp", run_name="test-run")
            assert tracker is not None


# ---------------------------------------------------------------------------
# TestMLflowTrackerOnStart
# ---------------------------------------------------------------------------


class TestMLflowTrackerOnStart:
    @pytest.fixture
    def mock_mlflow(self):
        mock = MagicMock()
        mock_run = MagicMock()
        mock.start_run.return_value = mock_run
        with patch.dict(sys.modules, {"mlflow": mock, "mlflow.models": MagicMock()}):
            yield mock

    @pytest.fixture
    def tracker(self, mock_mlflow):
        from rose.integrations.mlflow_tracker import MLflowTracker

        return MLflowTracker(experiment_name="test-exp", run_name="test-run")

    @pytest.fixture
    def simple_manifest(self):
        return _make_simple_manifest()

    def test_sets_experiment(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")

    def test_starts_run_with_name(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_logs_learner_type_param(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["learner_type"] == "SequentialActiveLearner"

    def test_logs_criterion_params(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["criterion/metric_name"] == "mean_squared_error_mse"
        assert logged_params["criterion/threshold"] == 0.01

    def test_logs_task_as_executable(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["task.simulation.as_executable"] is False

    def test_logs_task_decor_kwargs(self, tracker, mock_mlflow, simple_manifest):
        tracker.on_start(simple_manifest)
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["task.training.num_gpus"] == 2

    def test_no_criterion_logs_none(self, tracker, mock_mlflow):
        manifest = PipelineManifest(
            learner_type="SequentialActiveLearner",
            tasks={},
            criterion=None,
        )
        tracker.on_start(manifest)
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["criterion/metric_name"] is None


# ---------------------------------------------------------------------------
# TestMLflowTrackerOnIteration
# ---------------------------------------------------------------------------


class TestMLflowTrackerOnIteration:
    @pytest.fixture
    def mock_mlflow(self):
        mock = MagicMock()
        mock.start_run.return_value = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock, "mlflow.models": MagicMock()}):
            yield mock

    @pytest.fixture
    def tracker(self, mock_mlflow):
        from rose.integrations.mlflow_tracker import MLflowTracker

        return MLflowTracker(experiment_name="test-exp", run_name="test-run")

    @pytest.fixture
    def iteration_state(self):
        return IterationState(
            iteration=3,
            metric_value=0.05,
            metric_name="mse",
            state={"n_labeled": 25, "train_loss": 0.03, "label": "not_numeric"},
        )

    def test_logs_metric_value(self, tracker, mock_mlflow, iteration_state):
        tracker.on_iteration(iteration_state)
        mock_mlflow.log_metric.assert_any_call("mse", 0.05, step=3)

    def test_logs_scalar_state_values(self, tracker, mock_mlflow, iteration_state):
        tracker.on_iteration(iteration_state)
        mock_mlflow.log_metric.assert_any_call("n_labeled", 25, step=3)
        mock_mlflow.log_metric.assert_any_call("train_loss", 0.03, step=3)

    def test_skips_non_numeric_state_values(self, tracker, mock_mlflow, iteration_state):
        tracker.on_iteration(iteration_state)
        logged_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert "label" not in logged_keys

    def test_no_metric_value_skips_metric_log(self, tracker, mock_mlflow):
        state = IterationState(
            iteration=1,
            metric_value=None,
            metric_name="mse",
            state={},
        )
        tracker.on_iteration(state)
        logged_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert "mse" not in logged_keys

    def test_uses_fallback_metric_name(self, tracker, mock_mlflow):
        state = IterationState(
            iteration=2,
            metric_value=0.1,
            metric_name=None,
            state={},
        )
        tracker.on_iteration(state)
        mock_mlflow.log_metric.assert_any_call("metric", 0.1, step=2)


# ---------------------------------------------------------------------------
# TestMLflowTrackerOnStop
# ---------------------------------------------------------------------------


class TestMLflowTrackerOnStop:
    @pytest.fixture
    def mock_mlflow(self):
        mock = MagicMock()
        mock.start_run.return_value = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock, "mlflow.models": MagicMock()}):
            yield mock

    @pytest.fixture
    def tracker(self, mock_mlflow):
        from rose.integrations.mlflow_tracker import MLflowTracker

        return MLflowTracker(experiment_name="test-exp", run_name="test-run")

    def test_sets_stop_reason_tag(self, tracker, mock_mlflow):
        tracker.on_stop(IterationState(iteration=5), "criterion_met")
        mock_mlflow.set_tag.assert_any_call("stop_reason", "criterion_met")

    def test_sets_final_iteration_tag(self, tracker, mock_mlflow):
        tracker.on_stop(IterationState(iteration=5), "criterion_met")
        mock_mlflow.set_tag.assert_any_call("final_iteration", "5")

    def test_ends_run(self, tracker, mock_mlflow):
        tracker.on_stop(IterationState(iteration=1), "max_iter_reached")
        mock_mlflow.end_run.assert_called_once()

    def test_no_final_state_skips_iteration_tag(self, tracker, mock_mlflow):
        tracker.on_stop(None, "stopped")
        tag_keys = [c[0][0] for c in mock_mlflow.set_tag.call_args_list]
        assert "final_iteration" not in tag_keys

    @pytest.mark.parametrize(
        "reason",
        ["criterion_met", "max_iter_reached", "stopped", "error"],
    )
    def test_all_stop_reasons(self, tracker, mock_mlflow, reason):
        tracker.on_stop(None, reason)
        mock_mlflow.set_tag.assert_any_call("stop_reason", reason)


# ---------------------------------------------------------------------------
# TestMLflowTrackerOnStateUpdate
# ---------------------------------------------------------------------------


class TestMLflowTrackerOnStateUpdate:
    @pytest.fixture
    def mock_mlflow(self):
        mock = MagicMock()
        mock.start_run.return_value = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock, "mlflow.models": MagicMock()}):
            yield mock

    @pytest.fixture
    def tracker(self, mock_mlflow):
        from rose.integrations.mlflow_tracker import MLflowTracker

        return MLflowTracker(experiment_name="test-exp", run_name="test-run")

    def test_logs_numeric_live_metric(self, tracker, mock_mlflow):
        tracker._run = MagicMock()
        tracker.on_state_update("loss", 0.42)
        mock_mlflow.log_metric.assert_called_once_with("live.loss", 0.42)

    def test_skips_non_numeric(self, tracker, mock_mlflow):
        tracker._run = MagicMock()
        tracker.on_state_update("label", "abc")
        mock_mlflow.log_metric.assert_not_called()

    def test_skips_when_no_active_run(self, tracker, mock_mlflow):
        tracker._run = None
        tracker.on_state_update("loss", 0.42)
        mock_mlflow.log_metric.assert_not_called()
