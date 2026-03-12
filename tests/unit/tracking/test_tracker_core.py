"""Unit tests for rose.tracking: TrackerBase protocol and manifest dataclasses."""

from unittest.mock import MagicMock

from rose.learner import IterationState
from rose.tracking import (
    CriterionManifest,
    PipelineManifest,
    TaskManifest,
    TrackerBase,
)

# ---------------------------------------------------------------------------
# TestTrackerBase
# ---------------------------------------------------------------------------


class TestTrackerBase:
    """TrackerBase default no-op methods do not raise."""

    def test_default_noop_on_start(self):
        tracker = TrackerBase()
        mock_manifest = MagicMock(spec=PipelineManifest)
        tracker.on_start(mock_manifest)  # should not raise

    def test_default_noop_on_iteration(self):
        tracker = TrackerBase()
        mock_state = MagicMock(spec=IterationState)
        tracker.on_iteration(mock_state)  # should not raise

    def test_default_noop_on_stop(self):
        tracker = TrackerBase()
        tracker.on_stop(None, "error")  # should not raise


# ---------------------------------------------------------------------------
# TestPipelineManifest
# ---------------------------------------------------------------------------


class TestPipelineManifest:
    """PipelineManifest and related dataclass construction."""

    def test_manifest_defaults(self):
        manifest = PipelineManifest(learner_type="X")
        assert manifest.tasks == {}
        assert manifest.criterion is None
        assert manifest.parallel_count is None

    def test_criterion_manifest_fields(self):
        cm = CriterionManifest(
            func_name="check",
            func_module="mymod",
            as_executable=False,
            metric_name="mse",
            threshold=0.01,
            operator="<",
        )
        assert cm.func_name == "check"
        assert cm.func_module == "mymod"
        assert cm.as_executable is False
        assert cm.metric_name == "mse"
        assert cm.threshold == 0.01
        assert cm.operator == "<"

    def test_task_manifest_fields(self):
        tm = TaskManifest(
            func_name="sim",
            func_module="mymod",
            as_executable=True,
            decor_kwargs={"num_gpus": 4},
            log_params={"num_gpus": 4, "kernel": "rbf"},
        )
        assert tm.func_name == "sim"
        assert tm.func_module == "mymod"
        assert tm.as_executable is True
        assert tm.decor_kwargs == {"num_gpus": 4}
        assert tm.log_params == {"num_gpus": 4, "kernel": "rbf"}

    def test_task_manifest_log_params_defaults_empty(self):
        tm = TaskManifest(func_name="sim", func_module="mymod", as_executable=False)
        assert tm.log_params == {}
