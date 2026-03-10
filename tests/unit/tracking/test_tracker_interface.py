"""Unit tests for tracker wiring: add_tracker, _build_pipeline_manifest,
_notify_trackers_*, and full lifecycle through SequentialActiveLearner.start()."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import SequentialActiveLearner
from rose.learner import IterationState, TaskConfig
from rose.tracking import PipelineManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingTracker:
    """A simple tracker that records all lifecycle calls."""

    def __init__(self):
        self.started = False
        self.manifest = None
        self.iterations = []
        self.stopped = False
        self.stop_reason = None
        self.final_state = None
        self.state_updates = []

    def on_start(self, manifest):
        self.started = True
        self.manifest = manifest

    def on_iteration(self, state):
        self.iterations.append(state)

    def on_stop(self, final_state, reason):
        self.stopped = True
        self.stop_reason = reason
        self.final_state = final_state

    def on_state_update(self, key, value):
        self.state_updates.append((key, value))


# ---------------------------------------------------------------------------
# TestAddTracker
# ---------------------------------------------------------------------------


class TestAddTracker:
    """Tests for Learner.add_tracker()."""

    @pytest.fixture
    def mock_asyncflow(self):
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def learner(self, mock_asyncflow):
        return SequentialActiveLearner(mock_asyncflow)

    @pytest.fixture
    def recording_tracker(self):
        return RecordingTracker()

    def test_add_tracker_calls_on_start(self, learner, recording_tracker):
        learner.add_tracker(recording_tracker)
        assert recording_tracker.started is True

    def test_add_tracker_passes_manifest(self, learner, recording_tracker):
        learner.add_tracker(recording_tracker)
        assert isinstance(recording_tracker.manifest, PipelineManifest)
        assert recording_tracker.manifest.learner_type == "SequentialActiveLearner"

    def test_add_tracker_appends_to_trackers_list(self, learner, recording_tracker):
        learner.add_tracker(recording_tracker)
        assert len(learner._trackers) == 1

    def test_add_tracker_wires_state_update(self, learner, recording_tracker):
        learner.add_tracker(recording_tracker)
        learner.register_state("loss", 0.5)
        assert recording_tracker.state_updates == [("loss", 0.5)]

    def test_add_multiple_trackers(self, learner):
        tracker_a = RecordingTracker()
        tracker_b = RecordingTracker()
        learner.add_tracker(tracker_a)
        learner.add_tracker(tracker_b)
        assert tracker_a.started is True
        assert tracker_b.started is True
        assert len(learner._trackers) == 2


# ---------------------------------------------------------------------------
# TestBuildPipelineManifest
# ---------------------------------------------------------------------------


class TestBuildPipelineManifest:
    """Tests for Learner._build_pipeline_manifest()."""

    @pytest.fixture
    def mock_asyncflow(self):
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def learner_with_tasks(self, mock_asyncflow):
        learner = SequentialActiveLearner(mock_asyncflow)

        @learner.simulation_task(as_executable=False)
        async def sim(*args, **kwargs):
            return {}

        @learner.training_task(as_executable=False)
        async def train(*args, **kwargs):
            return {}

        return learner

    def test_learner_type_is_class_name(self, learner_with_tasks):
        manifest = learner_with_tasks._build_pipeline_manifest()
        assert manifest.learner_type == "SequentialActiveLearner"

    def test_registered_tasks_in_manifest(self, learner_with_tasks):
        manifest = learner_with_tasks._build_pipeline_manifest()
        assert "simulation" in manifest.tasks
        assert "training" in manifest.tasks

    def test_task_manifest_func_name(self, learner_with_tasks):
        manifest = learner_with_tasks._build_pipeline_manifest()
        assert manifest.tasks["simulation"].func_name == "sim"

    def test_task_manifest_as_executable_false(self, learner_with_tasks):
        manifest = learner_with_tasks._build_pipeline_manifest()
        assert manifest.tasks["simulation"].as_executable is False

    def test_no_criterion_yields_none(self, learner_with_tasks):
        manifest = learner_with_tasks._build_pipeline_manifest()
        assert manifest.criterion is None

    def test_criterion_manifest_populated(self, mock_asyncflow):
        learner = SequentialActiveLearner(mock_asyncflow)

        @learner.simulation_task(as_executable=False)
        async def sim(*args, **kwargs):
            return {}

        @learner.training_task(as_executable=False)
        async def train(*args, **kwargs):
            return {}

        @learner.active_learn_task(as_executable=False)
        async def active_learn(*args, **kwargs):
            return {}

        @learner.as_stop_criterion(metric_name="mse", threshold=0.1, operator="<")
        async def check(*args, **kwargs):
            return 0.05

        manifest = learner._build_pipeline_manifest()
        assert manifest.criterion is not None
        assert manifest.criterion.metric_name == "mse"
        assert manifest.criterion.threshold == 0.1
        assert manifest.criterion.operator == "<"


# ---------------------------------------------------------------------------
# TestNotifyTrackers
# ---------------------------------------------------------------------------


class TestNotifyTrackers:
    """Tests for _notify_trackers_iteration and _notify_trackers_stop."""

    @pytest.fixture
    def mock_asyncflow(self):
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def learner(self, mock_asyncflow):
        return SequentialActiveLearner(mock_asyncflow)

    def test_notify_iteration_calls_all_trackers(self, learner):
        tracker_a = RecordingTracker()
        tracker_b = RecordingTracker()
        learner.add_tracker(tracker_a)
        learner.add_tracker(tracker_b)

        state = IterationState(iteration=0)
        learner._notify_trackers_iteration(state)

        assert len(tracker_a.iterations) == 1
        assert len(tracker_b.iterations) == 1

    def test_notify_iteration_swallows_exception(self, learner):
        class BrokenTracker(RecordingTracker):
            def on_iteration(self, state):
                raise RuntimeError("boom")

        broken = BrokenTracker()
        good = RecordingTracker()

        learner._trackers.append(broken)
        learner._trackers.append(good)

        state = IterationState(iteration=0)
        # Should not raise
        learner._notify_trackers_iteration(state)
        # Good tracker should still receive the call
        assert len(good.iterations) == 1

    def test_notify_stop_calls_all_trackers(self, learner):
        tracker_a = RecordingTracker()
        tracker_b = RecordingTracker()
        learner.add_tracker(tracker_a)
        learner.add_tracker(tracker_b)

        state = IterationState(iteration=5)
        learner._notify_trackers_stop(state, "max_iter_reached")

        assert tracker_a.stopped is True
        assert tracker_b.stopped is True
        assert tracker_a.stop_reason == "max_iter_reached"
        assert tracker_b.stop_reason == "max_iter_reached"

    def test_notify_stop_swallows_exception(self, learner):
        class BrokenTracker(RecordingTracker):
            def on_stop(self, final_state, reason):
                raise RuntimeError("boom")

        broken = BrokenTracker()
        good = RecordingTracker()

        learner._trackers.append(broken)
        learner._trackers.append(good)

        state = IterationState(iteration=0)
        # Should not raise
        learner._notify_trackers_stop(state, "stopped")
        # Good tracker should still receive the call
        assert good.stopped is True


# ---------------------------------------------------------------------------
# TestTrackerLifecycleInLoop
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTrackerLifecycleInLoop:
    """Integration tests: tracker lifecycle through the SequentialActiveLearner loop."""

    @pytest.fixture
    def mock_asyncflow(self):
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def learner(self, mock_asyncflow):
        return SequentialActiveLearner(mock_asyncflow)

    @pytest.fixture
    def configured_learner(self, learner):
        learner.simulation_function = AsyncMock(return_value="sim_result")
        learner.training_function = AsyncMock(return_value="train_result")
        learner.active_learn_function = AsyncMock(return_value="active_result")
        learner.criterion_function = AsyncMock(return_value=False)
        learner._get_iteration_task_config = MagicMock(
            return_value=MagicMock(spec=TaskConfig)
        )
        learner._register_task = AsyncMock(return_value="task_result")
        learner._check_stop_criterion = MagicMock(return_value=(False, None))
        return learner

    def test_on_start_called_once_at_add_tracker(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)
        assert tracker.started is True

    @pytest.mark.asyncio
    async def test_on_iteration_called_per_iteration(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        async for _ in configured_learner.start(max_iter=3, skip_pre_loop=True):
            pass

        assert len(tracker.iterations) == 3

    @pytest.mark.asyncio
    async def test_on_iteration_receives_correct_iteration_number(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        async for _ in configured_learner.start(max_iter=3, skip_pre_loop=True):
            pass

        for i in range(3):
            assert tracker.iterations[i].iteration == i

    @pytest.mark.asyncio
    async def test_on_stop_called_after_max_iter(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        async for _ in configured_learner.start(max_iter=2, skip_pre_loop=True):
            pass

        assert tracker.stopped is True
        assert tracker.stop_reason == "max_iter_reached"

    @pytest.mark.asyncio
    async def test_on_stop_called_when_criterion_met(self, configured_learner):
        configured_learner._check_stop_criterion.side_effect = [(True, 0.005)]

        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        async for _ in configured_learner.start(max_iter=0, skip_pre_loop=True):
            pass

        assert tracker.stopped is True
        assert tracker.stop_reason == "criterion_met"

    @pytest.mark.asyncio
    async def test_on_stop_called_on_early_break(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        gen = configured_learner.start(max_iter=10, skip_pre_loop=True)
        async for _ in gen:
            break
        await gen.aclose()

        assert tracker.stopped is True

    @pytest.mark.asyncio
    async def test_on_stop_final_state_is_last_yielded(self, configured_learner):
        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        async for _ in configured_learner.start(max_iter=3, skip_pre_loop=True):
            pass

        assert tracker.final_state is not None
        assert tracker.final_state.iteration == 2

    @pytest.mark.asyncio
    async def test_on_stop_always_called_even_on_exception(self, configured_learner):
        call_count = 0

        async def raise_after_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Raise on 4th call (after sim, train, AL, criterion in iteration 0
            # then next sim/train prep fails)
            if call_count > 3:
                raise RuntimeError("simulated failure")
            return "task_result"

        configured_learner._register_task = raise_after_first

        tracker = RecordingTracker()
        configured_learner.add_tracker(tracker)

        try:
            async for _ in configured_learner.start(max_iter=5, skip_pre_loop=True):
                pass
        except RuntimeError:
            pass

        assert tracker.stopped is True

    @pytest.mark.asyncio
    async def test_multiple_trackers_all_get_lifecycle_calls(self, configured_learner):
        tracker_a = RecordingTracker()
        tracker_b = RecordingTracker()
        configured_learner.add_tracker(tracker_a)
        configured_learner.add_tracker(tracker_b)

        async for _ in configured_learner.start(max_iter=2, skip_pre_loop=True):
            pass

        for tracker in (tracker_a, tracker_b):
            assert tracker.started is True
            assert len(tracker.iterations) == 2
            assert tracker.stopped is True

    @pytest.mark.asyncio
    async def test_tracker_exception_does_not_break_learner(self, configured_learner):
        class NoisyTracker:
            def on_start(self, manifest):
                pass

            def on_iteration(self, state):
                raise RuntimeError("tracker error")

            def on_stop(self, final_state, reason):
                raise RuntimeError("tracker stop error")

            def on_state_update(self, key, value):
                pass

        configured_learner.add_tracker(NoisyTracker())

        states = []
        async for state in configured_learner.start(max_iter=2, skip_pre_loop=True):
            states.append(state)

        assert len(states) == 2
