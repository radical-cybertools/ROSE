"""Unit tests for core learner primitives: IterationState, LearnerConfig.get_task_config,
Learner base class state/callback machinery, and the _stream_parallel fan-in helper."""

import asyncio
import dataclasses
from unittest.mock import MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.learner import (
    IterationState,
    Learner,
    LearnerConfig,
    TaskConfig,
    _stream_parallel,
)

# ---------------------------------------------------------------------------
# IterationState
# ---------------------------------------------------------------------------


class TestIterationState:
    def test_default_values(self):
        state = IterationState(iteration=3)
        assert state.iteration == 3
        assert state.metric_name is None
        assert state.metric_value is None
        assert state.metric_threshold is None
        assert state.metric_history == []
        assert state.should_stop is False
        assert state.current_config is None
        assert state.learner_id is None
        assert state.state == {}

    def test_attribute_access_to_state_dict(self):
        state = IterationState(iteration=0, state={"loss": 0.5, "accuracy": 0.95})
        assert state.loss == 0.5
        assert state.accuracy == 0.95

    def test_missing_state_key_returns_none(self):
        state = IterationState(iteration=0)
        assert state.nonexistent_key is None

    def test_get_with_existing_key(self):
        state = IterationState(iteration=0, state={"loss": 0.5})
        assert state.get("loss") == 0.5

    def test_get_with_missing_key_returns_default(self):
        state = IterationState(iteration=0)
        assert state.get("missing", "default_val") == "default_val"

    def test_to_dict_contains_all_top_level_fields(self):
        state = IterationState(
            iteration=2,
            metric_name="mse",
            metric_value=0.05,
            metric_threshold=0.01,
            should_stop=True,
            learner_id=1,
        )
        d = state.to_dict()
        assert d["iteration"] == 2
        assert d["metric_name"] == "mse"
        assert d["metric_value"] == 0.05
        assert d["metric_threshold"] == 0.01
        assert d["should_stop"] is True
        assert d["learner_id"] == 1

    def test_to_dict_merges_state_dict(self):
        state = IterationState(
            iteration=0,
            state={"labeled_count": 100, "uncertainty": 0.3},
        )
        d = state.to_dict()
        assert d["labeled_count"] == 100
        assert d["uncertainty"] == 0.3

    def test_dataclasses_replace_sets_learner_id(self):
        state = IterationState(iteration=5, metric_value=0.1, metric_name="mse")
        replaced = dataclasses.replace(state, learner_id=3)
        assert replaced.learner_id == 3

    def test_dataclasses_replace_preserves_all_other_fields(self):
        original_state = {"x": 42}
        state = IterationState(
            iteration=5,
            metric_value=0.1,
            metric_name="mse",
            should_stop=False,
            learner_id=None,
            state=original_state,
        )
        replaced = dataclasses.replace(state, learner_id=7)
        assert replaced.iteration == 5
        assert replaced.metric_value == 0.1
        assert replaced.metric_name == "mse"
        assert replaced.should_stop is False
        assert replaced.state is original_state

    def test_learner_id_accepts_int(self):
        state = IterationState(iteration=0, learner_id=0)
        assert state.learner_id == 0

    def test_learner_id_accepts_str(self):
        state = IterationState(iteration=0, learner_id="learner-A")
        assert state.learner_id == "learner-A"

    def test_learner_id_accepts_none(self):
        state = IterationState(iteration=0, learner_id=None)
        assert state.learner_id is None


# ---------------------------------------------------------------------------
# LearnerConfig.get_task_config
# ---------------------------------------------------------------------------


class TestLearnerConfigGetTaskConfig:
    def test_returns_none_when_field_is_none(self):
        config = LearnerConfig()
        assert config.get_task_config("simulation", 0) is None
        assert config.get_task_config("training", 5) is None

    def test_returns_taskconfig_directly_for_all_iterations(self):
        tc = TaskConfig(kwargs={"--lr": "0.01"})
        config = LearnerConfig(training=tc)
        assert config.get_task_config("training", 0) is tc
        assert config.get_task_config("training", 5) is tc
        assert config.get_task_config("training", 99) is tc

    def test_exact_iteration_match_in_dict(self):
        tc_0 = TaskConfig(kwargs={"--n": "100"})
        tc_5 = TaskConfig(kwargs={"--n": "200"})
        config = LearnerConfig(simulation={0: tc_0, 5: tc_5, -1: TaskConfig()})
        assert config.get_task_config("simulation", 0) is tc_0
        assert config.get_task_config("simulation", 5) is tc_5

    def test_falls_back_to_minus_one_key(self):
        default_tc = TaskConfig(kwargs={"--n": "500"})
        config = LearnerConfig(simulation={0: TaskConfig(), -1: default_tc})
        assert config.get_task_config("simulation", 99) is default_tc
        assert config.get_task_config("simulation", 1) is default_tc

    def test_returns_none_when_dict_has_no_match_and_no_default(self):
        config = LearnerConfig(simulation={0: TaskConfig()})
        assert config.get_task_config("simulation", 7) is None

    def test_works_for_all_field_names(self):
        tc = TaskConfig(kwargs={"k": "v"})
        for field in (
            "simulation",
            "training",
            "active_learn",
            "environment",
            "update",
            "criterion",
        ):
            config = LearnerConfig(**{field: tc})
            assert config.get_task_config(field, 0) is tc


# ---------------------------------------------------------------------------
# Learner base class: state registry, callbacks, build_iteration_state
# ---------------------------------------------------------------------------


@pytest.fixture
def learner():
    mock_asyncflow = MagicMock(spec=WorkflowEngine)
    return Learner(mock_asyncflow)


class TestLearnerStateRegistry:
    def test_register_state_stores_value(self, learner):
        learner.register_state("loss", 0.5)
        assert learner.get_state("loss") == 0.5

    def test_get_state_returns_default_when_missing(self, learner):
        assert learner.get_state("nonexistent", "fallback") == "fallback"

    def test_get_all_state_returns_copy(self, learner):
        learner.register_state("a", 1)
        snapshot = learner.get_all_state()
        snapshot["a"] = 999  # mutate the copy
        assert learner.get_state("a") == 1  # original unchanged

    def test_clear_state_empties_registry(self, learner):
        learner.register_state("a", 1)
        learner.register_state("b", 2)
        learner.clear_state()
        assert learner.get_all_state() == {}

    def test_on_state_update_callback_invoked(self, learner):
        calls = []

        def cb(k, v):
            calls.append((k, v))

        learner.on_state_update(cb)
        learner.register_state("x", 42)
        assert calls == [("x", 42)]

    def test_multiple_callbacks_all_invoked(self, learner):
        calls_a, calls_b = [], []

        def cb_a(k, v):
            calls_a.append((k, v))

        def cb_b(k, v):
            calls_b.append((k, v))

        learner.on_state_update(cb_a)
        learner.on_state_update(cb_b)
        learner.register_state("y", 7)
        assert calls_a == [("y", 7)]
        assert calls_b == [("y", 7)]

    def test_callback_error_does_not_break_register_state(self, learner):
        def bad_callback(k, v):
            raise RuntimeError("boom")

        learner.on_state_update(bad_callback)
        # Should not raise
        learner.register_state("z", 99)
        assert learner.get_state("z") == 99

    def test_remove_state_callback(self, learner):
        calls = []

        def cb(k, v):
            calls.append((k, v))

        learner.on_state_update(cb)
        learner.remove_state_callback(cb)
        learner.register_state("a", 1)
        assert calls == []


class TestExtractStateFromResult:
    def test_dict_result_registers_all_keys(self, learner):
        learner._extract_state_from_result({"loss": 0.1, "acc": 0.9})
        assert learner.get_state("loss") == 0.1
        assert learner.get_state("acc") == 0.9

    def test_non_dict_result_does_nothing(self, learner):
        learner._extract_state_from_result("some_string")
        learner._extract_state_from_result(42)
        learner._extract_state_from_result(None)
        assert learner.get_all_state() == {}

    def test_excluded_keys_are_skipped(self, learner):
        learner._extract_state_from_result(
            {"loss": 0.1, "metric_value": 0.05, "should_stop": True},
            exclude_keys={"metric_value", "should_stop"},
        )
        assert learner.get_state("loss") == 0.1
        assert learner.get_state("metric_value") is None
        assert learner.get_state("should_stop") is None


class TestBuildIterationState:
    def test_builds_state_with_metric_info_from_criterion(self, learner):
        learner.criterion_function = {
            "metric_name": "mse",
            "threshold": 0.01,
        }
        state = learner.build_iteration_state(iteration=3, metric_value=0.05, should_stop=False)
        assert state.iteration == 3
        assert state.metric_name == "mse"
        assert state.metric_threshold == 0.01
        assert state.metric_value == 0.05
        assert state.should_stop is False

    def test_builds_state_with_registered_state(self, learner):
        learner.register_state("labeled_count", 200)
        state = learner.build_iteration_state(iteration=0)
        assert state.state["labeled_count"] == 200
        assert state.labeled_count == 200  # attribute-style access

    def test_metric_history_reflects_recorded_values(self, learner):
        learner.metric_values_per_iteration = {0: 0.5, 1: 0.3}
        state = learner.build_iteration_state(iteration=2, metric_value=0.1)
        assert state.metric_history == [0.5, 0.3]

    def test_current_config_stored_in_state(self, learner):
        cfg = LearnerConfig(training=TaskConfig(kwargs={"--lr": "0.001"}))
        state = learner.build_iteration_state(iteration=0, current_config=cfg)
        assert state.current_config is cfg

    def test_no_criterion_function_yields_none_metric_info(self, learner):
        learner.criterion_function = {}
        state = learner.build_iteration_state(iteration=0)
        assert state.metric_name is None
        assert state.metric_threshold is None


# ---------------------------------------------------------------------------
# compare_metric
# ---------------------------------------------------------------------------


class TestCompareMetric:
    def test_less_than(self, learner):
        from rose.metrics import MEAN_SQUARED_ERROR_MSE

        assert learner.compare_metric(MEAN_SQUARED_ERROR_MSE, 0.005, 0.01) is True
        assert learner.compare_metric(MEAN_SQUARED_ERROR_MSE, 0.02, 0.01) is False

    def test_greater_than_custom_operator(self, learner):
        assert learner.compare_metric("MY_METRIC", 5.0, 3.0, operator=">") is True
        assert learner.compare_metric("MY_METRIC", 1.0, 3.0, operator=">") is False

    def test_equal_operator(self, learner):
        assert learner.compare_metric("MY_METRIC", 1.0, 1.0, operator="==") is True
        assert learner.compare_metric("MY_METRIC", 1.1, 1.0, operator="==") is False

    def test_custom_metric_without_operator_raises(self, learner):
        with pytest.raises(ValueError, match="Operator value must be provided"):
            learner.compare_metric("UNKNOWN_METRIC", 0.5, 0.1)

    def test_unknown_operator_raises(self, learner):
        with pytest.raises(ValueError, match="Unknown comparison operator"):
            learner.compare_metric("MY_METRIC", 0.5, 0.1, operator="!=")


# ---------------------------------------------------------------------------
# _stream_parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamParallel:
    async def test_empty_run_fns_completes_immediately(self):
        results = []
        async for state in _stream_parallel([]):
            results.append(state)
        assert results == []

    async def test_single_learner_streams_all_states(self):
        state_a = IterationState(iteration=0)
        state_b = IterationState(iteration=1)

        async def run(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("state", state_a))
                await queue.put(("state", state_b))
            finally:
                await queue.put(("done", None))

        results = []
        async for s in _stream_parallel([run]):
            results.append(s)

        assert results == [state_a, state_b]

    async def test_two_learners_all_states_yielded(self):
        s0 = IterationState(iteration=0, learner_id=0)
        s1 = IterationState(iteration=0, learner_id=1)

        async def run_0(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("state", s0))
            finally:
                await queue.put(("done", None))

        async def run_1(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("state", s1))
            finally:
                await queue.put(("done", None))

        results = []
        async for s in _stream_parallel([run_0, run_1]):
            results.append(s)

        assert len(results) == 2
        assert s0 in results
        assert s1 in results

    async def test_done_count_terminates_loop(self):
        """All N 'done' signals must arrive before _stream_parallel exits."""
        barrier = asyncio.Event()

        async def slow_run(queue: asyncio.Queue) -> None:
            await barrier.wait()
            try:
                await queue.put(("state", IterationState(iteration=0)))
            finally:
                await queue.put(("done", None))

        async def fast_run(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("state", IterationState(iteration=0)))
            finally:
                await queue.put(("done", None))
                barrier.set()

        results = []
        async for s in _stream_parallel([slow_run, fast_run]):
            results.append(s)

        assert len(results) == 2

    async def test_error_is_propagated_after_all_done(self):
        exc = RuntimeError("learner exploded")

        async def failing_run(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("error", exc))
            finally:
                await queue.put(("done", None))

        with pytest.raises(RuntimeError, match="learner exploded"):
            async for _ in _stream_parallel([failing_run]):
                pass

    async def test_error_from_one_does_not_suppress_states_from_other(self):
        good_state = IterationState(iteration=0, learner_id=0)

        async def good_run(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("state", good_state))
            finally:
                await queue.put(("done", None))

        async def bad_run(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("error", ValueError("oops")))
            finally:
                await queue.put(("done", None))

        results = []
        with pytest.raises(ValueError, match="oops"):
            async for s in _stream_parallel([good_run, bad_run]):
                results.append(s)

        # Good learner's state was streamed before exception re-raised
        assert good_state in results

    async def test_only_first_error_is_raised(self):
        """When two learners both fail, only the first error is re-raised."""

        async def fail_a(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("error", ValueError("error-A")))
            finally:
                await queue.put(("done", None))

        async def fail_b(queue: asyncio.Queue) -> None:
            try:
                await queue.put(("error", ValueError("error-B")))
            finally:
                await queue.put(("done", None))

        with pytest.raises(ValueError):
            async for _ in _stream_parallel([fail_a, fail_b]):
                pass
