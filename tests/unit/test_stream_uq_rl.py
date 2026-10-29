"""Unit tests for stream mode of the UQ and RL learners: window placement, publish gate
(incl. UQ publish_on), stop reasons, skip_environment_step and parallel broadcast."""

import asyncio
from unittest.mock import MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose import Stream
from rose.reinforcement_learning.reinforcement_learner import (
    ParallelExperience,
    ParallelReinforcementLearner,
    SequentialReinforcementLearner,
)
from rose.uncertainty_quantification.uq_active_learner import ParallelUQLearner, SeqUQLearner


def _noop(*args, **kwargs):
    return None


def task(name, **extra):
    return {"func": _noop, "args": (), "kwargs": {}, "decor_kwargs": {}, "name": name, **extra}


def criterion(**extra):
    return task("crit", operator="<", threshold=1.0, metric_name="test_metric", **extra)


async def numbers(n):
    for i in range(n):
        yield i


async def collect(gen, timeout=5.0):
    async def run():
        return [state async for state in gen]

    return await asyncio.wait_for(run(), timeout)


def recorder(metrics=None):
    """Mock _register_task: records (task name, args); criterion results come from ``metrics``,
    a function of (task_obj, call count)."""
    calls = []

    def reg(task_obj, deps=None):
        # like asyncflow: the task is submitted on registration, awaiting is optional
        calls.append((task_obj["name"], task_obj["args"], deps))
        result = {}
        if task_obj["name"] == "crit":
            result = metrics(task_obj, sum(1 for c in calls if c[0] == "crit"))
        future = asyncio.get_running_loop().create_future()
        future.set_result(result)
        return future

    return reg, calls


def windows(calls, name):
    return [args[0] for task_name, args, _ in calls if task_name == name]


# ---------------------------------------------------------------------------
# UQ
# ---------------------------------------------------------------------------


def make_uq(metrics=None, uncertainty=False):
    learner = SeqUQLearner(MagicMock(spec=WorkflowEngine))
    learner.simulation_function = task("sim")
    learner.training_function = task("train")
    learner.prediction_function = task("pred")
    learner.active_learn_function = task("acl")
    if metrics is not None:
        learner.criterion_function = criterion()
    if uncertainty:
        learner.uncertainty_function = task(
            "uq", operator="<", threshold=1.0, uq_metric_name="predictive_entropy"
        )
    reg, calls = recorder(metrics)
    learner._register_task = reg
    return learner, calls


@pytest.mark.asyncio
async def test_uq_window_goes_to_every_model_simulation():
    learner, calls = make_uq()
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(learner.start(model_names=["m1", "m2"], stream=stream))

    sims = windows(calls, "sim")
    assert sims == [[0], [0], [1], [1]]
    assert sims[0] is not sims[1]  # each model gets its own copy
    assert [s.model_ready for s in states] == [True, True]  # no gate configured


@pytest.mark.parametrize(
    ("publish_on", "expected"), [("all", [False, True]), ("any", [True, True])]
)
@pytest.mark.asyncio
async def test_uq_publish_on(publish_on, expected):
    # iteration 0: only m1 meets the criterion; iteration 1: both do
    def metrics(task_obj, n):
        return 2.0 if n == 2 else 0.5

    learner, calls = make_uq(metrics)
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(
        learner.start(model_names=["m1", "m2"], stream=stream, publish_on=publish_on)
    )

    assert [s.model_ready for s in states] == expected
    assert len(windows(calls, "sim")) == 4  # no model retired in stream mode


@pytest.mark.asyncio
async def test_uq_threshold_in_stream_mode_runs_full_iteration():
    learner, calls = make_uq(uncertainty=True)
    learner._check_uncertainty = MagicMock(return_value=(True, 0.1))
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(learner.start(model_names=["m1"], stream=stream))

    assert len(states) == 2  # one yield per iteration, loop not ended
    assert all(s.model_ready and s.should_stop for s in states)
    assert sum(1 for c in calls if c[0] == "acl") == 2


@pytest.mark.asyncio
async def test_uq_invalid_publish_on():
    learner, _ = make_uq()
    with pytest.raises(ValueError, match="publish_on"):
        await collect(learner.start(model_names=["m1"], stream=Stream(), publish_on="most"))


@pytest.mark.asyncio
async def test_parallel_uq_broadcast_and_stop_reason(monkeypatch):
    reg, calls = recorder()

    def child_reg(self, task_obj, deps=None):
        return reg(task_obj, deps)

    monkeypatch.setattr(SeqUQLearner, "_register_task", child_reg)
    learner = ParallelUQLearner(MagicMock(spec=WorkflowEngine))
    learner.simulation_function = task("sim")
    learner.training_function = task("train")
    learner.prediction_function = task("pred")
    learner.active_learn_function = task("acl")
    tracker = MagicMock()
    learner._trackers.append(tracker)
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(
        learner.start(learner_names=["a", "b"], model_names=["m1"], stream=stream)
    )

    assert sorted(windows(calls, "sim")) == [[0], [0], [1], [1]]
    assert len(states) == 4
    assert tracker.on_stop.call_args.args[1] == "stream_exhausted"


# ---------------------------------------------------------------------------
# RL
# ---------------------------------------------------------------------------


def make_rl(cls=SequentialReinforcementLearner, metrics=None):
    learner = cls(MagicMock(spec=WorkflowEngine))
    learner.environment_function = task("env")
    learner.update_function = task("update")
    if metrics is not None:
        learner.criterion_function = criterion()
    reg, calls = recorder(metrics)
    learner._register_task = reg
    return learner, calls


@pytest.mark.asyncio
async def test_rl_window_to_environment_without_criterion():
    learner, calls = make_rl()
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(learner.start(stream=stream))

    assert windows(calls, "env") == [[0], [1]]
    assert [deps for name, _, deps in calls if name == "env"] == [None, None]
    assert all(s.model_ready for s in states)


@pytest.mark.asyncio
async def test_rl_no_criterion_batch_mode_runs():
    # pre-existing UnboundLocalError: stop_task was unset without a criterion
    learner, calls = make_rl()
    states = await collect(learner.start(max_iter=2))
    assert len(states) == 2


@pytest.mark.asyncio
async def test_rl_window_to_update_with_skip_environment_step():
    learner, calls = make_rl()
    stream = Stream(batch_size=1, sources=numbers(2))

    await collect(learner.start(stream=stream, skip_environment_step=True))

    assert windows(calls, "update") == [[0], [1]]
    assert windows(calls, "env") == []


@pytest.mark.asyncio
async def test_rl_criterion_gates_publishing():
    learner, _ = make_rl(metrics=lambda task_obj, n: 0.5 if n == 1 else 2.0)
    published = []
    learner.on_model_ready(published.append)
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(learner.start(stream=stream))

    assert [s.model_ready for s in states] == [True, False]
    assert published == [states[0]]


def make_experience(skip_env=False):
    learner = ParallelExperience(MagicMock(spec=WorkflowEngine))
    learner.environment_functions = {"e1": task("env"), "e2": task("env")}
    learner.update_function = task("update")
    learner.merge_banks = MagicMock()
    reg, calls = recorder()
    learner._register_task = reg
    return learner, calls


@pytest.mark.asyncio
async def test_experience_window_to_every_environment():
    learner, calls = make_experience()
    tracker = MagicMock()
    learner._trackers.append(tracker)
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(learner.start(stream=stream))

    envs = windows(calls, "env")
    assert envs == [[0], [0], [1], [1]]
    assert envs[0] is not envs[1]
    assert tracker.on_iteration.call_count == len(states) == 2
    assert tracker.on_stop.call_args.args[1] == "stream_exhausted"


@pytest.mark.asyncio
async def test_experience_skip_environment_step():
    learner, calls = make_experience()
    stream = Stream(batch_size=1, sources=numbers(2))

    await collect(learner.start(stream=stream, skip_environment_step=True))

    assert windows(calls, "update") == [[0], [1]]
    assert windows(calls, "env") == []
    learner.merge_banks.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_rl_broadcast_and_skip_environment_step(monkeypatch):
    reg, calls = recorder()

    def child_reg(self, task_obj, deps=None):
        return reg(task_obj, deps)

    monkeypatch.setattr(SequentialReinforcementLearner, "_register_task", child_reg)
    learner = ParallelReinforcementLearner(MagicMock(spec=WorkflowEngine))
    learner.update_function = task("update")  # no environment: skip_environment_step
    stream = Stream(batch_size=1, sources=numbers(2))

    states = await collect(
        learner.start(parallel_learners=2, stream=stream, skip_environment_step=True)
    )

    assert sorted(windows(calls, "update")) == [[0], [0], [1], [1]]
    assert len(states) == 4


@pytest.mark.asyncio
async def test_parallel_rl_stop_unblocks_waiting_learners(monkeypatch):
    reg, _ = recorder()

    def child_reg(self, task_obj, deps=None):
        return reg(task_obj, deps)

    monkeypatch.setattr(SequentialReinforcementLearner, "_register_task", child_reg)
    learner = ParallelReinforcementLearner(MagicMock(spec=WorkflowEngine))
    learner.environment_function = task("env")
    learner.update_function = task("update")
    tracker = MagicMock()
    learner._trackers.append(tracker)

    pending = asyncio.ensure_future(collect(learner.start(parallel_learners=2, stream=Stream())))
    await asyncio.sleep(0.1)
    learner.stop()

    assert await asyncio.wait_for(pending, timeout=5.0) == []
    assert tracker.on_stop.call_args.args[1] == "stopped"


@pytest.mark.asyncio
async def test_uq_criterion_uses_per_iteration_config():
    from rose.learner import TaskConfig
    from rose.uncertainty_quantification.uq_learner import UQLearnerConfig

    learner, calls = make_uq(metrics=lambda task_obj, n: 0.5)
    seen = []
    reg = learner._register_task

    def spy(task_obj, deps=None):
        if task_obj["name"] == "crit":
            seen.append(dict(task_obj["kwargs"]))
        return reg(task_obj, deps)

    learner._register_task = spy
    config = UQLearnerConfig(criterion=lambda i: TaskConfig(kwargs={"it": i}))

    await collect(learner.start(model_names=["m1"], max_iter=1, learning_config=config))

    assert seen == [{"it": 0, "--model_name": "m1"}]
