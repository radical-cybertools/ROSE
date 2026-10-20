"""Unit tests for Stream and stream mode of the active learners: windowing, publish gate, stop(),
exhaustion, conflation, source errors and parallel broadcast."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose import Stream
from rose.al.active_learner import ParallelActiveLearner, SequentialActiveLearner


def task(name, **extra):
    return {
        "func": AsyncMock(),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
        "name": name,
        **extra,
    }


def configure(learner, criterion_results=None):
    """Set mock tasks; criterion results are taken from the list (metric, compared '< 1.0')."""
    learner.simulation_function = task("sim")
    learner.training_function = task("train")
    learner.active_learn_function = task("acl")
    if criterion_results is not None:
        learner.criterion_function = task(
            "crit", operator="<", threshold=1.0, metric_name="test_metric"
        )
    return iter(criterion_results or ())


def make_learner(criterion_results=None):
    """Create a sequential learner with mocked task submission; ``calls`` records (task, args)."""
    learner = SequentialActiveLearner(MagicMock(spec=WorkflowEngine))
    metrics = configure(learner, criterion_results)
    calls = []

    async def mock_reg(task_obj, deps=None):
        if task_obj.get("metric_name"):
            return next(metrics)
        calls.append((task_obj["name"], task_obj["args"]))
        return {"last_task": task_obj["name"]}

    learner._register_task = AsyncMock(side_effect=mock_reg)
    return learner, calls


def windows(calls, name="sim"):
    return [args[0] for task_name, args in calls if task_name == name]


async def collect(gen, timeout=5.0):
    async def run():
        return [state async for state in gen]

    return await asyncio.wait_for(run(), timeout)


async def numbers(n, fail=False):
    for i in range(n):
        yield i
    if fail:
        raise RuntimeError("sensor failed")


@pytest.mark.asyncio
async def test_windows_batched_and_source_exhaustion():
    learner, calls = make_learner()
    tracker = MagicMock()
    learner.add_tracker(tracker)
    stream = Stream(batch_size=2, sources=numbers(4))

    states = await collect(learner.start(stream=stream))

    assert windows(calls) == [[0, 1], [2, 3]]
    assert [s.iteration for s in states] == [0, 1]
    assert all(s.window_size == 2 for s in states)
    assert tracker.on_stop.call_args.args[1] == "stream_exhausted"


@pytest.mark.asyncio
async def test_window_goes_to_training_without_simulation():
    learner, calls = make_learner()
    stream = Stream(batch_size=2, sources=numbers(4))

    await collect(learner.start(stream=stream, skip_simulation_step=True))

    assert windows(calls, "train") == [[0, 1], [2, 3]]
    assert windows(calls, "sim") == []


def test_sources_at_construction_need_no_event_loop():
    # pumps are deferred to the first window pull, so construction works outside a loop
    stream = Stream(batch_size=2, sources=numbers(4))
    learner, calls = make_learner()

    async def run():
        states = await collect(learner.start(stream=stream))
        assert windows(calls) == [[0, 1], [2, 3]]
        assert len(states) == 2

    asyncio.run(run())


@pytest.mark.asyncio
async def test_criterion_is_publish_gate_not_termination():
    learner, _ = make_learner(criterion_results=[0.5, 2.0])
    published = []
    learner.on_model_ready(published.append)
    stream = Stream(batch_size=1)
    await stream.feed("a")
    await stream.feed("b")
    stream.close()

    states = await collect(learner.start(stream=stream))

    # criterion met on first window (0.5 < 1.0) but loop continued
    assert [s.should_stop for s in states] == [True, False]
    assert [s.model_ready for s in states] == [True, False]
    assert len(published) == 1 and published[0] is states[0]


@pytest.mark.asyncio
async def test_every_window_publishes_without_criterion():
    learner, _ = make_learner()
    published = []

    async def callback(state):
        published.append(state.iteration)

    learner.on_model_ready(callback)
    learner.on_model_ready(MagicMock(side_effect=RuntimeError("publish failed")))  # ignored

    states = await collect(learner.start(stream=Stream(batch_size=1, sources=numbers(3))))

    assert all(s.model_ready for s in states)
    assert published == [0, 1, 2]


@pytest.mark.asyncio
async def test_stop_unblocks_empty_stream():
    learner, _ = make_learner()

    async def run():
        async for _ in learner.start(stream=Stream()):
            pytest.fail("no data was fed, no state expected")

    task = asyncio.ensure_future(run())
    await asyncio.sleep(0.1)
    learner.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert learner.is_stopped


@pytest.mark.asyncio
async def test_max_iter_caps_without_extra_window():
    learner, calls = make_learner()
    stream = Stream(batch_size=1)
    for i in range(5):
        await stream.feed(i)

    states = await collect(learner.start(stream=stream, max_iter=2))

    assert len(states) == 2
    assert windows(calls) == [[0], [1]]


@pytest.mark.asyncio
async def test_max_wait_flushes_partial_window():
    learner, calls = make_learner()
    stream = Stream(batch_size=10, max_wait=0.05)
    await stream.feed(1)
    await stream.feed(2)

    async def run():
        async for _ in learner.start(stream=stream):
            learner.stop()

    await asyncio.wait_for(run(), timeout=5.0)
    assert windows(calls) == [[1, 2]]


@pytest.mark.asyncio
async def test_conflate_keeps_newest_items():
    learner, calls = make_learner()
    stream = Stream(batch_size=2, conflate=True)
    for i in range(6):
        await stream.feed(i)

    async def run():
        async for _ in learner.start(stream=stream):
            learner.stop()

    await asyncio.wait_for(run(), timeout=5.0)
    assert windows(calls) == [[4, 5]]


@pytest.mark.asyncio
async def test_conflate_bounds_backlog_at_ingestion():
    stream = Stream(batch_size=2, conflate=True)
    for i in range(100):
        await stream.feed(i)
    # backlog is dropped at ingestion, not at window collection
    assert len(stream._backlog) <= 2


@pytest.mark.asyncio
async def test_source_error_is_raised():
    learner, calls = make_learner()
    stream = Stream(batch_size=1, sources=numbers(1, fail=True))

    with pytest.raises(RuntimeError, match="sensor failed"):
        await collect(learner.start(stream=stream))
    assert windows(calls) == [[0]]


@pytest.mark.asyncio
async def test_source_error_is_raised_while_other_sources_run():
    async def endless():
        while True:
            yield "x"
            await asyncio.sleep(0.01)

    learner, _ = make_learner()
    stream = Stream(batch_size=1, sources=[numbers(1, fail=True), endless()])

    with pytest.raises(RuntimeError, match="sensor failed"):
        await collect(learner.start(stream=stream))


@pytest.mark.asyncio
async def test_max_wait_keeps_items_arriving_at_timeout():
    stream = Stream(batch_size=100, max_wait=0.001)
    sub = stream._subscribe()
    received = []

    async def produce():
        for i in range(200):
            await stream.feed(i)
            await asyncio.sleep(0.0005)
        stream.close()

    producer = asyncio.ensure_future(produce())
    while window := await sub.next_window():
        received.extend(window)
    await producer
    assert received == list(range(200))


@pytest.mark.asyncio
async def test_invalid_use_raises():
    with pytest.raises(ValueError):
        Stream(batch_size=0)
    with pytest.raises(ValueError):
        Stream(max_wait=-1.0)

    learner, _ = make_learner()
    with pytest.raises(ValueError, match="skip_pre_loop"):
        async for _ in learner.start(stream=Stream(), skip_pre_loop=True):
            pass

    stream = Stream()
    stream.close()
    with pytest.raises(RuntimeError):
        await stream.feed(1)


@pytest.mark.asyncio
async def test_parallel_broadcasts_every_item(monkeypatch):
    calls = {0: [], 1: []}

    async def mock_reg(self, task_obj, deps=None):
        calls[self.learner_id].append((task_obj["name"], task_obj["args"]))
        return {}

    monkeypatch.setattr(SequentialActiveLearner, "_register_task", mock_reg)
    learner = ParallelActiveLearner(MagicMock(spec=WorkflowEngine))
    configure(learner)
    published = []
    learner.on_model_ready(published.append)
    stream = Stream(batch_size=2)
    for i in range(4):
        await stream.feed(i)
    stream.close()

    states = await collect(learner.start(parallel_learners=2, stream=stream))

    assert windows(calls[0]) == windows(calls[1]) == [[0, 1], [2, 3]]
    assert len(states) == 4
    assert sorted(s.learner_id for s in published) == [0, 0, 1, 1]


@pytest.mark.asyncio
async def test_parallel_stop_unblocks_waiting_learners(monkeypatch):
    monkeypatch.setattr(SequentialActiveLearner, "_register_task", AsyncMock(return_value={}))
    learner = ParallelActiveLearner(MagicMock(spec=WorkflowEngine))
    configure(learner)

    task = asyncio.ensure_future(collect(learner.start(parallel_learners=2, stream=Stream())))
    await asyncio.sleep(0.1)
    learner.stop()
    assert await asyncio.wait_for(task, timeout=5.0) == []
