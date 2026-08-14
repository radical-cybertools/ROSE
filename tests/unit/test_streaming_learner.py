"""Unit tests for StreamingActiveLearner: windowing, publish-gate criterion, stop() and source-
exhaustion termination."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.al.streaming_learner import StreamingActiveLearner


def make_learner(batch_size=2, max_wait=None, conflate=False, criterion_results=None, sources=None):
    """Create a learner with mocked task submission.

    Training results echo the received window; criterion results are taken from the given list
    (metric, compared as '< 1.0').
    """
    learner = StreamingActiveLearner(
        MagicMock(spec=WorkflowEngine),
        batch_size=batch_size,
        max_wait=max_wait,
        conflate=conflate,
        sources=sources,
    )
    learner.training_function = {"func": AsyncMock(), "args": (), "kwargs": {}, "decor_kwargs": {}}
    learner.active_learn_function = {
        "func": AsyncMock(),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
    }
    metrics = iter(criterion_results or ())
    if criterion_results is not None:
        learner.criterion_function = {
            "func": AsyncMock(),
            "args": (),
            "kwargs": {},
            "decor_kwargs": {},
            "operator": "<",
            "threshold": 1.0,
            "metric_name": "test_metric",
        }

    windows = []

    async def mock_reg(task_obj, deps=None):
        if task_obj.get("metric_name"):
            return next(metrics)
        if task_obj["args"]:  # training task: window is first arg
            windows.append(task_obj["args"][0])
        return "result"

    learner._register_task = AsyncMock(side_effect=mock_reg)
    return learner, windows


@pytest.mark.asyncio
async def test_windows_batched_and_source_exhaustion():
    learner, windows = make_learner(batch_size=2)

    async def source():
        for i in range(4):
            yield i

    learner.attach_source(source())

    states = [state async for state in learner.start()]

    assert windows == [[0, 1], [2, 3]]
    assert [s.iteration for s in states] == [0, 1]
    assert all(s.window_size == 2 for s in states)


def test_sources_at_construction_need_no_event_loop():
    # pumps are deferred to start(), so construction works outside a loop
    async def source():
        for i in range(4):
            yield i

    learner, windows = make_learner(batch_size=2, sources=source())

    async def run():
        states = [state async for state in learner.start()]
        assert windows == [[0, 1], [2, 3]]
        assert len(states) == 2

    asyncio.run(run())


@pytest.mark.asyncio
async def test_criterion_is_publish_gate_not_termination():
    learner, _ = make_learner(batch_size=1, criterion_results=[0.5, 2.0])
    published = []
    learner.on_model_ready(published.append)

    await learner.feed("a")
    await learner.feed("b")

    states = []

    async def run():
        async for state in learner.start():
            states.append(state)
            if len(states) == 2:
                learner.stop()

    await asyncio.wait_for(run(), timeout=5.0)

    # criterion met on first window (0.5 < 1.0) but loop continued
    assert [s.should_stop for s in states] == [True, False]
    assert len(published) == 1 and published[0] is states[0]


@pytest.mark.asyncio
async def test_stop_unblocks_empty_queue():
    learner, _ = make_learner()

    async def run():
        async for _ in learner.start():
            pytest.fail("no data was fed, no state expected")

    task = asyncio.ensure_future(run())
    await asyncio.sleep(0.1)
    learner.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert learner.is_stopped


@pytest.mark.asyncio
async def test_max_wait_flushes_partial_window():
    learner, windows = make_learner(batch_size=10, max_wait=0.05)

    await learner.feed(1)
    await learner.feed(2)

    async def run():
        async for _ in learner.start():
            learner.stop()

    await asyncio.wait_for(run(), timeout=5.0)
    assert windows == [[1, 2]]


@pytest.mark.asyncio
async def test_conflate_keeps_newest_items():
    learner, windows = make_learner(batch_size=2, conflate=True)
    for i in range(6):
        await learner.feed(i)

    async def run():
        async for _ in learner.start():
            learner.stop()

    await asyncio.wait_for(run(), timeout=5.0)
    assert windows == [[4, 5]]


@pytest.mark.asyncio
async def test_missing_tasks_raise():
    learner = StreamingActiveLearner(MagicMock(spec=WorkflowEngine))
    with pytest.raises(ValueError):
        async for _ in learner.start():
            pass


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        StreamingActiveLearner(MagicMock(spec=WorkflowEngine), batch_size=0)
    with pytest.raises(ValueError):
        StreamingActiveLearner(MagicMock(spec=WorkflowEngine), max_wait=-1.0)


@pytest.mark.asyncio
async def test_conflate_bounds_queue_at_ingestion():
    learner, _ = make_learner(batch_size=2, conflate=True)
    for i in range(100):
        await learner.feed(i)
    # backlog is dropped at ingestion, not at window collection
    assert learner._queue.qsize() <= 2
