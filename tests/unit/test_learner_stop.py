import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import SequentialActiveLearner


@pytest.mark.asyncio
async def test_learner_stop_immediate():
    """Test that stop() terminates the learning loop immediately."""
    mock_asyncflow = MagicMock(spec=WorkflowEngine)
    learner = SequentialActiveLearner(mock_asyncflow)

    # Mock required functions as they are stored by decorators
    learner.simulation_function = {
        "func": AsyncMock(return_value="sim"),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
    }
    learner.training_function = {
        "func": AsyncMock(return_value="train"),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
    }
    learner.active_learn_function = {
        "func": AsyncMock(return_value={"info": "acl"}),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
    }
    learner.criterion_function = {
        "func": AsyncMock(return_value=0.5),
        "args": (),
        "kwargs": {},
        "decor_kwargs": {},
        "operator": ">",
        "threshold": 0.1,
        "metric_name": "test_metric",
    }

    # Mock _register_task to return a future
    async def mock_reg(task_obj, deps=None):
        return "result"

    learner._register_task = AsyncMock(side_effect=mock_reg)
    learner._check_stop_criterion = MagicMock(return_value=(False, 0.5))

    count = 0

    async def run_learner():
        nonlocal count
        async for _state in learner.start(max_iter=100, skip_pre_loop=True):
            count += 1
            if count == 1:
                learner.stop()

    # Run with a timeout to ensure it doesn't hang if stop() fails
    try:
        await asyncio.wait_for(run_learner(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("Learner loop did not terminate after stop() within timeout")

    assert count == 1
    assert learner.is_stopped
