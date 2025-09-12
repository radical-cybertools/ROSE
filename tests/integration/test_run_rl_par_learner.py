from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from rose.rl.reinforcement_learner import ParallelReinforcementLearner
from rose.metrics import GREATER_THAN_THRESHOLD

@pytest.mark.asyncio
async def test_rl_pipeline_functions():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    rl = ParallelReinforcementLearner(asyncflow)

    @rl.environment_task(as_executable=False)
    async def environment(*args):
        return [1, 2, 3, 4]

    @rl.update_task(as_executable=False)
    async def update(data, *args):
        return sum(data) / len(data)

    @rl.as_stop_criterion(
        metric_name="MODEL_REWARD",
        threshold=20,
        operator=GREATER_THAN_THRESHOLD,
        as_executable=False,
    )
    async def check_reward(val, *args):
        return val > 2

    await rl.learn(parallel_learners=5, max_iter=2)

    scores = rl.get_metric_results()

    assert scores != {}

    await rl.shutdown()
