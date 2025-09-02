from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from rose.rl.reinforcement_learner import SequentialReinforcementLearner
from rose.metrics import GREATER_THAN_THRESHOLD


@pytest.mark.asyncio
async def test_rl_pipeline_functions():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    rl = SequentialReinforcementLearner(asyncflow)

    @rl.environment_task(as_executable=False)
    async def environment(*args):
        return [[1, 2, 3, 4],[2, 4, 6, 8],[3, 6, 9, 12],[4, 8, 12, 16]]

    @rl.update_task(as_executable=False)
    async def update(data):
        return sum([sum(i) for i in data]) / len(data)

    @rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=20, operator=GREATER_THAN_THRESHOLD, as_executable=False)
    async def check_reward(val):
        return val > 15

    await rl.learn(max_iter=2)

    scores = rl.get_metric_results()

    assert scores != {}

    await rl.shutdown()
        
