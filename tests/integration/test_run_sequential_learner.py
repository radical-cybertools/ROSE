from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE


@pytest.mark.asyncio
async def test_active_learning_pipeline_functions():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    @acl.simulation_task(as_executable=False)
    async def simulation(*args):
        return [1, 2, 3, 4]

    @acl.training_task(as_executable=False)
    async def training(data):
        return {"mean": sum(data) / len(data)}

    @acl.active_learn_task(as_executable=False)
    async def active_learn(sim, trained_model):
        return abs(trained_model["mean"] - 2.5)

    @acl.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1, as_executable=False
    )
    async def check_mse(*args):
        return 0.05  # Return a metric value below threshold

    async for _state in acl.start(max_iter=5):
        pass  # Let it run to completion

    scores = acl.get_metric_results()

    assert scores != {}

    await acl.shutdown()
