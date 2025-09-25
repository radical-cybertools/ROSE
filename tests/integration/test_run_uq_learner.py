from concurrent.futures import ThreadPoolExecutor

import pytest
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from rose.uq.uq_learner import ParallelUQLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE, PREDICTIVE_ENTROPY


@pytest.mark.asyncio
async def test_active_learning_pipeline_functions():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = ParallelUQLearner(asyncflow)

    @learner.simulation_task(as_executable=False)
    async def simulation(*args):
        return [1, 2, 3, 4]

    @learner.training_task(as_executable=False)
    async def training(data):
        return {"mean": sum(data) / len(data)}

    @learner.active_learn_task(as_executable=False)
    async def active_learn(sim, trained_model):
        return abs(trained_model["mean"] - 2.5)

    @learner.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1, as_executable=False
    )
    async def check_mse(val):
        return val < 2.6

    @learner.prediction_task(as_executable=False)
    async def prediction(sim, trained_model):
        return abs(trained_model["mean"] - 2.5)

    @learner.uncertainty_quantification(
        uq_metric_name=PREDICTIVE_ENTROPY, threshold=1.0, query_size=10
    )
    async def check_uq(val):
        return val < 2.6

    await learner.teach(
        learner_names=["l1", "l2"],
        learner_configs={"l1": None, "l2": None},
        model_names=["m1"],
        max_iter=2,
    )

    scores = learner.get_metric_results()

    assert scores != {}

    await learner.shutdown()
