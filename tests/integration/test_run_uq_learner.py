from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from rose.metrics import MEAN_SQUARED_ERROR_MSE, PREDICTIVE_ENTROPY
from rose.uq import UQ_REGISTRY, register_uq
from rose.uq.uq_active_learner import ParallelUQLearner, SeqUQLearner


@pytest.mark.asyncio
async def test_active_learning_pipeline_functions():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = ParallelUQLearner(asyncflow)

    @learner.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs):
        return [1, 2, 3, 4]

    @learner.training_task(as_executable=False)
    async def training(data, **kwargs):
        return {"mean": sum(data) / len(data)}

    @learner.active_learn_task(as_executable=False)
    async def active_learn(sim, trained_model, **kwargs):
        return abs(trained_model["mean"] - 2.5)

    @learner.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1, as_executable=False
    )
    async def check_mse(*args, **kwargs):
        # Return a numeric value for the criterion check
        return 0.05

    @learner.prediction_task(as_executable=False)
    async def prediction(trained_model, **kwargs):
        return abs(trained_model["mean"] - 2.5)

    @learner.uncertainty_quantification(
        uq_metric_name=PREDICTIVE_ENTROPY,
        threshold=1.0,
        query_size=10,
        as_executable=False,
    )
    async def check_uq(*args, **kwargs):
        # UQ function receives prediction results which is a list
        # Calculate mean or just return a simple value for testing
        return 0.5

    # Use the new start() method instead of teach()
    results = await learner.start(
        learner_names=["l1", "l2"],
        learner_configs={"l1": None, "l2": None},
        model_names=["m1"],
        max_iter=2,
    )

    # Verify we got results from both learners
    assert len(results) == 2
    assert all(state is not None for state in results)

    scores = learner.get_metric_results()
    uq_scores = learner.get_uncertainty_results()

    assert scores != {}
    assert uq_scores != {}

    @register_uq("custom_uq")
    def confidence_score(self, mc_preds):
        """
        Custom classification metric: 1 - max predicted probability.
        Lower max prob = higher uncertainty.
        """
        mc_preds, _ = self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)  # [n_instances, n_classes]
        max_prob = np.max(mean_probs, axis=1)
        return 1.0 - max_prob

    print("Available metrics:", list(UQ_REGISTRY.keys()))

    assert "custom_uq" in UQ_REGISTRY

    await learner.shutdown()


@pytest.mark.asyncio
async def test_uqlearner_runs_with_mock_functions():
    engine = MagicMock(spec=WorkflowEngine)
    learner = SeqUQLearner(engine)

    # Mock functions
    learner.simulation_function = {"kwargs": {}}
    learner.training_function = {"kwargs": {}}
    learner.prediction_function = {"kwargs": {}}
    learner.active_learn_function = {"kwargs": {}}
    learner.criterion_function = {
        "kwargs": {},
        "metric_name": "test",
        "threshold": 0.1,
        "operator": "<",
    }
    learner.uncertainty_function = {
        "kwargs": {},
        "uq_metric_name": "test",
        "threshold": 1.0,
        "query_size": 10,
        "operator": "<",
    }

    # Patch internal helpers
    learner._get_iteration_task_config = MagicMock(return_value={"kwargs": {}})
    learner._register_task = AsyncMock(
        side_effect=lambda config, deps=None: {"result": 42}
    )
    learner._check_stop_criterion = MagicMock(return_value=(True, 0.1))
    learner._check_uncertainty = MagicMock(return_value=(False, 0.5))
    learner.build_iteration_state = MagicMock()

    # Use the new start() method
    iteration_count = 0
    async for state in learner.start(
        model_names=["modelA"], max_iter=1, num_predictions=1
    ):
        iteration_count += 1
        # Verify the iteration ran
        assert state is not None

    # Should have run at least one iteration
    assert iteration_count >= 1
