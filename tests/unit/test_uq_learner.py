from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.uq import UQScorer
from rose.uq.uq_activeLearner import ParallelUQLearner, SeqUQLearner
from rose.uq.uq_learner import UQLearnerConfig



class TestParallelUQLearner:
    """Test cases for ParallelUQLearner class."""

    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock WorkflowEngine for testing."""
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def parallel_learner(self, mock_asyncflow):
        """Create a ParallelUQLearner instance for testing."""
        return ParallelUQLearner(mock_asyncflow)

    @pytest.fixture
    def configured_parallel_learner(self, parallel_learner):
        """Create a fully configured ParallelUQLearner for testing."""
        parallel_learner.simulation_function = AsyncMock(return_value="sim_result")
        parallel_learner.training_function = AsyncMock(return_value="train_result")
        parallel_learner.active_learn_function = AsyncMock(return_value="active_result")
        parallel_learner.prediction_function = AsyncMock(
            return_value="prediction_result"
        )
        parallel_learner.criterion_function = AsyncMock(return_value=False)
        parallel_learner.uncertainty_function = AsyncMock(return_value=False)
        return parallel_learner

    def test_create_sequential_learner(self, configured_parallel_learner):
        """Test _create_sequential_learner
        method creates properly configured learner."""
        learner_name = "learner-0"

        sequential = configured_parallel_learner._create_sequential_learner(
            learner_name
        )

        # Verify it's a UQLearner instance
        assert isinstance(sequential, SeqUQLearner)

        # Verify functions are copied
        assert (
            sequential.simulation_function
            == configured_parallel_learner.simulation_function
        )
        assert (
            sequential.training_function
            == configured_parallel_learner.training_function
        )
        assert (
            sequential.active_learn_function
            == configured_parallel_learner.active_learn_function
        )
        assert (
            sequential.criterion_function
            == configured_parallel_learner.criterion_function
        )

        # Verify learner_name is set
        assert sequential.learner_name == learner_name

    def test_convert_to_sequential_config(self, parallel_learner):
        """Test _convert_to_sequential_config method."""
        # Test with None input
        result = parallel_learner._convert_to_sequential_config(None)
        assert result is None

        def ensure_task(val):
            if isinstance(val, str):
                return {"name": val, "params": {}}
            return val
        
        # Test with actual config
        mock_config = MagicMock(spec=UQLearnerConfig)
        sim = ensure_task("sim_params")
        train = ensure_task("train_params")
        active = ensure_task("active_params")
        crit = ensure_task("criterion_params")
        pred = ensure_task("prediction_params")
        uncert = ensure_task("uncertainty_params")

        mock_config.simulation = sim
        mock_config.training = train
        mock_config.active_learn = active
        mock_config.criterion = crit
        mock_config.prediction = pred
        mock_config.uncertainty = uncert

        with patch("rose.uq.uq_activeLearner.UQLearnerConfig") as mock_lrnr_config:
            result = parallel_learner._convert_to_sequential_config(mock_config)

            mock_lrnr_config.assert_called_once_with(
                simulation=sim,
                training=train,
                active_learn=active,
                criterion=crit,
                prediction=pred,
                uncertainty=uncert,
            )

    @pytest.mark.asyncio
    async def test_teach_validation_errors(self, parallel_learner):
        """Test teach method validation and error cases."""

        # Test with missing functions
        with pytest.raises(
            Exception,
            match="Simulation, Training, and Active Learning functions must be set!",
        ):
            await parallel_learner.teach(
                learner_names=["l1", "l2"],
                learner_configs={"l1": None, "l2": None},
                model_names=["m1"],
                max_iter=1,
            )

        # Set functions but test missing stop criteria
        parallel_learner.simulation_function = AsyncMock()
        parallel_learner.training_function = AsyncMock()
        parallel_learner.active_learn_function = AsyncMock()
        parallel_learner.prediction_function = AsyncMock()
        parallel_learner.uncertainty_function = AsyncMock()

        # Test with missing functions
        with pytest.raises(
            Exception,
            match="learner_configs length must match learner_names",
        ):
            await parallel_learner.teach(
                learner_names=["l1", "l2"],
                learner_configs={"l1": None},
                model_names=["m1"],
                max_iter=1,
            )

        with pytest.raises(
            Exception,
            match="Either max_iter or stop_criterion_function must be provided.",
        ):
            await parallel_learner.teach(
                learner_names=["l1", "l2"], model_names=["m1"], max_iter=0
            )

        with pytest.raises(
            Exception,
            match="task_type must be classification or regression",
        ):
            _ = UQScorer(task_type="task_type")

    @pytest.mark.asyncio
    async def test_scorer_validation_errors(self, parallel_learner):
        """Test teach method validation and error cases."""

        with pytest.raises(
            Exception,
            match="task_type must be classification or regression",
        ):
            scorer = UQScorer(task_type="task_type")

        scorer = UQScorer(task_type="classification")
        assert scorer.task_type == "classification"

        mc_preds = [[1, 2], [3, 4, 5]]
        y_true = [[1, 2], [3, 4, 5]]
        with pytest.raises(
            TypeError, match="Fail to convert <class 'list'> mc_preds to numpy"
        ):
            scorer._validate_inputs(mc_preds)

        mc_preds = np.ones((2, 3, 4))
        with pytest.raises(
            TypeError, match="Fail to convert <class 'list'> y_true to numpy"
        ):
            scorer._validate_inputs(mc_preds, y_true)

        mc_preds = np.ones((2, 3))
        y_true = np.ones(2)
        with pytest.raises(
            ValueError,
            match=r"For classification, mc_preds must have\s+3 "
            r"dimensions.*got shape \(.+\)",
        ):
            scorer._validate_inputs(mc_preds)

        mc_preds = np.ones((2, 3, 4))
        y_true = np.ones((2, 3, 4))
        with pytest.raises(
            ValueError,
            match=r"For classification, y_true must have "
            r"2 dimensions \[n_instances, n_classes\], "
            r"got shape \(.+\)",
        ):
            scorer._validate_inputs(mc_preds, y_true)

        scorer = UQScorer(task_type="regression")
        assert scorer.task_type == "regression"

        mc_preds = mc_preds = [[1, 2], [3, 4, 5]]
        y_true = mc_preds = [[1, 2], [3, 4, 5]]
        with pytest.raises(
            TypeError, match="Fail to convert <class 'list'> mc_preds to numpy"
        ):
            scorer._validate_inputs(mc_preds)

        mc_preds = np.ones((2, 3))
        with pytest.raises(
            TypeError, match="Fail to convert <class 'list'> y_true to numpy"
        ):
            scorer._validate_inputs(mc_preds, y_true)

        mc_preds = np.ones((2, 3, 4))
        y_true = np.ones((2, 3))
        with pytest.raises(
            ValueError,
            match=r"For regression, mc_preds must have 2 dimensions "
            r"\[n_mc_samples, n_instances\], got shape \(.+\)",
        ):
            scorer._validate_inputs(mc_preds)

        mc_preds = np.ones((2, 3))
        with pytest.raises(
            ValueError,
            match=r"For regression, y_true must have "
            r"1 dimension \[n_instances\], "
            r"y_true shape is \(.+\)",
        ):
            scorer._validate_inputs(mc_preds, y_true)

    @pytest.mark.asyncio
    async def test_teach_successful_parallel_execution(
        self, configured_parallel_learner
    ):
        """Test successful parallel execution of multiple learners."""
        # Mock the sequential learner creation and execution
        mock_sequential = MagicMock(spec=SeqUQLearner)
        mock_sequential.teach = AsyncMock(return_value="learner_result")
        mock_sequential.metric_values_per_iteration = {"metric1": [1, 2, 3]}
        mock_sequential.uncertainty_values_per_iteration = {"UQmetric1": [1, 2, 3]}

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            return_value=mock_sequential,
        ):
            with patch.object(
                configured_parallel_learner,
                "_convert_to_sequential_config",
                return_value=None,
            ):
                results = await configured_parallel_learner.teach(
                    learner_names=["l1", "l2"],
                    learner_configs={"l1": None, "l2": None},
                    model_names=["m1"],
                    max_iter=1,
                )

                # Verify results
                assert len(results) == 2
                assert all(result == "learner_result" for result in results)

                # Verify sequential learners were called
                assert mock_sequential.teach.call_count == 2
                print(
                    "metric_values_per_iteration",
                    configured_parallel_learner.metric_values_per_iteration,
                )
                print(
                    "uncertainty_values_per_iteration",
                    configured_parallel_learner.uncertainty_values_per_iteration,
                )

                # Verify metric collection
                assert (
                    "learner-l1"
                    in configured_parallel_learner.metric_values_per_iteration.keys()
                )
                assert (
                    "learner-l2"
                    in configured_parallel_learner.metric_values_per_iteration.keys()
                )

    @pytest.mark.asyncio
    async def test_teach_learner_failure_handling(self, configured_parallel_learner):
        """Test handling of learner failures in parallel execution."""
        # Create a mock sequential learner that fails
        mock_sequential = MagicMock(spec=SeqUQLearner)
        mock_sequential.teach = AsyncMock(side_effect=Exception("Learner failed"))

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            return_value=mock_sequential,
        ):
            with patch.object(
                configured_parallel_learner,
                "_convert_to_sequential_config",
                return_value=None,
            ):
                # Mock print to capture error message
                with patch("builtins.print") as mock_print:
                    # Should raise exception due to learner failure
                    with pytest.raises(Exception, match="Learner failed"):
                        await configured_parallel_learner.teach(
                            learner_names=["l1"],
                            model_names=["m1"],
                            learner_configs={"l1": None},
                            max_iter=1,
                        )

                    # Verify error was printed
                    mock_print.assert_any_call(
                        "[Parallel-Learner-l1] Failed with error: Learner failed"
                    )
