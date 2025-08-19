from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import (
    LearnerConfig,
    ParallelActiveLearner,
    SequentialActiveLearner,
)


class TestParallelActiveLearner:
    """Test cases for ParallelActiveLearner class."""

    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock WorkflowEngine for testing."""
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def parallel_learner(self, mock_asyncflow):
        """Create a ParallelActiveLearner instance for testing."""
        return ParallelActiveLearner(mock_asyncflow)

    @pytest.fixture
    def configured_parallel_learner(self, parallel_learner):
        """Create a fully configured ParallelActiveLearner for testing."""
        parallel_learner.simulation_function = AsyncMock(return_value="sim_result")
        parallel_learner.training_function = AsyncMock(return_value="train_result")
        parallel_learner.active_learn_function = AsyncMock(return_value="active_result")
        parallel_learner.criterion_function = AsyncMock(return_value=False)
        return parallel_learner

    def test_create_sequential_learner(self, configured_parallel_learner):
        """Test _create_sequential_learner method creates properly configured learner."""
        learner_id = 1
        config = None

        sequential = configured_parallel_learner._create_sequential_learner(learner_id, config)

        # Verify it's a SequentialActiveLearner instance
        assert isinstance(sequential, SequentialActiveLearner)

        # Verify functions are copied
        assert sequential.simulation_function == configured_parallel_learner.simulation_function
        assert sequential.training_function == configured_parallel_learner.training_function
        assert sequential.active_learn_function == configured_parallel_learner.active_learn_function
        assert sequential.criterion_function == configured_parallel_learner.criterion_function

        # Verify learner_id is set
        assert sequential.learner_id == learner_id

    def test_convert_to_sequential_config(self, parallel_learner):
        """Test _convert_to_sequential_config method."""
        # Test with None input
        result = parallel_learner._convert_to_sequential_config(None)
        assert result is None

        # Test with actual config
        mock_config = MagicMock(spec=LearnerConfig)
        mock_config.simulation = "sim_params"
        mock_config.training = "train_params"
        mock_config.active_learn = "active_params"
        mock_config.criterion = "criterion_params"

        with patch('rose.al.active_learner.LearnerConfig') as MockLearnerConfig:
            result = parallel_learner._convert_to_sequential_config(mock_config)

            MockLearnerConfig.assert_called_once_with(
                simulation="sim_params",
                training="train_params",
                active_learn="active_params",
                criterion="criterion_params"
            )

    @pytest.mark.asyncio
    async def test_teach_validation_errors(self, parallel_learner):
        """Test teach method validation and error cases."""
        # Test with parallel_learners < 2
        with pytest.raises(ValueError, match="For single learner, use SequentialActiveLearner"):
            await parallel_learner.teach(parallel_learners=1)

        # Test with missing functions
        with pytest.raises(Exception, match="Simulation, Training, and Active Learning functions must be set!"):
            await parallel_learner.teach(parallel_learners=2, max_iter=1)

        # Set functions but test missing stop criteria
        parallel_learner.simulation_function = AsyncMock()
        parallel_learner.training_function = AsyncMock()
        parallel_learner.active_learn_function = AsyncMock()

        with pytest.raises(Exception, match="Either max_iter or stop_criterion_function must be provided."):
            await parallel_learner.teach(parallel_learners=2, max_iter=0)

        # Test learner_configs length mismatch
        parallel_learner.criterion_function = AsyncMock()
        learner_configs = [None]  # Only 1 config for 2 learners

        with pytest.raises(ValueError, match="learner_configs length must match parallel_learners"):
            await parallel_learner.teach(
                parallel_learners=2,
                max_iter=1,
                learner_configs=learner_configs
            )

    @pytest.mark.asyncio
    async def test_teach_successful_parallel_execution(self, configured_parallel_learner):
        """Test successful parallel execution of multiple learners."""
        # Mock the sequential learner creation and execution
        mock_sequential = MagicMock(spec=SequentialActiveLearner)
        mock_sequential.teach = AsyncMock(return_value="learner_result")
        mock_sequential.metric_values_per_iteration = {"metric1": [1, 2, 3]}

        with patch.object(configured_parallel_learner, '_create_sequential_learner',
                         return_value=mock_sequential):
            with patch.object(configured_parallel_learner, '_convert_to_sequential_config',
                             return_value=None):

                results = await configured_parallel_learner.teach(
                    parallel_learners=2,
                    max_iter=1
                )

                # Verify results
                assert len(results) == 2
                assert all(result == "learner_result" for result in results)

                # Verify sequential learners were called
                assert mock_sequential.teach.call_count == 2

                # Verify metric collection
                assert "learner-0" in configured_parallel_learner.metric_values_per_iteration
                assert "learner-1" in configured_parallel_learner.metric_values_per_iteration

    @pytest.mark.asyncio
    async def test_teach_learner_failure_handling(self, configured_parallel_learner):
        """Test handling of learner failures in parallel execution."""
        # Create a mock sequential learner that fails
        mock_sequential = MagicMock(spec=SequentialActiveLearner)
        mock_sequential.teach = AsyncMock(side_effect=Exception("Learner failed"))

        with patch.object(configured_parallel_learner, '_create_sequential_learner',
                         return_value=mock_sequential):
            with patch.object(configured_parallel_learner, '_convert_to_sequential_config',
                             return_value=None):
                with patch('builtins.print') as mock_print:  # Mock print to capture error message

                    # Should raise exception due to learner failure
                    with pytest.raises(Exception, match="Learner failed"):
                        await configured_parallel_learner.teach(
                            parallel_learners=2,
                            max_iter=1
                        )

                    # Verify error was printed
                    mock_print.assert_any_call("ActiveLearner-0] failed with error: Learner failed")
