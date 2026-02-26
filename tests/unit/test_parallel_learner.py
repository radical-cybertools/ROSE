from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import (
    LearnerConfig,
    ParallelActiveLearner,
    SequentialActiveLearner,
)
from rose.learner import IterationState


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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

        with patch("rose.al.active_learner.LearnerConfig") as mock_learner_config:
            result = parallel_learner._convert_to_sequential_config(mock_config)

            mock_learner_config.assert_called_once_with(
                simulation="sim_params",
                training="train_params",
                active_learn="active_params",
                criterion="criterion_params",
            )

    @pytest.mark.asyncio
    async def test_start_validation_errors(self, parallel_learner):
        """Test start method validation and error cases."""
        # Test with parallel_learners < 2
        parallel_learner.simulation_function = None
        parallel_learner.training_function = None
        parallel_learner.active_learn_function = None
        parallel_learner.criterion_function = None

        with pytest.raises(ValueError, match="For single learner, use SequentialActiveLearner"):
            await parallel_learner.start(parallel_learners=1)

        # Test with missing simulation functions it should raise error about
        # simulation first
        with pytest.raises(
            ValueError,
            match="Simulation function must be set when not using simulation pool!",
        ):
            await parallel_learner.start(parallel_learners=2, max_iter=1)

        # Test with missing simulation functions and skip_simulation_step
        # it should raise an error about missing train/active_learn tasks
        with pytest.raises(
            ValueError,
            match="Training and Active Learning functions must be set!",
        ):
            await parallel_learner.start(parallel_learners=2, max_iter=1, skip_simulation_step=True)

        # Set functions but test missing stop criteria
        parallel_learner.simulation_function = AsyncMock()
        parallel_learner.training_function = AsyncMock()
        parallel_learner.active_learn_function = AsyncMock()

        with pytest.raises(
            Exception,
            match="Either max_iter > 0 or criterion_function must be provided.",
        ):
            await parallel_learner.start(parallel_learners=2, max_iter=0)

        # Test learner_configs length mismatch
        parallel_learner.criterion_function = AsyncMock()
        learner_configs = [None]  # Only 1 config for 2 learners

        with pytest.raises(ValueError, match="learner_configs length must match parallel_learners"):
            await parallel_learner.start(
                parallel_learners=2, max_iter=1, learner_configs=learner_configs
            )

    @pytest.mark.asyncio
    async def test_start_successful_parallel_execution(self, configured_parallel_learner):
        """Test successful parallel execution of multiple learners."""

        # Create a mock that yields one state then stops (async generator)
        async def mock_start(*args, **kwargs):
            yield IterationState(iteration=0, metric_value=0.5, should_stop=True)

        mock_sequential = MagicMock()
        mock_sequential.start = mock_start
        mock_sequential.metric_values_per_iteration = {"metric1": [1, 2, 3]}

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            return_value=mock_sequential,
        ):
            results = await configured_parallel_learner.start(parallel_learners=2, max_iter=1)

            # Verify results
            assert len(results) == 2
            # Results are IterationState objects
            assert all(isinstance(r, IterationState) for r in results)

            # Verify metric collection
            assert "learner-0" in configured_parallel_learner.metric_values_per_iteration
            assert "learner-1" in configured_parallel_learner.metric_values_per_iteration

    @pytest.mark.asyncio
    async def test_start_learner_failure_handling(self, configured_parallel_learner):
        """Test handling of learner failures in parallel execution."""

        # Mock one successful and one failing sequential learner
        def create_learner_side_effect(learner_id, config):
            mock_learner = MagicMock()
            mock_learner.metric_values_per_iteration = {}

            if learner_id == 0:

                async def success_start(*args, **kwargs):
                    yield IterationState(iteration=0, should_stop=True)

                mock_learner.start = success_start
            else:

                async def fail_start(*args, **kwargs):
                    raise Exception("Learner failed")
                    yield  # Make it a generator

                mock_learner.start = fail_start

            return mock_learner

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            side_effect=create_learner_side_effect,
        ):
            # Mock print to capture error message
            with patch("builtins.print") as mock_print:
                # Should raise exception due to learner failure
                with pytest.raises(Exception, match="Learner failed"):
                    await configured_parallel_learner.start(parallel_learners=2, max_iter=1)

                # Verify error was printed (learner 1 fails, not 0)
                mock_print.assert_any_call("ActiveLearner-1] failed with error: Learner failed")
