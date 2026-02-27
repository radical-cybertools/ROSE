from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from radical.asyncflow import WorkflowEngine

from rose.learner import IterationState, LearnerConfig
from rose.rl.reinforcement_learner import (
    ParallelReinforcementLearner,
    SequentialReinforcementLearner,
)


class TestParallelReinforcementLearner:
    """Test cases for ParallelReinforcementLearner class."""

    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock WorkflowEngine for testing."""
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def parallel_learner(self, mock_asyncflow):
        """Create a ParallelReinforcementLearner instance for testing."""
        return ParallelReinforcementLearner(mock_asyncflow)

    @pytest.fixture
    def configured_parallel_learner(self, parallel_learner):
        """Create a fully configured ParallelReinforcementLearner for testing."""
        parallel_learner.environment_function = AsyncMock(return_value="env_result")
        parallel_learner.update_function = AsyncMock(return_value="update_result")
        parallel_learner.criterion_function = AsyncMock(return_value=False)

        return parallel_learner

    def test_init(self, mock_asyncflow):
        """Test ParallelReinforcementLearner initialization."""
        prl = ParallelReinforcementLearner(mock_asyncflow)
        assert prl.asyncflow == mock_asyncflow

    def test_create_sequential_learner(self, configured_parallel_learner):
        """Test _create_sequential_learner method."""
        learner_id = 1
        config = None

        sequential_learner = configured_parallel_learner._create_sequential_learner(
            learner_id, config
        )

        assert isinstance(sequential_learner, SequentialReinforcementLearner)
        assert sequential_learner.learner_id == learner_id
        assert (
            sequential_learner.environment_function
            == configured_parallel_learner.environment_function
        )
        assert sequential_learner.update_function == configured_parallel_learner.update_function
        assert (
            sequential_learner.criterion_function == configured_parallel_learner.criterion_function
        )

    def test_convert_to_sequential_config(self, parallel_learner):
        """Test _convert_to_sequential_config method."""
        # Test with None config
        result = parallel_learner._convert_to_sequential_config(None)
        assert result is None

        # Test with actual config
        mock_config = MagicMock(spec=LearnerConfig)
        mock_config.environment = "env_params"
        mock_config.update = "upd_params"
        mock_config.criterion = "crit_params"

        with patch("rose.rl.reinforcement_learner.LearnerConfig") as mock_learner_config:
            result = parallel_learner._convert_to_sequential_config(mock_config)

            mock_learner_config.assert_called_once_with(
                environment="env_params",
                update="upd_params",
                criterion="crit_params",
            )

    @pytest.mark.asyncio
    async def test_start_missing_environment_function(self, parallel_learner):
        """Test that start raises exception when environment function is missing."""
        parallel_learner.environment_function = None
        parallel_learner.update_function = AsyncMock()

        with pytest.raises(ValueError, match="Environment and Update functions"):
            await parallel_learner.start(parallel_learners=2, max_iter=1)

    @pytest.mark.asyncio
    async def test_start_missing_update_function(self, parallel_learner):
        """Test that start raises exception when update function is missing."""
        parallel_learner.environment_function = AsyncMock()
        parallel_learner.update_function = None

        with pytest.raises(ValueError, match="Environment and Update functions"):
            await parallel_learner.start(parallel_learners=2, max_iter=1)

    @pytest.mark.asyncio
    async def test_start_invalid_parallel_learners_count(self, configured_parallel_learner):
        """Test that start raises exception when parallel_learners < 2."""
        with pytest.raises(ValueError) as excinfo:
            await configured_parallel_learner.start(parallel_learners=1, max_iter=1)

        assert "For single learner, use SequentialReinforcementLearner" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_start_without_iterations_or_criterion(self, configured_parallel_learner):
        """Test that start raises exception when neither max_iter nor criterion_function is
        provided."""
        configured_parallel_learner.criterion_function = None

        with pytest.raises(ValueError, match="Either max_iter > 0 or criterion"):
            await configured_parallel_learner.start(parallel_learners=2)

    @pytest.mark.asyncio
    async def test_start_mismatched_config_length(self, configured_parallel_learner):
        """Test that start raises exception when learner_configs length doesn't match
        parallel_learners."""
        learner_configs = [LearnerConfig(), LearnerConfig()]  # Length 2

        with pytest.raises(ValueError) as excinfo:
            await configured_parallel_learner.start(
                parallel_learners=3,  # Different length
                max_iter=1,
                learner_configs=learner_configs,
            )

        assert "learner_configs length must match parallel_learners" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_start_successful_execution(self, configured_parallel_learner):
        """Test that start executes successfully with parallel learners."""

        # Create a mock that yields one state then stops
        async def mock_start(*args, **kwargs):
            yield IterationState(iteration=0, metric_value=0.5, should_stop=True)

        mock_sequential_learner = MagicMock()
        mock_sequential_learner.start = mock_start
        mock_sequential_learner.metric_values_per_iteration = {"metric1": [1, 2, 3]}

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            return_value=mock_sequential_learner,
        ):
            results = await configured_parallel_learner.start(parallel_learners=2, max_iter=1)

            assert len(results) == 2
            # Results are IterationState objects
            assert all(isinstance(r, IterationState) for r in results)

            # Verify metric storage
            assert "learner-0" in configured_parallel_learner.metric_values_per_iteration
            assert "learner-1" in configured_parallel_learner.metric_values_per_iteration

    @pytest.mark.asyncio
    async def test_start_handles_learner_exceptions(self, configured_parallel_learner):
        """Test that start properly handles exceptions from individual learners."""

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
            # The exception should propagate up and be raised
            with pytest.raises(Exception) as excinfo:
                await configured_parallel_learner.start(parallel_learners=2, max_iter=1)

            assert "Learner failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_start_skip_pre_loop(self, configured_parallel_learner):
        """Test that start passes skip_pre_loop flag to sequential learners."""
        start_calls = []

        async def mock_start(*args, **kwargs):
            start_calls.append(kwargs)
            yield IterationState(iteration=0, should_stop=True)

        mock_sequential_learner = MagicMock()
        mock_sequential_learner.start = mock_start
        mock_sequential_learner.metric_values_per_iteration = {}

        with patch.object(
            configured_parallel_learner,
            "_create_sequential_learner",
            return_value=mock_sequential_learner,
        ):
            await configured_parallel_learner.start(
                parallel_learners=2, max_iter=1, skip_pre_loop=True
            )

            # Verify that sequential learners were called with skip_pre_loop=True
            assert len(start_calls) == 2
            for call in start_calls:
                assert call.get("skip_pre_loop") is True
