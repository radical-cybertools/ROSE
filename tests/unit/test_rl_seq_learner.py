from unittest.mock import AsyncMock, MagicMock

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.al.reinforcement_learner import (
    SequentialReinforcementLearner,
    TaskConfig,
)

class TestSequentialReinforcementLearner:
    """Test cases for SequentialReinforcementLearner class."""
    
    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock WorkflowEngine for testing."""
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def sequential_learner(self, mock_asyncflow):
        """Create a SequentialReinforcementLearner instance for testing."""
        return SequentialReinforcementLearner(mock_asyncflow)
    
    @pytest.fixture
    def configured_learner(self, sequential_learner):
        """Create a fully configured SequentialReinforcementLearner for testing."""
        sequential_learner.environment_function = AsyncMock(return_value="env_result")
        sequential_learner.update_function = AsyncMock(return_value="update_result")
        sequential_learner.criterion_function = AsyncMock(return_value=False)
        
        # Mock the parent class methods
        sequential_learner._get_iteration_task_config = MagicMock(
            return_value=MagicMock(spec=TaskConfig)
        )
        # Create a function that returns a new coroutine each time
        async def mock_task_coroutine():
            return "task_result"
        
        sequential_learner._register_task = MagicMock(
            side_effect=lambda *args, **kwargs: mock_task_coroutine()
        )
        sequential_learner._check_stop_criterion = MagicMock(return_value=(False, None))

        return sequential_learner

    @pytest.mark.asyncio
    async def test_learn_missing_required_functions(self, sequential_learner):
        """Test that learn raises exception when required functions are missing."""
        
        # Ensure functions are actually None
        sequential_learner.environment_function = None
        sequential_learner.update_function = None
        
        with pytest.raises(ValueError) as excinfo:
            await sequential_learner.learn(max_iter=1)

        assert "Environment function must be defined" in str(excinfo.value) or \
                "Update function must be defined" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_learn_successful_execution(self, configured_learner):
        """Test that learn executes successfully with all functions defined."""

        await configured_learner.learn(max_iter=2)

        # Check that the functions were called
        assert configured_learner.environment_function.call_count == 2
        assert configured_learner.update_function.call_count == 2
        assert configured_learner.criterion_function.call_count == 2
        
        # Check that the asyncflow's run method was called
        assert configured_learner.asyncflow.run.call_count >= 6
        # (2 environment tasks + 2 update tasks + 2 stop criterion checks)
        
    @pytest.mark.asyncio
    async def test_learn_stops_on_criterion(self, configured_learner):
        """Test that learn stops when the stop criterion is met."""
        # Modify the criterion function to return True on the second call
        configured_learner.criterion_function = AsyncMock(side_effect=[False, True])
        configured_learner._check_stop_criterion = MagicMock(side_effect=[(False, None), (True, "stop_reason")])

        await configured_learner.learn(max_iter=5)

        # Check that the functions were called
        assert configured_learner.environment_function.call_count == 2
        assert configured_learner.update_function.call_count == 2
        assert configured_learner.criterion_function.call_count == 2

        # Check that the asyncflow's run method was called
        assert configured_learner.asyncflow.run.call_count >= 6
        # (2 environment tasks + 2 update tasks + 2 stop criterion checks)

    @pytest.mark.asyncio
    async def test_shutdown_calls_asyncflow_shutdown(self, configured_learner):
        """Test that shutdown calls the asyncflow's shutdown method."""
        await configured_learner.shutdown()
        configured_learner.asyncflow.shutdown.assert_called_once()
        await configured_learner.asyncflow.shutdown()
        configured_learner.asyncflow.shutdown.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_get_metric_results_returns_scores(self, configured_learner):
        """Test that get_metric_results returns the metric scores."""
        # Mock some metric results
        configured_learner.metric_results = {
            'MODEL_REWARD': [10, 20, 30]
        }
        scores = configured_learner.get_metric_results()
        assert scores == {'MODEL_REWARD': [10, 20, 30]}

    @pytest.mark.asyncio
    async def test_learn_raises_without_iterations_or_criterion(self, sequential_learner):
        """Test that learn raises exception when neither max_iter nor criterion_function is provided."""
        sequential_learner.environment_function = AsyncMock()
        sequential_learner.update_function = AsyncMock()
        with pytest.raises(ValueError) as excinfo:
            await sequential_learner.learn()
        assert "must be defined" in str(excinfo.value)
        sequential_learner.update_function = AsyncMock()
        sequential_learner.environment_function = AsyncMock()
        sequential_learner.update_function = AsyncMock()
        sequential_learner.criterion_function = None
        
        with pytest.raises(
            ValueError,
            match="Either max_iter > 0 or criterion_function must be provided.",
        ):
            await sequential_learner.learn(max_iter=0)

    @pytest.mark.asyncio
    async def test_learn_with_max_iterations(self, configured_learner):
        """Test learn method runs for specified max_iter iterations."""
        # The configured_learner fixture already sets up proper coroutines

        # Run with max_iter=2
        await configured_learner.learn(max_iter=2)

        # Verify _register_task was called for environment and update tasks
        # Should be called for: 2 environment tasks + 2 update tasks + 2 stop criterion checks = 6 times
        assert configured_learner._register_task.call_count >= 4


