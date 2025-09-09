from unittest.mock import AsyncMock, MagicMock

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.rl.reinforcement_learner import (
    SequentialReinforcementLearner,
    TaskConfig
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
        
        with pytest.raises(Exception) as excinfo:
            await sequential_learner.learn(max_iter=1)

        assert "Environment function must be set" in str(excinfo.value) or \
                "Update function must be set" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_learn_successful_execution(self, configured_learner):
        """Test that learn executes successfully with all functions defined."""

        await configured_learner.learn(max_iter=2)

        # Check that the functions were called
        assert configured_learner._register_task.call_count == 8
        
    @pytest.mark.asyncio
    async def test_learn_stops_on_criterion(self, configured_learner):
        """Test that learn stops when the stop criterion is met."""
        # Modify the criterion function to return True on the second call
        configured_learner.criterion_function = AsyncMock(side_effect=[False, True])
        configured_learner._check_stop_criterion = MagicMock(
            side_effect=[(False, None), (True, "stop_reason")]
            )

        await configured_learner.learn(max_iter=5)

        # Check that the functions were called
        assert configured_learner._register_task.call_count == 6

    @pytest.mark.asyncio
    async def test_learn_without_iterations_or_criterion(self, sequential_learner):
        """Test that learn raises exception when """
        """neither max_iter nor criterion_function is provided."""
        sequential_learner.environment_function = AsyncMock()
        sequential_learner.update_function = AsyncMock()
        with pytest.raises(Exception) as excinfo:
            await sequential_learner.learn()
        assert "Either max_iter or stop_criterion_function" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_learn_with_max_iterations(self, configured_learner):
        """Test learn method runs for specified max_iter iterations."""
        # The configured_learner fixture already sets up proper coroutines

        # Run with max_iter=2
        await configured_learner.learn(max_iter=2)

        # Verify _register_task was called for environment and update tasks
        # Should be called for: 8 times
        assert configured_learner._register_task.call_count == 8


