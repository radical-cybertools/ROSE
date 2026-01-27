from unittest.mock import AsyncMock, MagicMock

import pytest
from radical.asyncflow import WorkflowEngine

from rose.rl.reinforcement_learner import SequentialReinforcementLearner, TaskConfig


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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

        # Use AsyncMock for _register_task to avoid unawaited coroutine warnings
        sequential_learner._register_task = AsyncMock(return_value="task_result")
        sequential_learner._check_stop_criterion = MagicMock(return_value=(False, None))

        return sequential_learner

    @pytest.mark.asyncio
    async def test_start_missing_required_functions(self, sequential_learner):
        """Test that start raises exception when required functions are missing."""

        # Ensure functions are actually None
        sequential_learner.environment_function = None
        sequential_learner.update_function = None

        with pytest.raises(ValueError, match="Environment function must be set"):
            async for _ in sequential_learner.start(max_iter=1):
                pass

        # Test with environment set but no update
        sequential_learner.environment_function = AsyncMock()
        with pytest.raises(ValueError, match="Update function must be set"):
            async for _ in sequential_learner.start(max_iter=1):
                pass

    @pytest.mark.asyncio
    async def test_start_successful_execution(self, configured_learner):
        """Test that start executes successfully with all functions defined."""
        states = []
        async for state in configured_learner.start(max_iter=2):
            states.append(state)

        # Should yield 2 states (one per iteration)
        assert len(states) == 2

        # Check that the functions were called
        assert configured_learner._register_task.call_count >= 4

    @pytest.mark.asyncio
    async def test_start_stops_on_criterion(self, configured_learner):
        """Test that start stops when the stop criterion is met."""
        # Modify the criterion function to return True on the second call
        configured_learner._check_stop_criterion = MagicMock(
            side_effect=[(False, 0.5), (True, 0.01)]
        )

        states = []
        async for state in configured_learner.start(max_iter=5):
            states.append(state)

        # Should yield 2 states then stop
        assert len(states) == 2

    @pytest.mark.asyncio
    async def test_start_without_iterations_or_criterion(self, sequential_learner):
        """Test that start raises exception when neither max_iter nor
        criterion_function is provided."""
        sequential_learner.environment_function = AsyncMock()
        sequential_learner.update_function = AsyncMock()

        with pytest.raises(
            ValueError, match="Either max_iter > 0 or criterion_function"
        ):
            async for _ in sequential_learner.start():
                pass

    @pytest.mark.asyncio
    async def test_start_with_max_iterations(self, configured_learner):
        """Test start method runs for specified max_iter iterations."""
        states = []
        async for state in configured_learner.start(max_iter=2):
            states.append(state)

        # Should yield 2 states
        assert len(states) == 2

        # Verify _register_task was called
        assert configured_learner._register_task.call_count >= 4

    @pytest.mark.asyncio
    async def test_start_early_break(self, configured_learner):
        """Test that user can break out of start() early."""
        # Set up criterion to never stop
        configured_learner._check_stop_criterion.return_value = (False, 0.5)

        count = 0
        async for state in configured_learner.start(max_iter=10, skip_pre_loop=True):
            count += 1
            if count >= 3:
                break

        # Should have only run 3 iterations
        assert count == 3

    @pytest.mark.asyncio
    async def test_set_next_config(self, configured_learner):
        """Test that set_next_config updates pending config."""
        from rose.learner import LearnerConfig, TaskConfig

        # Initially no pending config
        assert configured_learner._pending_config is None

        # Set a config
        new_config = LearnerConfig(
            update=TaskConfig(kwargs={"learning_rate": 0.001})
        )
        configured_learner.set_next_config(new_config)

        # Should be stored as pending
        assert configured_learner._pending_config == new_config

    @pytest.mark.asyncio
    async def test_iteration_state_attributes(self, configured_learner):
        """Test that IterationState has expected attributes."""
        configured_learner._check_stop_criterion.return_value = (True, 0.05)

        states = []
        async for state in configured_learner.start(max_iter=1, skip_pre_loop=True):
            states.append(state)

        assert len(states) == 1
        state = states[0]

        # Check core attributes
        assert state.iteration == 0
        assert state.metric_value == 0.05
        assert state.should_stop is True
