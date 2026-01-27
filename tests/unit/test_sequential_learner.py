from unittest.mock import AsyncMock, MagicMock

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import SequentialActiveLearner
from rose.learner import TaskConfig


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestSequentialActiveLearner:
    """Test cases for SequentialActiveLearner class."""

    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock WorkflowEngine for testing."""
        return MagicMock(spec=WorkflowEngine)

    @pytest.fixture
    def sequential_learner(self, mock_asyncflow):
        """Create a SequentialActiveLearner instance for testing."""
        return SequentialActiveLearner(mock_asyncflow)

    @pytest.fixture
    def configured_learner(self, sequential_learner):
        """Create a fully configured SequentialActiveLearner for testing."""
        sequential_learner.simulation_function = AsyncMock(return_value="sim_result")
        sequential_learner.training_function = AsyncMock(return_value="train_result")
        sequential_learner.active_learn_function = AsyncMock(
            return_value="active_result"
        )
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
        sequential_learner.simulation_function = None
        sequential_learner.training_function = None
        sequential_learner.active_learn_function = None
        sequential_learner.criterion_function = None

        # Test with no functions set (should fail on simulation function check)
        with pytest.raises(
            ValueError,
            match=r"Simulation function must be set when not using simulation pool!",
        ):
            async for _ in sequential_learner.start(max_iter=1):
                pass

        # Test with only simulation function set
        # (should fail on training/active learning check)
        sequential_learner.simulation_function = AsyncMock()
        with pytest.raises(
            ValueError, match=r"Training and Active Learning functions must be set!"
        ):
            async for _ in sequential_learner.start(max_iter=1):
                pass

        # Test with simulation and training set but no active learning
        # (should still fail on training/active learning check)
        sequential_learner.training_function = AsyncMock()
        with pytest.raises(
            ValueError, match=r"Training and Active Learning functions must be set!"
        ):
            async for _ in sequential_learner.start(max_iter=1):
                pass

    @pytest.mark.asyncio
    async def test_start_missing_stop_criteria(self, sequential_learner):
        """Test that start raises exception when no stopping criteria provided."""
        sequential_learner.simulation_function = AsyncMock()
        sequential_learner.training_function = AsyncMock()
        sequential_learner.active_learn_function = AsyncMock()
        sequential_learner.criterion_function = None

        with pytest.raises(
            ValueError,
            match="Either max_iter > 0 or criterion_function must be provided.",
        ):
            async for _ in sequential_learner.start(max_iter=0):
                pass

    @pytest.mark.asyncio
    async def test_start_with_max_iterations(self, configured_learner):
        """Test start method runs for specified max_iter iterations."""
        # Run with max_iter=2
        states = []
        async for state in configured_learner.start(max_iter=2, skip_pre_loop=True):
            states.append(state)

        # Should yield 2 states (one per iteration)
        assert len(states) == 2

        # Verify _register_task was called for active learning tasks
        assert configured_learner._register_task.call_count >= 4

    @pytest.mark.asyncio
    async def test_start_with_stop_criterion(self, configured_learner):
        """Test start method stops when criterion function returns True."""
        # Set up criterion to stop after first iteration
        configured_learner._check_stop_criterion.side_effect = [
            (True, 0.005),  # Stop on first iteration
        ]

        states = []
        async for state in configured_learner.start(max_iter=0, skip_pre_loop=True):
            states.append(state)

        # Should yield 1 state then stop
        assert len(states) == 1

        # Verify _check_stop_criterion was called
        assert configured_learner._check_stop_criterion.call_count == 1

    @pytest.mark.asyncio
    async def test_start_continues_when_criterion_false(self, configured_learner):
        """Test start method continues when criterion function returns False."""
        # Set up criterion to not stop on first iteration, but stop on second
        configured_learner._check_stop_criterion.side_effect = [
            (False, 0.1),  # Don't stop on first iteration
            (True, 0.005),  # Stop on second iteration
        ]

        states = []
        async for state in configured_learner.start(max_iter=0, skip_pre_loop=True):
            states.append(state)

        # Should yield 2 states
        assert len(states) == 2

        # Verify _check_stop_criterion was called twice
        assert configured_learner._check_stop_criterion.call_count == 2

    @pytest.mark.asyncio
    async def test_start_skip_pre_loop(self, configured_learner):
        """Test start method behavior when skip_pre_loop is True."""
        # Set up stop criterion to stop immediately
        configured_learner._check_stop_criterion.return_value = (True, 0.005)

        states = []
        async for state in configured_learner.start(max_iter=0, skip_pre_loop=True):
            states.append(state)

        # Should yield 1 state
        assert len(states) == 1

        # When skip_pre_loop is True, should not register pre-loop
        # simulation/training tasks
        # Verify the method completes without pre-loop setup
        assert configured_learner._register_task.call_count >= 1

    @pytest.mark.asyncio
    async def test_start_early_break(self, configured_learner):
        """Test that user can break out of start() early."""
        # Set up criterion to never stop
        configured_learner._check_stop_criterion.return_value = (False, 0.5)

        count = 0
        async for _state in configured_learner.start(max_iter=10, skip_pre_loop=True):
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
            training=TaskConfig(kwargs={"--lr": "0.001"})
        )
        configured_learner.set_next_config(new_config)

        # Should be stored as pending
        assert configured_learner._pending_config == new_config
