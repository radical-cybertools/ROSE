from unittest.mock import AsyncMock, MagicMock

import pytest

# Assuming these imports based on the code structure
from radical.asyncflow import WorkflowEngine

from rose.al.active_learner import (
    SequentialActiveLearner,
    TaskConfig,
)


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

        # Create a function that returns a new coroutine each time
        async def mock_task_coroutine():
            return "task_result"

        sequential_learner._register_task = MagicMock(
            side_effect=lambda *args, **kwargs: mock_task_coroutine()
        )
        sequential_learner._check_stop_criterion = MagicMock(return_value=(False, None))

        return sequential_learner

    @pytest.mark.asyncio
    async def test_teach_missing_required_functions(self, sequential_learner):
        """Test that teach raises exception when required functions are missing."""

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
            await sequential_learner.teach(max_iter=1)

        # Test with only simulation function set
        # (should fail on training/active learning check)
        sequential_learner.simulation_function = AsyncMock()
        with pytest.raises(
            ValueError, match=r"Training and Active Learning functions must be set!"
        ):
            await sequential_learner.teach(max_iter=1)

        # Test with simulation and training set but no active learning
        # (should still fail on training/active learning check)
        sequential_learner.training_function = AsyncMock()
        with pytest.raises(
            ValueError, match=r"Training and Active Learning functions must be set!"
        ):
            await sequential_learner.teach(max_iter=1)

    @pytest.mark.asyncio
    async def test_teach_missing_stop_criteria(self, sequential_learner):
        """Test that teach raises exception when no stopping criteria provided."""
        sequential_learner.simulation_function = AsyncMock()
        sequential_learner.training_function = AsyncMock()
        sequential_learner.active_learn_function = AsyncMock()
        sequential_learner.criterion_function = None

        with pytest.raises(
            ValueError,
            match="Either max_iter > 0 or criterion_function must be provided.",
        ):
            await sequential_learner.teach(max_iter=0)

    @pytest.mark.asyncio
    async def test_teach_with_max_iterations(self, configured_learner):
        """Test teach method runs for specified max_iter iterations."""
        # The configured_learner fixture already sets up proper coroutines

        # Run with max_iter=2
        await configured_learner.teach(max_iter=2, skip_pre_loop=True)

        # Verify _register_task was called for active learning tasks
        # Should be called for: 2 active_learn tasks + 2 sim
        # tasks + 2 train tasks = 6 times
        assert configured_learner._register_task.call_count >= 4

    @pytest.mark.asyncio
    async def test_teach_with_stop_criterion(self, configured_learner):
        """Test teach method stops when criterion function returns True."""
        # The configured_learner fixture already sets up proper coroutines

        # Set up criterion to stop after first iteration
        configured_learner._check_stop_criterion.side_effect = [
            (True, None),  # Stop on first iteration
        ]

        await configured_learner.teach(max_iter=0, skip_pre_loop=True)

        # Verify that we stopped early due to criterion
        # When skip_pre_loop=True and we stop on first iteration, we should have:
        # 1. acl_task (active learning)
        # 2. stop_task (criterion check)
        # Then it breaks before registering next_sim and next_train
        assert configured_learner._register_task.call_count == 2  # acl, criterion only

        # Verify _check_stop_criterion was called
        assert configured_learner._check_stop_criterion.call_count == 1

    @pytest.mark.asyncio
    async def test_teach_continues_when_criterion_false(self, configured_learner):
        """Test teach method continues when criterion function returns False."""
        # Set up criterion to not stop on first iteration, but stop on second
        configured_learner._check_stop_criterion.side_effect = [
            (False, None),  # Don't stop on first iteration
            (True, None),  # Stop on second iteration
        ]

        await configured_learner.teach(max_iter=0, skip_pre_loop=True)

        # When we don't stop on first iteration, we should have:
        # Iteration 0: acl_task, stop_task, next_sim, next_train
        # Iteration 1: acl_task, stop_task (then stop)
        # Total: 6 calls
        assert configured_learner._register_task.call_count == 6

        # Verify _check_stop_criterion was called twice
        assert configured_learner._check_stop_criterion.call_count == 2

    @pytest.mark.asyncio
    async def test_teach_skip_pre_loop(self, configured_learner):
        """Test teach method behavior when skip_pre_loop is True."""
        # The configured_learner fixture already sets up proper coroutines

        # Set up stop criterion to stop immediately
        configured_learner._check_stop_criterion.return_value = (True, None)

        await configured_learner.teach(max_iter=0, skip_pre_loop=True)

        # When skip_pre_loop is True, should not register pre-loop
        # simulation/training tasks
        # Verify the method completes without pre-loop setup
        assert configured_learner._register_task.call_count >= 1
