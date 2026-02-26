"""
Minimal Example: Active Learning with start() API and ProcessPoolExecutor

This example demonstrates:
1. Using ConcurrentExecutionBackend with ProcessPoolExecutor
2. The start() API for granular control over the AL loop
3. Return-based state: tasks return dicts, ROSE extracts state automatically
4. Dynamic configuration adjustment via set_next_config()

All tasks are Python functions (as_executable=False) running in a process pool.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import typeguard

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al import SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE

import logging
from radical.asyncflow.logging import init_default_logger

logger = logging.getLogger(__name__)

# =============================================================================
# File-based state (works across processes)
# =============================================================================
DATA_FILE = Path("/tmp/rose_al_data.pkl")


def target_function(X: np.ndarray) -> np.ndarray:
    """Target function to learn: sinusoidal with noise."""
    return np.sin(2 * np.pi * X) + 0.1 * np.random.randn(*X.shape)


def save_data(X_labeled, y_labeled, X_pool, y_pool, model=None):
    """Save data to file for cross-process access."""
    with open(DATA_FILE, "wb") as f:
        pickle.dump({
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "model": model,
        }, f)


def load_data():
    """Load data from file."""
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Task Functions (defined at module level for pickling)
# =============================================================================
@typeguard.typechecked
async def simulation(*args, n_initial: int = 10, n_pool: int = 100) -> dict:
    """Generate initial labeled data and unlabeled pool.

    Args:
        *args: Dependency results from previous iteration (if any)
        n_initial: Number of initial labeled samples
        n_pool: Size of the unlabeled pool
    """
    np.random.seed(42)

    # Initial labeled set
    X_labeled = np.random.uniform(0, 1, (n_initial, 1))
    y_labeled = target_function(X_labeled)

    # Unlabeled pool
    X_pool = np.random.uniform(0, 1, (n_pool, 1))
    y_pool = target_function(X_pool)

    # Save to file
    save_data(X_labeled, y_labeled, X_pool, y_pool)

    # Return state - ROSE extracts this automatically
    return {
        "labeled_count": n_initial,
        "unlabeled_count": n_pool,
    }

@typeguard.typechecked
async def training(*args, length_scale: float = 1.0) -> dict:
    """Train a Gaussian Process model.

    Args:
        *args: Dependency results (simulation result passed automatically)
        length_scale: GP kernel length scale
    """
    data = load_data()

    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    model.fit(data["X_labeled"], data["y_labeled"].ravel())

    # Save model
    save_data(
        data["X_labeled"], data["y_labeled"],
        data["X_pool"], data["y_pool"],
        model=model
    )

    return {"length_scale": length_scale,
            "n_samples": len(data["X_labeled"])}

@typeguard.typechecked
async def active_learn(*args, n_select: int = 5) -> dict:
    """Select samples using uncertainty sampling (GP variance).

    Args:
        *args: Dependency results (sim and train results passed automatically)
        n_select: Number of samples to select
    """
    data = load_data()
    model = data["model"]
    X_pool = data["X_pool"]
    y_pool = data["y_pool"]
    X_labeled = data["X_labeled"]
    y_labeled = data["y_labeled"]

    # Predict with uncertainty
    _, std = model.predict(X_pool, return_std=True)

    # Select most uncertain samples
    indices = np.argsort(std)[-n_select:]

    # Move to labeled set
    X_labeled = np.vstack([X_labeled, X_pool[indices]])
    y_labeled = np.vstack([y_labeled, y_pool[indices].reshape(-1, 1)])

    # Remove from pool
    X_pool = np.delete(X_pool, indices, axis=0)
    y_pool = np.delete(y_pool, indices, axis=0)

    # Save updated data
    save_data(X_labeled, y_labeled, X_pool, y_pool, model=model)

    # Return state - ROSE extracts this automatically
    return {
        "labeled_count": len(X_labeled),
        "unlabeled_count": len(X_pool),
        "mean_uncertainty": float(np.mean(std)),
        "max_uncertainty": float(np.max(std)),
    }


async def check_mse(*args) -> float:
    """Compute MSE on validation data.

    Args:
        *args: Dependency results (acl result passed automatically)
    """
    data = load_data()
    model = data["model"]

    X_val = np.linspace(0, 1, 50).reshape(-1, 1)
    y_val = target_function(X_val)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


# =============================================================================
# Main Example
# =============================================================================
async def main():
    init_default_logger(logging.DEBUG)
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    learner = SequentialActiveLearner(asyncflow)

    # Register tasks (functions defined at module level)
    learner.simulation_task(as_executable=False)(simulation)
    learner.training_task(as_executable=False)(training)
    learner.active_learn_task(as_executable=False)(active_learn)
    learner.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=0.01,
        as_executable=False,
    )(check_mse)

    # Run Active Learning with iterate() for granular control
    print("=" * 50)

    async for state in learner.start(max_iter=10):
        print(f"[Iteration {state.iteration}]")
        print(f"MSE: {state.metric_value:.4f} (target: {state.metric_threshold})")
        print(f"Labeled: {state.labeled_count}, Pool: {state.unlabeled_count}")
        print(f"Mean uncertainty: {state.mean_uncertainty:.4f}")

        # Access training task return values (now automatically extracted)
        print(f"Training: length_scale={state.length_scale}, n_samples={state.n_samples}")

        # Dynamic adjustment: increase samples when uncertainty is low
        if state.mean_uncertainty and state.mean_uncertainty < 0.15:
            learner.set_next_config(
                LearnerConfig(active_learn=TaskConfig(kwargs={"n_select": 10}))
            )
            print("Low uncertainty, selecting 10 samples next")

        # Custom early stopping (lower than ROSE's threshold of 0.01)
        if state.metric_value and state.metric_value < 0.001:
            print("MSE below 0.005, stopping early!")
            break

        if state.unlabeled_count < 5:
            print("Pool exhausted, stopping!")
            break

    # Summary
    print("\n" + "=" * 50)
    print(state.to_dict())

    await asyncflow.shutdown()

    # Cleanup
    if DATA_FILE.exists():
        DATA_FILE.unlink()


if __name__ == "__main__":
    asyncio.run(main())
