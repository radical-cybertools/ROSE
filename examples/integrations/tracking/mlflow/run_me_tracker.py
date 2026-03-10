"""MLflow OnP Tracker — adaptive GP surrogate with kernel schedule logging.

Science
-------
Gaussian Process surrogate for the 2-D Rosenbrock function — a non-convex
benchmark commonly used to validate surrogate optimisation algorithms. The
key scientific feature here is an *adaptive kernel schedule*: the GP kernel
length-scale is tightened over iterations as the surrogate becomes more
accurate, mimicking the practice of curriculum-style surrogate refinement.

  Iteration  0–9  : RBF(length_scale=0.5) — broad, explores the landscape
  Iteration 10–19 : RBF(length_scale=0.2) — focused, refines local detail
  Iteration 20+   : RBF(length_scale=0.08) — tight, captures fine structure

Stop criterion : MSE < 1e-6 on a fixed 400-point validation grid.

MLflow captures
---------------
  ``on_start``      → run params: learner_type, criterion threshold/operator,
                      task names, as_executable flag, and any keys declared in
                      ``log_params`` at decoration time (e.g. kernel, num_gpus)
  ``on_iteration``  → mse (step metric), n_labeled, n_pool, mean_std,
                      train_mse, log_marginal_likelihood per iteration
  ``on_stop``       → tag stop_reason, final_iteration

Difference from the manual ``mlflow_rose.py`` example
------------------------------------------------------
The old example wires MLflow manually inside the ``async for`` loop.
This example uses ``learner.add_tracker(MLflowTracker(...))`` — a single
line before ``start()`` — and the learner calls the tracker automatically.
No MLflow code appears inside the loop.

Requirements
------------
    pip install rose[mlflow] scikit-learn numpy

Usage
-----
    python run_me_tracker.py

    # View results:
    mlflow ui --port 5000
    # Open http://localhost:5000  → experiment "ROSE-Rosenbrock-Surrogate"
"""

import asyncio
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from rose.al.active_learner import SequentialActiveLearner
from rose.integrations.mlflow_tracker import MLflowTracker
from rose.learner import LearnerConfig, TaskConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 35
MSE_THRESHOLD = 1e-6
N_INITIAL = 20
N_POOL = 800
N_SELECT = 15
DATA_FILE = Path(tempfile.gettempdir()) / "rosenbrock_al_data.pkl"

# Kernel schedule: iteration threshold → (length_scale, noise_level)
KERNEL_SCHEDULE: dict[int, tuple[float, float]] = {
    0: (0.50, 0.05),  # broad kernel — exploration phase
    10: (0.20, 0.01),  # medium kernel — transition phase
    20: (0.08, 0.002),  # tight kernel  — refinement phase
}


# ---------------------------------------------------------------------------
# Ground truth: 2-D Rosenbrock  f(x,y) = (1-x)² + 100(y-x²)²
# ---------------------------------------------------------------------------
def rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock function rescaled to [0,1]² input, output in [0,1]."""
    x = X[:, 0] * 4.0 - 2.0  # [0,1] → [-2, 2]
    y = X[:, 1] * 4.0 - 2.0  # [0,1] → [-2, 2]
    z = (1 - x) ** 2 + 100 * (y - x**2) ** 2
    return (z / 3600.0).reshape(-1, 1)  # normalise to ≈[0,1]


# ---------------------------------------------------------------------------
# Shared state helpers
# ---------------------------------------------------------------------------
def save_state(data: dict) -> None:
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)


def load_state() -> dict:
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# ROSE task functions
# ---------------------------------------------------------------------------
async def simulation(*args, n_initial: int = N_INITIAL, n_pool: int = N_POOL) -> dict:
    """Sample initial labeled set and unlabeled candidate pool.

    Idempotent: main-loop calls after the pre-loop initialisation return
    current statistics without resetting the accumulated labeled set.
    """
    if DATA_FILE.exists():
        data = load_state()
        return {"n_labeled": len(data["X_labeled"]), "n_pool": len(data["X_pool"])}
    rng = np.random.default_rng(7)
    X_labeled = rng.uniform(0, 1, (n_initial, 2))
    y_labeled = rosenbrock(X_labeled)
    X_pool = rng.uniform(0, 1, (n_pool, 2))
    y_pool = rosenbrock(X_pool)
    save_state(
        {
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "model": None,
        }
    )
    return {"n_labeled": n_initial, "n_pool": n_pool}


async def training(
    *args,
    length_scale: float = 0.5,
    noise_level: float = 0.05,
) -> dict:
    """Fit GP surrogate; kernel hyperparameters come from the iteration config."""
    data = load_state()
    kernel = RBF(length_scale=[length_scale] * 2) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=0
    )
    gp.fit(data["X_labeled"], data["y_labeled"].ravel())
    lml = float(gp.log_marginal_likelihood_value_)
    train_mse = float(mean_squared_error(data["y_labeled"], gp.predict(data["X_labeled"])))
    save_state({**data, "model": gp})
    return {
        "train_mse": train_mse,
        "log_marginal_likelihood": lml,
        "n_labeled": len(data["X_labeled"]),
        "length_scale_used": length_scale,
    }


async def active_learn(*args, n_select: int = N_SELECT) -> dict:
    """Uncertainty sampling: label the most uncertain pool candidates."""
    data = load_state()
    gp, X_pool, y_pool = data["model"], data["X_pool"], data["y_pool"]
    X_labeled, y_labeled = data["X_labeled"], data["y_labeled"]

    if len(X_pool) == 0:
        return {"n_labeled": len(X_labeled), "n_pool": 0, "mean_std": 0.0, "max_std": 0.0}

    _, std = gp.predict(X_pool, return_std=True)
    n_sel = min(n_select, len(X_pool))
    idx = np.argsort(std)[-n_sel:]

    X_labeled = np.vstack([X_labeled, X_pool[idx]])
    y_labeled = np.vstack([y_labeled, y_pool[idx]])
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx, axis=0)
    save_state(
        {**data, "X_labeled": X_labeled, "y_labeled": y_labeled, "X_pool": X_pool, "y_pool": y_pool}
    )
    return {
        "n_labeled": len(X_labeled),
        "n_pool": len(X_pool),
        "mean_std": float(std.mean()),
        "max_std": float(std.max()),
    }


async def check_mse(*args) -> float:
    """MSE on a fixed 20×20 validation grid — the stop criterion metric."""
    data = load_state()
    grid = np.linspace(0, 1, 20)
    xx, yy = np.meshgrid(grid, grid)
    X_val = np.column_stack([xx.ravel(), yy.ravel()])
    y_pred = data["model"].predict(X_val)
    return float(mean_squared_error(rosenbrock(X_val), y_pred))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    DATA_FILE.unlink(missing_ok=True)  # always start fresh

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = SequentialActiveLearner(asyncflow)

    # Register all tasks first — add_tracker() fires on_start(manifest) immediately,
    # so tasks must be registered before the tracker is attached.
    @learner.simulation_task(as_executable=False)
    async def sim(*args, **kwargs):
        return await simulation(*args, **kwargs)

    @learner.training_task(
        as_executable=False,
        log_params={"kernel": "RBF+WhiteKernel", "kernel_schedule": "adaptive"},
    )
    async def train(*args, **kwargs):
        return await training(*args, **kwargs)

    @learner.active_learn_task(as_executable=False)
    async def active(*args, **kwargs):
        return await active_learn(*args, **kwargs)

    @learner.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=MSE_THRESHOLD,
        operator="<",
        as_executable=False,
    )
    async def criterion(*args, **kwargs):
        return await check_mse(*args, **kwargs)

    # ── Attach tracker after all tasks are registered ─────────────────────
    # on_start(manifest) fires here — the manifest is now complete.
    learner.add_tracker(
        MLflowTracker(
            experiment_name="ROSE-Rosenbrock-Surrogate",
            run_name="gp-adaptive-kernel",
        )
    )

    print("=" * 58)
    print("ROSE + MLflowTracker — Rosenbrock GP Surrogate")
    print("=" * 58)

    # Build LearnerConfig objects from the schedule
    configs = {
        it: LearnerConfig(
            training=TaskConfig(
                kwargs={
                    "length_scale": ls,
                    "noise_level": nl,
                }
            )
        )
        for it, (ls, nl) in KERNEL_SCHEDULE.items()
    }

    async for state in learner.start(max_iter=MAX_ITERATIONS):
        print(
            f"[iter {state.iteration:3d}]  "
            f"MSE={state.metric_value:.5f}  "
            f"labeled={state.get('n_labeled'):3d}  "
            f"ls={state.get('length_scale_used'):.2f}  "
            f"LML={state.get('log_marginal_likelihood'):+.1f}"
        )

        # Inject next kernel config if a schedule boundary is reached.
        # MLflow sees current_config change automatically in the next
        # on_iteration() call — no manual log_params() call needed.
        next_iter = state.iteration + 1
        if next_iter in configs:
            learner.set_next_config(configs[next_iter])
            ls, _ = KERNEL_SCHEDULE[next_iter]
            print(f"kernel schedule: length_scale={ls}")

    await learner.shutdown()
    DATA_FILE.unlink(missing_ok=True)

    print()
    print("View results:")
    print("mlflow ui --port 5000")
    print("# Experiment: ROSE-Rosenbrock-Surrogate")
    print("# Metrics: MEAN_SQUARED_ERROR_MSE, n_labeled, LML")
    print("# Tags: stop_reason, final_iteration")


if __name__ == "__main__":
    asyncio.run(main())
