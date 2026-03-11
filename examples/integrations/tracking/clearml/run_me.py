"""ClearML Tracker — parallel ensemble active learning for material property prediction.

Science
-------
Ensemble-based active learning for a materials science regression task:
predicting the formation energy of hypothetical crystal structures from
a set of structural descriptors (5-D Coulomb matrix eigenvalues). Two
parallel learners (``"ensemble-A"`` and ``"ensemble-B"``) train on different
random seeds, building diverse GP models. Their per-iteration metrics are
logged as separate ClearML scalar series inside the same task, enabling direct
convergence comparison in the ClearML UI.

Stop criterion: MSE < 0.01 on a held-out 200-point test set.

Scientific value:
  - Ensemble diversity (different seeds) reduces systematic error
  - ClearML overlay plot shows whether both seeds converge at the same rate
  - Parallel execution: both learners run concurrently, not sequentially

ClearML captures
----------------
  ``on_start``      → hyperparams: learner_type, criterion threshold/operator,
                      task names, as_executable flag
  ``on_iteration``  → per-learner scalars: mse, train_mse, n_labeled, n_pool
                      (each as a separate ClearML series per learner_id)
  ``on_stop``       → task tag "stop:criterion_met" or "stop:max_iter_reached"

Requirements
------------
    pip install rose[clearml] scikit-learn numpy

Usage
-----
    python run_me.py

    # View results:
    # Open ClearML web UI → project "ROSE-Materials-UQ"
    # Scalars tab: overlay mse curves for ensemble-A vs ensemble-B
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

from rose.al.active_learner import ParallelActiveLearner
from rose.integrations.clearml_tracker import ClearMLTracker
from rose.learner import LearnerConfig, TaskConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 20
MSE_THRESHOLD = 0.01
N_INITIAL = 20
N_POOL = 400
N_SELECT_AL = 12
LEARNER_NAMES = ["ensemble-A", "ensemble-B"]
SEEDS = {"ensemble-A": 11, "ensemble-B": 37}

DATA_FILE = Path(tempfile.gettempdir()) / "materials_al_{name}.pkl"


# ---------------------------------------------------------------------------
# Synthetic materials dataset: 5-D Coulomb matrix feature space
# Ground truth: formation energy proxy  f(x) = sum_i sin(2πxᵢ) · xᵢ
# ---------------------------------------------------------------------------
def formation_energy(X: np.ndarray) -> np.ndarray:
    """Proxy formation energy function — non-linear 5-D benchmark."""
    val = np.sum(np.sin(2 * np.pi * X) * X, axis=1)
    return ((val - val.min()) / (val.max() - val.min() + 1e-9)).reshape(-1, 1)


def save_state(name: str, data: dict) -> None:
    path = Path(str(DATA_FILE).format(name=name))
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_state(name: str) -> dict:
    path = Path(str(DATA_FILE).format(name=name))
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# ROSE task functions — each receives --learner_name to isolate per-seed state
# ---------------------------------------------------------------------------
async def simulation(*args, **kwargs) -> dict:
    """Generate initial labeled structures and unlabeled candidate pool.

    Idempotent: main-loop calls after pre-loop initialisation return
    current statistics without resetting the accumulated labeled set.
    """
    name = kwargs.get("--learner_name", "default")
    path = Path(str(DATA_FILE).format(name=name))
    if path.exists():
        data = load_state(name)
        return {"n_labeled": len(data["X_labeled"]), "n_pool": len(data["X_pool"])}
    seed = SEEDS.get(name, 0)
    rng = np.random.default_rng(seed)
    X_labeled = rng.uniform(0, 1, (N_INITIAL, 5))
    y_labeled = formation_energy(X_labeled)
    X_pool = rng.uniform(0, 1, (N_POOL, 5))
    y_pool = formation_energy(X_pool)
    save_state(
        name,
        {
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "model": None,
            "seed": seed,
        },
    )
    return {"n_labeled": N_INITIAL, "n_pool": N_POOL}


async def training(*args, **kwargs) -> dict:
    """Fit GP model on current labeled set (seed-specific random init)."""
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    kernel = RBF(length_scale=[0.3] * 5) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=data["seed"],
    )
    gp.fit(data["X_labeled"], data["y_labeled"].ravel())
    lml = float(gp.log_marginal_likelihood_value_)
    train_mse = float(mean_squared_error(data["y_labeled"], gp.predict(data["X_labeled"])))
    save_state(name, {**data, "model": gp})
    return {
        "train_mse": train_mse,
        "log_marginal_likelihood": lml,
        "n_labeled": len(data["X_labeled"]),
    }


async def active_learn(*args, **kwargs) -> dict:
    """Select most uncertain pool candidates and add them to the labeled set."""
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    gp, X_pool, y_pool = data["model"], data["X_pool"], data["y_pool"]
    X_labeled, y_labeled = data["X_labeled"], data["y_labeled"]

    if len(X_pool) == 0:
        return {"n_labeled": len(X_labeled), "n_pool": 0}

    _, std = gp.predict(X_pool, return_std=True)
    n_sel = min(N_SELECT_AL, len(X_pool))
    idx = np.argsort(std)[-n_sel:]
    X_labeled = np.vstack([X_labeled, X_pool[idx]])
    y_labeled = np.vstack([y_labeled, y_pool[idx]])
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx, axis=0)
    save_state(
        name,
        {
            **data,
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "X_pool": X_pool,
            "y_pool": y_pool,
        },
    )
    return {
        "n_labeled": len(X_labeled),
        "n_pool": len(X_pool),
        "mean_std": float(std.mean()),
    }


async def check_accuracy(*args, **kwargs) -> float:
    """MSE on a fixed 200-point held-out test set — stop criterion."""
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    rng = np.random.default_rng(999)
    X_test = rng.uniform(0, 1, (200, 5))
    y_test = formation_energy(X_test)
    y_pred = data["model"].predict(X_test)
    return float(mean_squared_error(y_test, y_pred))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # Always start fresh — clear stale per-learner data files
    for name in LEARNER_NAMES:
        Path(str(DATA_FILE).format(name=name)).unlink(missing_ok=True)

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = ParallelActiveLearner(asyncflow)

    # Register all tasks first — add_tracker() fires on_start(manifest) immediately,
    # so tasks must be registered before the tracker is attached.
    @learner.simulation_task(as_executable=False)
    async def sim(*args, **kwargs):
        return await simulation(*args, **kwargs)

    @learner.training_task(as_executable=False)
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
        return await check_accuracy(*args, **kwargs)

    # ── Attach tracker after all tasks are registered ─────────────────────
    # on_start(manifest) fires here — the manifest is now complete.
    learner.add_tracker(
        ClearMLTracker(
            project_name="ROSE-Materials-UQ",
            task_name="parallel-ensemble-gp",
            learner_names=LEARNER_NAMES,
        )
    )

    # Per-learner configs: inject --learner_name so each task accesses its own state file
    learner_configs = [
        LearnerConfig(
            simulation=TaskConfig(kwargs={"--learner_name": name}),
            training=TaskConfig(kwargs={"--learner_name": name}),
            active_learn=TaskConfig(kwargs={"--learner_name": name}),
            criterion=TaskConfig(kwargs={"--learner_name": name}),
        )
        for name in LEARNER_NAMES
    ]

    print("=" * 60)
    print("ROSE + ClearMLTracker — Parallel Ensemble Active Learning")
    print("Learners:", LEARNER_NAMES)
    print("=" * 60)

    async for state in learner.start(
        parallel_learners=len(LEARNER_NAMES),
        max_iter=MAX_ITERATIONS,
        learner_configs=learner_configs,
    ):
        label = (
            LEARNER_NAMES[state.learner_id]
            if isinstance(state.learner_id, int)
            else state.learner_id
        )
        print(
            f"[{label:12s}  iter {state.iteration:3d}]  "
            f"MSE={state.metric_value:.5f}  "
            f"labeled={state.get('n_labeled')}  "
            f"pool={state.get('n_pool')}"
        )

    await learner.shutdown()

    # Clean up per-learner data files
    for name in LEARNER_NAMES:
        Path(str(DATA_FILE).format(name=name)).unlink(missing_ok=True)

    print()
    print("View results:")
    print("  Open ClearML web UI → project 'ROSE-Materials-UQ'")
    print("  → Scalars tab: compare mse curves for ensemble-A vs ensemble-B")
    print("  → Tags: stop:criterion_met or stop:max_iter_reached per learner")


if __name__ == "__main__":
    asyncio.run(main())
