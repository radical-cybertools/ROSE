"""ClearML PnP Tracker — parallel ensemble UQ for material property prediction.

Science
-------
Ensemble-based UQ active learning for a materials science regression task:
predicting the formation energy of hypothetical crystal structures from
a set of structural descriptors (e.g. Coulomb matrix eigenvalues). Two
parallel learners (``"ensemble-A"`` and ``"ensemble-B"``) train on different
random seeds, building diverse GP models. Their per-iteration metrics are
reported as separate ClearML tasks inside one project, enabling direct
comparison of convergence across seeds.

UQ metric   : predictive entropy across the two-model ensemble
              — measures how much the models disagree on unseen structures
UQ threshold: stop when mean entropy < 0.02 (ensemble is sufficiently confident)
Stop criterion: MSE < 0.005 on a held-out test set

Scientific value:
  - Ensemble diversity (different seeds) reduces systematic error
  - Predictive entropy quantifies aleatoric + epistemic uncertainty
  - ClearML overlay plot shows whether both seeds converge at the same rate

ClearML captures
----------------
  ``on_start``      → hyperparams: criterion threshold, task names, seed info
  ``on_iteration``  → per-learner scalars: mse, mean_entropy, max_entropy,
                      n_labeled, n_pool (each as a separate ClearML series)
  ``on_stop``       → task tag "stop:criterion_met" or "stop:max_iter_reached"

Requirements
------------
    pip install rose[clearml] scikit-learn numpy

Usage
-----
    python run_me.py

    # View results:
    # Open ClearML web UI → project "ROSE-Materials-UQ"
    # Compare "mse" scalar curves between the two parallel learner tasks
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

from rose.integrations.clearml_tracker import ClearMLTracker
from rose.metrics import MEAN_SQUARED_ERROR_MSE, PREDICTIVE_ENTROPY
from rose.uq.uq_active_learner import ParallelUQLearner
from rose.uq.uq_learner import UQLearnerConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 15
MSE_THRESHOLD = 0.005
UQ_THRESHOLD = 0.02       # mean entropy nats — stop when ensemble agrees
UQ_QUERY_SIZE = 10        # top-10 most uncertain structures per iteration
N_INITIAL = 20
N_POOL = 400
N_SELECT_AL = 12
LEARNER_NAMES = ["ensemble-A", "ensemble-B"]
SEEDS = {"ensemble-A": 11, "ensemble-B": 37}

DATA_FILE = Path(tempfile.gettempdir()) / "materials_uq_{name}.pkl"


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
# ROSE task functions — each receives --learner_name to isolate state
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
    save_state(name, {"X_labeled": X_labeled, "y_labeled": y_labeled,
                      "X_pool": X_pool, "y_pool": y_pool, "model": None,
                      "seed": seed})
    return {"n_labeled": N_INITIAL, "n_pool": N_POOL}


async def training(*args, **kwargs) -> dict:
    """Fit GP model on current labeled set (seed-specific random init)."""
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    rng_state = data["seed"]
    kernel = RBF(length_scale=[0.3] * 5) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, normalize_y=True,
        random_state=rng_state,
    )
    gp.fit(data["X_labeled"], data["y_labeled"].ravel())
    lml = float(gp.log_marginal_likelihood_value_)
    train_mse = float(mean_squared_error(
        data["y_labeled"], gp.predict(data["X_labeled"])
    ))
    save_state(name, {**data, "model": gp})
    return {
        "train_mse": train_mse,
        "log_marginal_likelihood": lml,
        "n_labeled": len(data["X_labeled"]),
    }


async def prediction(*args, **kwargs) -> dict:
    """Generate predictions + uncertainties from this ensemble member."""
    name = kwargs.get("--learner_name", "default")
    model_name = kwargs.get("--model_name", "gp")
    data = load_state(name)
    gp = data["model"]
    y_pred, std = gp.predict(data["X_pool"], return_std=True)
    # Return dict keyed by model name — used by UQ aggregation
    return {model_name: {"y_pred": y_pred.tolist(), "std": std.tolist()}}


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
    save_state(name, {**data, "X_labeled": X_labeled, "y_labeled": y_labeled,
                      "X_pool": X_pool, "y_pool": y_pool})
    return {"n_labeled": len(X_labeled), "n_pool": len(X_pool),
            "mean_std": float(std.mean()), "max_std": float(std.max())}


async def check_accuracy(*args, **kwargs) -> float:
    """MSE on a fixed 200-point held-out test set — stop criterion."""
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    rng = np.random.default_rng(999)
    X_test = rng.uniform(0, 1, (200, 5))
    y_test = formation_energy(X_test)
    y_pred = data["model"].predict(X_test)
    return float(mean_squared_error(y_test, y_pred))


async def check_uq(*args, **kwargs) -> float:
    """Compute mean predictive entropy across the ensemble.

    Predictive entropy H = -sum_k p_k * log(p_k) is approximated here
    using the normalised GP predictive standard deviations as a proxy
    for the probability mass at each candidate point.
    """
    name = kwargs.get("--learner_name", "default")
    data = load_state(name)
    gp = data["model"]
    X_pool = data["X_pool"]

    if len(X_pool) == 0:
        return 0.0

    _, std = gp.predict(X_pool, return_std=True)
    # Approximate entropy: H(x) = 0.5 * log(2πe * σ²) for Gaussian
    entropy = 0.5 * np.log(2 * np.pi * np.e * (std**2 + 1e-12))
    return float(entropy.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = ParallelUQLearner(asyncflow)

    # ── Single line to enable full ClearML tracking ───────────────────────
    # Each yielded state (learner_id = "ensemble-A" / "ensemble-B") is logged
    # as a separate scalar series inside the same ClearML task, making
    # per-seed comparison trivial in the ClearML UI.
    learner.add_tracker(
        ClearMLTracker(
            project_name="ROSE-Materials-UQ",
            task_name="parallel-ensemble-gp",
        )
    )

    @learner.simulation_task(as_executable=False)
    async def sim(*args, **kwargs):
        return await simulation(*args, **kwargs)

    @learner.training_task(as_executable=False)
    async def train(*args, **kwargs):
        return await training(*args, **kwargs)

    @learner.prediction_task(as_executable=False)
    async def predict(*args, **kwargs):
        return await prediction(*args, **kwargs)

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

    @learner.uncertainty_quantification(
        uq_metric_name=PREDICTIVE_ENTROPY,
        threshold=UQ_THRESHOLD,
        query_size=UQ_QUERY_SIZE,
        operator="<",
        as_executable=False,
    )
    async def uq(*args, **kwargs):
        return await check_uq(*args, **kwargs)

    # Per-learner configs: each seed gets its own learner_name kwarg
    learner_configs = {
        name: UQLearnerConfig(
            simulation=dict(kwargs={"--learner_name": name}),
            training=dict(kwargs={"--learner_name": name}),
            prediction=dict(kwargs={"--learner_name": name}),
            active_learn=dict(kwargs={"--learner_name": name}),
            criterion=dict(kwargs={"--learner_name": name}),
            uncertainty=dict(kwargs={"--learner_name": name}),
        )
        for name in LEARNER_NAMES
    }

    print("=" * 60)
    print("ROSE + ClearMLTracker — Parallel Ensemble UQ")
    print("Learners:", LEARNER_NAMES)
    print("=" * 60)

    async for state in learner.start(
        learner_names=LEARNER_NAMES,
        model_names=["gp-ensemble"],
        num_predictions=1,
        max_iter=MAX_ITERATIONS,
        learner_configs=learner_configs,
    ):
        print(
            f"[{state.learner_id}  iter {state.iteration:3d}]  "
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
