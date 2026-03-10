"""Native ROSE FileTracker — HPC-safe append-only JSON Lines logging.

Science
-------
Gaussian Process surrogate for the Branin-Hoo benchmark function, a canonical
test problem in surrogate-based optimisation with two input variables and a
known ground truth. The learner uses GP predictive variance (uncertainty
sampling) to select the most informative points from the candidate pool.

Stop criterion : MSE < 0.01 on a fixed 400-point validation grid.

Tracker
-------
``HPC_FileTracker`` implements ``TrackerBase`` using append-only JSON Lines
output — one record per iteration. JSON Lines is the correct format for HPC:

  - Append-only: each write is atomic; no record is ever overwritten
  - Survives job preemption: all completed iterations are already on disk
  - Human-readable and importable with ``pandas.read_json(..., lines=True)``

``on_state_update`` captures the fitted GP log-marginal-likelihood (LML) each
time it is registered mid-iteration, before the full snapshot is built.

Requirements
------------
    pip install numpy scikit-learn

Usage
-----
    python run_me.py

    # Inspect results:
    python -c "import pandas; print(pandas.read_json('branin_run.jsonl', lines=True))"
"""

import asyncio
import json
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from rose import IterationState, PipelineManifest, TrackerBase
from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 20
MSE_THRESHOLD = 0.01
N_INITIAL = 15
N_POOL = 300
N_SELECT = 10
DATA_FILE = Path(tempfile.gettempdir()) / "branin_al_data.pkl"
JSONL_FILE = Path("branin_run.jsonl")


# ---------------------------------------------------------------------------
# Ground truth: Branin-Hoo function (rescaled to [0,1]²)
# ---------------------------------------------------------------------------
def branin(X: np.ndarray) -> np.ndarray:
    """Branin-Hoo benchmark: f(x1,x2) has 3 global minima at f≈0.398."""
    x1 = X[:, 0] * 15.0 - 5.0  # rescale [0,1] → [-5, 10]
    x2 = X[:, 1] * 15.0  # rescale [0,1] → [0, 15]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    # Normalise to [0,1] range for MSE interpretability
    return (y / 300.0).reshape(-1, 1)


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
# HPC_FileTracker — native TrackerBase, zero external dependencies
# ---------------------------------------------------------------------------
class HPC_FileTracker(TrackerBase):
    """Append-only JSON Lines tracker designed for HPC job preemption safety.

    Each call to ``_write`` is a single ``f.write()`` — atomic at the OS
    level on POSIX filesystems. If the job is preempted mid-run, all
    iterations written before the preemption are intact on disk.

    Post-processing::

        import pandas
        df = pandas.read_json("branin_run.jsonl", lines=True)
        iterations = df[df.event == "iteration"]
        iterations.plot(x="iteration", y="mse", logy=True)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on new run — remove if you want to resume and append
        self._path.write_text("")
        self._t0: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def on_start(self, manifest: PipelineManifest) -> None:
        self._t0 = time.time()
        self._write(
            {
                "event": "start",
                "ts": self._t0,
                "learner_type": manifest.learner_type,
                "tasks": list(manifest.tasks.keys()),
                "criterion": {
                    "metric": manifest.criterion.metric_name,
                    "threshold": manifest.criterion.threshold,
                    "operator": manifest.criterion.operator,
                }
                if manifest.criterion
                else None,
            }
        )
        print(f"[Tracker] Logging to {self._path.resolve()}")

    def on_iteration(self, state: IterationState) -> None:
        self._write(
            {
                "event": "iteration",
                "iteration": state.iteration,
                "elapsed_s": round(time.time() - self._t0, 3),
                # Core metric (MSE from stop criterion)
                "mse": state.metric_value,
                "should_stop": state.should_stop,
                # Task outputs auto-extracted by register_state() inside tasks
                "n_labeled": state.get("n_labeled"),
                "n_pool": state.get("n_pool"),
                "mean_std": state.get("mean_std"),
                "max_std": state.get("max_std"),
                "train_mse": state.get("train_mse"),
                # LML streamed mid-iteration via on_state_update
                "log_marginal_likelihood": state.get("log_marginal_likelihood"),
            }
        )

    def on_stop(self, final_state: IterationState | None, reason: str) -> None:
        self._write(
            {
                "event": "stop",
                "reason": reason,
                "elapsed_s": round(time.time() - self._t0, 3),
                "final_iteration": final_state.iteration if final_state else None,
                "final_mse": final_state.metric_value if final_state else None,
            }
        )
        print(f"[Tracker] Run complete — reason={reason!r}, file={self._path.resolve()}")

    def on_state_update(self, key: str, value: Any) -> None:
        """Stream mid-iteration updates (e.g. LML computed inside training task)."""
        if key == "log_marginal_likelihood":
            self._write(
                {
                    "event": "live",
                    "key": key,
                    "value": float(value),
                    "ts": time.time(),
                }
            )

    # ── Internal ───────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# ROSE task functions
# ---------------------------------------------------------------------------
async def simulation(*args, n_initial: int = N_INITIAL, n_pool: int = N_POOL) -> dict:
    """Sample initial labeled set and candidate pool from Branin input space.

    Idempotent: if the data file already exists (main-loop calls after the
    pre-loop initialisation), this is a no-op that returns current statistics
    without resetting the accumulated labeled set.
    """
    if DATA_FILE.exists():
        data = load_state()
        return {"n_labeled": len(data["X_labeled"]), "n_pool": len(data["X_pool"])}
    rng = np.random.default_rng(42)
    X_labeled = rng.uniform(0, 1, (n_initial, 2))
    y_labeled = branin(X_labeled)
    X_pool = rng.uniform(0, 1, (n_pool, 2))
    y_pool = branin(X_pool)
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


async def training(*args, length_scale: float = 0.3, noise_level: float = 0.01) -> dict:
    """Fit an ARD GP surrogate on the current labeled set."""
    data = load_state()
    kernel = RBF(length_scale=[length_scale] * 2) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=0
    )
    gp.fit(data["X_labeled"], data["y_labeled"].ravel())
    lml = float(gp.log_marginal_likelihood_value_)
    y_pred_train = gp.predict(data["X_labeled"])
    train_mse = float(mean_squared_error(data["y_labeled"], y_pred_train))
    save_state({**data, "model": gp})
    return {
        "train_mse": train_mse,
        "log_marginal_likelihood": lml,
        "n_labeled": len(data["X_labeled"]),
    }


async def active_learn(*args, n_select: int = N_SELECT) -> dict:
    """Uncertainty sampling: add the n_select most uncertain pool points to the labeled set."""
    data = load_state()
    gp = data["model"]
    X_pool, y_pool = data["X_pool"], data["y_pool"]
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
    """Evaluate surrogate MSE on a fixed 20×20 validation grid over [0,1]²."""
    data = load_state()
    gp = data["model"]
    grid = np.linspace(0, 1, 20)
    xx, yy = np.meshgrid(grid, grid)
    X_val = np.column_stack([xx.ravel(), yy.ravel()])
    y_val = branin(X_val)
    y_pred = gp.predict(X_val)
    return float(mean_squared_error(y_val, y_pred))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = SequentialActiveLearner(asyncflow)

    # ── Attach tracker — logs everything automatically from here on ──────
    learner.add_tracker(HPC_FileTracker(JSONL_FILE))

    @learner.simulation_task(as_executable=False)
    async def sim(*args, **kwargs):
        return await simulation(*args, **kwargs)

    @learner.training_task(as_executable=False)
    async def train(*args, **kwargs):
        result = await training(*args, **kwargs)
        # Register LML so on_state_update fires → tracker captures it live
        learner.register_state("log_marginal_likelihood", result["log_marginal_likelihood"])
        return result

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

    print("=" * 55)
    print("ROSE Native FileTracker — Branin-Hoo GP Surrogate")
    print("=" * 55)

    async for state in learner.start(max_iter=MAX_ITERATIONS):
        print(
            f"[iter {state.iteration:3d}]  "
            f"MSE={state.metric_value:.5f}  "
            f"labeled={state.get('n_labeled'):3d}  "
            f"pool={state.get('n_pool'):3d}  "
            f"LML={state.get('log_marginal_likelihood'):+.1f}"
        )

    await learner.shutdown()
    DATA_FILE.unlink(missing_ok=True)

    print()
    print("Replay run with:")
    print(
        f'  python -c "import pandas; '
        f"print(pandas.read_json('{JSONL_FILE}', lines=True)"
        f"[['iteration','mse','n_labeled']].dropna())\""
    )


if __name__ == "__main__":
    asyncio.run(main())
