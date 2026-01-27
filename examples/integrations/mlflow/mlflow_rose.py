"""
MLflow + ROSE Integration Example: Complementary Relationship

This example demonstrates how ROSE and MLflow work together:
- ROSE: Orchestrates the active learning workflow (execution engine)
- MLflow: Tracks experiments, logs metrics, and manages model artifacts (observability)

ROSE handles WHAT runs and WHEN (workflow orchestration)
MLflow handles WHAT happened and HOW WELL (experiment tracking)

Requirements:
    pip install mlflow scikit-learn numpy

Usage:
    python mlflow_rose.py

    # View results in MLflow UI:
    mlflow ui --port 5000
    # Then open http://localhost:5000
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import tempfile
from datetime import datetime

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MLflow for experiment tracking
import mlflow
from mlflow.models import infer_signature

# ROSE for workflow orchestration
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine
from rose.al import SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig
from rose.metrics import MEAN_SQUARED_ERROR_MSE


# =============================================================================
# Configuration
# =============================================================================
EXPERIMENT_NAME = "ROSE_Active_Learning"
DATA_FILE = Path(tempfile.gettempdir()) / "rose_mlflow_data.pkl"
MAX_ITERATIONS = 15
MSE_THRESHOLD = 0.002
N_INITIAL_SAMPLES = 10
N_POOL_SAMPLES = 200
N_SELECT_PER_ITERATION = 5


# =============================================================================
# Target Function (Ground Truth)
# =============================================================================
def target_function(X: np.ndarray) -> np.ndarray:
    """Complex target function: combination of sinusoids with noise.

    This simulates a real-world scenario where we have an expensive
    simulation that we want to approximate with a surrogate model.
    """
    return (
        np.sin(2 * np.pi * X) +
        0.5 * np.cos(4 * np.pi * X) +
        0.1 * np.random.randn(*X.shape)
    )


# =============================================================================
# Data Persistence (Cross-Process Communication)
# =============================================================================
def save_data(X_labeled, y_labeled, X_pool, y_pool, model=None, metadata=None):
    """Save data and model state to file."""
    with open(DATA_FILE, "wb") as f:
        pickle.dump({
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "model": model,
            "metadata": metadata or {},
        }, f)


def load_data():
    """Load data and model state from file."""
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


# =============================================================================
# ROSE Task Functions
# =============================================================================
async def simulation(*args, n_initial: int = N_INITIAL_SAMPLES,
                     n_pool: int = N_POOL_SAMPLES) -> dict:
    """Generate initial labeled data and unlabeled pool.

    In real applications, this would run expensive HPC simulations.
    ROSE orchestrates when and where these simulations execute.
    """
    np.random.seed(42)

    # Initial labeled set (expensive to obtain)
    X_labeled = np.random.uniform(0, 1, (n_initial, 1))
    y_labeled = target_function(X_labeled)

    # Unlabeled pool (candidates for labeling)
    X_pool = np.random.uniform(0, 1, (n_pool, 1))
    y_pool = target_function(X_pool)  # Hidden labels (oracle)

    save_data(X_labeled, y_labeled, X_pool, y_pool)

    return {
        "labeled_count": n_initial,
        "unlabeled_count": n_pool,
        "simulation_complete": True,
    }


async def training(*args, length_scale: float = 0.5,
                   noise_level: float = 0.1) -> dict:
    """Train a Gaussian Process surrogate model.

    ROSE ensures this runs after simulation completes.
    """
    data = load_data()

    # Build and train GP model
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    model.fit(data["X_labeled"], data["y_labeled"].ravel())

    # Compute training metrics
    y_train_pred = model.predict(data["X_labeled"])
    train_mse = mean_squared_error(data["y_labeled"], y_train_pred)

    # Extract learned hyperparameters
    learned_params = model.kernel_.get_params()

    save_data(
        data["X_labeled"], data["y_labeled"],
        data["X_pool"], data["y_pool"],
        model=model,
        metadata={"train_mse": train_mse, "kernel_params": str(learned_params)}
    )

    return {
        "train_mse": float(train_mse),
        "length_scale": length_scale,
        "noise_level": noise_level,
        "n_training_samples": len(data["X_labeled"]),
    }


async def active_learn(*args, n_select: int = N_SELECT_PER_ITERATION,
                       strategy: str = "uncertainty") -> dict:
    """Select informative samples using active learning.

    ROSE manages the iteration loop and task dependencies.
    """
    data = load_data()
    model = data["model"]
    X_pool = data["X_pool"]
    y_pool = data["y_pool"]
    X_labeled = data["X_labeled"]
    y_labeled = data["y_labeled"]

    if len(X_pool) == 0:
        return {
            "labeled_count": len(X_labeled),
            "unlabeled_count": 0,
            "mean_uncertainty": 0.0,
            "max_uncertainty": 0.0,
            "samples_selected": 0,
            "pool_exhausted": True,
        }

    # Predict with uncertainty quantification
    y_pred, std = model.predict(X_pool, return_std=True)

    # Selection strategy
    if strategy == "uncertainty":
        # Select most uncertain samples
        scores = std
    elif strategy == "random":
        scores = np.random.rand(len(X_pool))
    else:
        scores = std

    # Select top samples
    n_select = min(n_select, len(X_pool))
    indices = np.argsort(scores)[-n_select:]

    # Compute selection statistics
    selected_uncertainties = std[indices]

    # Move selected samples to labeled set
    X_labeled = np.vstack([X_labeled, X_pool[indices]])
    y_labeled = np.vstack([y_labeled, y_pool[indices].reshape(-1, 1)])

    # Remove from pool
    X_pool = np.delete(X_pool, indices, axis=0)
    y_pool = np.delete(y_pool, indices, axis=0)

    save_data(X_labeled, y_labeled, X_pool, y_pool, model=model)

    return {
        "labeled_count": len(X_labeled),
        "unlabeled_count": len(X_pool),
        "mean_uncertainty": float(np.mean(std)),
        "max_uncertainty": float(np.max(std)),
        "min_uncertainty": float(np.min(std)),
        "selected_mean_uncertainty": float(np.mean(selected_uncertainties)),
        "samples_selected": n_select,
        "selection_strategy": strategy,
        "pool_exhausted": len(X_pool) == 0,
    }


async def check_mse(*args) -> float:
    """Evaluate model on held-out validation data.

    Returns the metric value that ROSE uses for stopping criterion.
    """
    data = load_data()
    model = data["model"]

    # Dense validation grid
    X_val = np.linspace(0, 1, 100).reshape(-1, 1)
    y_val = target_function(X_val)
    y_pred = model.predict(X_val)

    return float(mean_squared_error(y_val, y_pred))


# =============================================================================
# MLflow Integration Layer
# =============================================================================
class MLflowROSETracker:
    """Integrates MLflow tracking with ROSE active learning workflow.

    This class demonstrates the complementary relationship:
    - ROSE decides what to run and manages execution
    - MLflow records what happened and stores artifacts
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.run = None
        self.iteration_metrics = []

    def start_experiment(self, config: dict):
        """Initialize MLflow experiment and run."""
        mlflow.set_experiment(self.experiment_name)

        self.run = mlflow.start_run(
            run_name=f"rose_al_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Log configuration parameters
        mlflow.log_params({
            "max_iterations": config.get("max_iterations", MAX_ITERATIONS),
            "mse_threshold": config.get("mse_threshold", MSE_THRESHOLD),
            "n_initial_samples": config.get("n_initial", N_INITIAL_SAMPLES),
            "n_pool_samples": config.get("n_pool", N_POOL_SAMPLES),
            "n_select_per_iteration": config.get("n_select", N_SELECT_PER_ITERATION),
            "orchestrator": "ROSE",
            "learner_type": "SequentialActiveLearner",
        })

        # Tag the run
        mlflow.set_tags({
            "framework": "ROSE+MLflow",
            "task_type": "active_learning",
            "model_type": "GaussianProcessRegressor",
        })

        print(f"MLflow Run ID: {self.run.info.run_id}")
        print(f"MLflow Experiment: {self.experiment_name}")

    def log_iteration(self, state):
        """Log metrics from a ROSE iteration to MLflow.

        Args:
            state: IterationState from ROSE learner
        """
        iteration = state.iteration

        # Core metrics
        metrics = {
            "mse": state.metric_value,
            "labeled_count": state.labeled_count,
            "unlabeled_count": state.unlabeled_count,
        }

        # Uncertainty metrics (if available)
        if state.mean_uncertainty is not None:
            metrics["mean_uncertainty"] = state.mean_uncertainty
        if state.max_uncertainty is not None:
            metrics["max_uncertainty"] = state.max_uncertainty
        if state.min_uncertainty is not None:
            metrics["min_uncertainty"] = state.min_uncertainty

        # Training metrics
        if state.train_mse is not None:
            metrics["train_mse"] = state.train_mse

        # Log all metrics with step
        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value, step=iteration)

        # Store for final summary
        self.iteration_metrics.append({
            "iteration": iteration,
            **metrics
        })

    def log_model(self, model, X_sample, y_sample):
        """Log the trained model to MLflow model registry."""
        # Infer signature from sample data
        signature = infer_signature(X_sample, model.predict(X_sample))

        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="surrogate_model",
            signature=signature,
            registered_model_name=f"{self.experiment_name}_GP_Model",
        )

    def log_final_evaluation(self, model):
        """Log comprehensive final evaluation metrics."""
        # Generate evaluation data
        X_eval = np.linspace(0, 1, 200).reshape(-1, 1)
        y_true = target_function(X_eval)
        y_pred, y_std = model.predict(X_eval, return_std=True)

        # Compute various metrics
        final_metrics = {
            "final_mse": mean_squared_error(y_true, y_pred),
            "final_mae": mean_absolute_error(y_true, y_pred),
            "final_r2": r2_score(y_true, y_pred),
            "final_mean_uncertainty": float(np.mean(y_std)),
            "total_iterations": len(self.iteration_metrics),
        }

        mlflow.log_metrics(final_metrics)

        # Log learning curve as artifact
        self._log_learning_curve()

        return final_metrics

    def _log_learning_curve(self):
        """Create and log learning curve visualization."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            iterations = [m["iteration"] for m in self.iteration_metrics]
            mse_values = [m.get("mse", 0) for m in self.iteration_metrics]
            labeled_counts = [m.get("labeled_count", 0) for m in self.iteration_metrics]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # MSE over iterations
            ax1.plot(iterations, mse_values, 'b-o', linewidth=2, markersize=6)
            ax1.axhline(y=MSE_THRESHOLD, color='r', linestyle='--',
                       label=f'Threshold ({MSE_THRESHOLD})')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('MSE')
            ax1.set_title('Active Learning: MSE vs Iteration')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # MSE vs labeled samples
            ax2.plot(labeled_counts, mse_values, 'g-s', linewidth=2, markersize=6)
            ax2.axhline(y=MSE_THRESHOLD, color='r', linestyle='--',
                       label=f'Threshold ({MSE_THRESHOLD})')
            ax2.set_xlabel('Number of Labeled Samples')
            ax2.set_ylabel('MSE')
            ax2.set_title('Active Learning: MSE vs Sample Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

            plt.tight_layout()

            # Save and log
            curve_path = Path(tempfile.gettempdir()) / "learning_curve.png"
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()

            mlflow.log_artifact(str(curve_path), artifact_path="plots")
            curve_path.unlink()

        except ImportError:
            print("matplotlib not available, skipping learning curve plot")

    def end_experiment(self, success: bool = True):
        """Finalize MLflow run."""
        if self.run:
            mlflow.set_tag("status", "success" if success else "failed")
            mlflow.end_run()
            print(f"MLflow run completed: {self.run.info.run_id}")


# =============================================================================
# Main: ROSE + MLflow Integration
# =============================================================================
async def main():
    """Run active learning with ROSE orchestration and MLflow tracking."""

    print("=" * 60)
    print("ROSE + MLflow Integration Example")
    print("=" * 60)
    print("\nROSE: Orchestrates the active learning workflow")
    print("MLflow: Tracks experiments, metrics, and models")
    print("=" * 60)

    # Initialize MLflow tracker
    tracker = MLflowROSETracker(EXPERIMENT_NAME)
    tracker.start_experiment({
        "max_iterations": MAX_ITERATIONS,
        "mse_threshold": MSE_THRESHOLD,
        "n_initial": N_INITIAL_SAMPLES,
        "n_pool": N_POOL_SAMPLES,
        "n_select": N_SELECT_PER_ITERATION,
    })

    try:
        # Initialize ROSE workflow engine
        engine = await ConcurrentExecutionBackend(ProcessPoolExecutor(max_workers=4))
        asyncflow = await WorkflowEngine.create(engine)

        # Create ROSE active learner
        learner = SequentialActiveLearner(asyncflow)

        # Register task functions with ROSE
        learner.simulation_task(as_executable=False)(simulation)
        learner.training_task(as_executable=False)(training)
        learner.active_learn_task(as_executable=False)(active_learn)
        learner.as_stop_criterion(
            metric_name=MEAN_SQUARED_ERROR_MSE,
            threshold=MSE_THRESHOLD,
            as_executable=False,
        )(check_mse)

        print("\n[ROSE] Starting active learning loop...")
        print("-" * 60)

        # Main active learning loop - ROSE orchestrates, MLflow records
        final_state = None
        async for state in learner.start(max_iter=MAX_ITERATIONS):
            # ROSE yields control at each iteration with current state
            print(f"\n[Iteration {state.iteration}]")
            print(f"  MSE: {state.metric_value:.6f} (threshold: {state.metric_threshold})")
            print(f"  Labeled: {state.labeled_count}, Pool: {state.unlabeled_count}")

            if state.mean_uncertainty:
                print(f"  Uncertainty - mean: {state.mean_uncertainty:.4f}, "
                      f"max: {state.max_uncertainty:.4f}")

            # MLflow logs the iteration metrics
            tracker.log_iteration(state)

            # Dynamic configuration based on state (ROSE feature)
            if state.mean_uncertainty and state.mean_uncertainty < 0.01:
                # Increase batch size when uncertainty is low
                learner.set_next_config(
                    LearnerConfig(active_learn=TaskConfig(kwargs={"n_select": 10}))
                )
                print("  [Config] Low uncertainty detected, increasing batch size")

            # Check for pool exhaustion
            if state.unlabeled_count < N_SELECT_PER_ITERATION:
                print("  [Warning] Pool nearly exhausted")

            final_state = state

            # Custom early stopping (in addition to ROSE's criterion)
            if state.metric_value and state.metric_value < MSE_THRESHOLD / 2:
                print(f"  [Early Stop] MSE well below threshold")
                break

        print("\n" + "-" * 60)
        print("[ROSE] Active learning completed")

        # Log final model to MLflow
        if final_state:
            data = load_data()
            if data.get("model"):
                print("\n[MLflow] Logging final model...")
                tracker.log_model(
                    data["model"],
                    data["X_labeled"][:10],
                    data["y_labeled"][:10]
                )

                print("[MLflow] Computing final evaluation metrics...")
                final_metrics = tracker.log_final_evaluation(data["model"])

                print("\n" + "=" * 60)
                print("Final Results")
                print("=" * 60)
                print(f"  Total iterations: {final_metrics['total_iterations']}")
                print(f"  Final MSE: {final_metrics['final_mse']:.6f}")
                print(f"  Final MAE: {final_metrics['final_mae']:.6f}")
                print(f"  Final R2:  {final_metrics['final_r2']:.4f}")
                print(f"  Final labeled samples: {final_state.labeled_count}")

        # Shutdown ROSE
        await asyncflow.shutdown()
        tracker.end_experiment(success=True)

    except Exception as e:
        print(f"\n[Error] {e}")
        tracker.end_experiment(success=False)
        raise

    finally:
        # Cleanup temporary data file
        if DATA_FILE.exists():
            DATA_FILE.unlink()

    print("\n" + "=" * 60)
    print("To view results, run:")
    print("  mlflow ui --port 5000")
    print("Then open http://192.168.0.172:5000")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
