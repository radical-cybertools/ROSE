"""Parallel active learner via YAML spec — per-learner shell executables.

Two learners race in parallel:
  - linear_regression: uses LinearRegression
  - ridge_regression:  uses Ridge(alpha=1.0)

Each learner runs its own shell scripts, writing to isolated files
(sim_a.pkl / model_a.pkl for learner 0, sim_b.pkl / model_b.pkl for learner 1).
The shared Python criterion averages MSE across both.

Run locally:
    python run_me.py
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec.builder import LearnerBuilder


async def main():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    builder = LearnerBuilder(Path(__file__).parent / "workflow.yaml", asyncflow)
    cfg = builder.config
    learner = builder.build()

    lc = builder.build_learner_configs()
    async for state in learner.start(
        max_iter=cfg.learner.max_iter,
        parallel_learners=len(lc),
        learner_configs=lc,
    ):
        label = cfg.learners[state.learner_id].label
        print(f"[{label}]  iter {state.iteration:>2d}  mse={state.metric_value:.4f}")

    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
