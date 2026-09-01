"""Sequential active learner via YAML spec — Python functions.

Run locally:
    python run_me.py

On HPC via AMSC (bridge already running):
    export RADICAL_BRIDGE_URL="https://<bridge-host>:8000"
    # Then use service_utils.run(load_spec("workflow.yaml").workflow) instead.
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # make tasks.py importable

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec.builder import LearnerBuilder


async def main():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    builder = LearnerBuilder(Path(__file__).parent / "workflow.yaml", asyncflow)
    cfg = builder.config
    learner = builder.build()

    async for state in learner.start(max_iter=cfg.learner.max_iter):
        print(f"iter {state.iteration:>2d}  mse={state.metric_value:.4f}")

    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
