"""UQ active learner via YAML spec — Python functions, run locally."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.spec import load_spec
from rose.spec.runner import run_learner


async def main():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    spec = load_spec(Path(__file__).parent / "workflow.yaml")
    await run_learner(spec.config, asyncflow)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))  # make tasks.py importable
    asyncio.run(main())
