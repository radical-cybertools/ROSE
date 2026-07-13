#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path


def _run_local(yaml_path: Path, backend: str) -> None:
    sys.path.insert(0, str(yaml_path.resolve().parent))
    import rhapsody
    from radical.asyncflow import WorkflowEngine

    from rose.spec import load_spec
    from rose.spec.builder import LearnerBuilder

    async def _main():
        spec = load_spec(yaml_path)
        cfg = spec.config
        engine = await rhapsody.get_backend(backend)
        asyncflow = await WorkflowEngine.create(engine)
        builder = LearnerBuilder(cfg, asyncflow)
        learner = builder.build()

        start_kwargs: dict = {"max_iter": cfg.learner.max_iter}
        if cfg.learner.type == "parallel_active_learner":
            lcs = builder.build_learner_configs()
            if lcs is not None:
                start_kwargs["parallel_learners"] = len(lcs)
                start_kwargs["learner_configs"] = lcs
            else:
                start_kwargs["parallel_learners"] = cfg.learner.parallel_learners
        else:
            ic = builder.build_learner_config()
            if ic is not None:
                start_kwargs["initial_config"] = ic

        try:
            async for state in learner.start(**start_kwargs):
                print(f"[iter {state.iteration}]  metric={state.metric_value}", flush=True)
        finally:
            await asyncflow.shutdown()

    asyncio.run(_main())


def main():
    parser = argparse.ArgumentParser(prog="rose")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Execute a ROSE workflow YAML")
    run_p.add_argument("yaml", type=Path, help="Path to workflow YAML")
    run_p.add_argument(
        "--local", action="store_true", help="Run locally using rhapsody concurrent backend"
    )
    run_p.add_argument(
        "--backend",
        default="concurrent",
        help="Rhapsody backend name for --local (default: concurrent)",
    )

    args = parser.parse_args()

    if args.command == "run":
        if not args.local:
            parser.error("specify a mode: --local  (--orbit coming soon)")
        _run_local(args.yaml, args.backend)
    else:
        parser.print_help()
        sys.exit(1)
