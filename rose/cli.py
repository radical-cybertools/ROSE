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


def _run_remote(yaml_path: Path) -> None:
    from rose.remote import run_remote
    from rose.spec import load_spec

    spec = load_spec(yaml_path)
    if spec.config.remote.target is None:
        sys.exit(
            "--remote requires a 'remote.target' block in the spec "
            f"({yaml_path}) — see docs/user-guide/spec-api.md#remote-config"
        )
    asyncio.run(run_remote(spec))


def main():
    parser = argparse.ArgumentParser(prog="rose")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Execute a ROSE workflow YAML")
    run_p.add_argument("yaml", type=Path, help="Path to workflow YAML")
    run_p.add_argument(
        "--local", action="store_true", help="Run locally using rhapsody concurrent backend"
    )
    run_p.add_argument(
        "--remote",
        action="store_true",
        help="Run on a remote HPC target via the ORBIT broker "
        "(spec-driven; see 'remote.target' in the YAML)",
    )
    run_p.add_argument(
        "--backend",
        default="dragon_v3",
        help="Rhapsody backend name for --local (default: concurrent)",
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.local and args.remote:
            parser.error("--local and --remote are mutually exclusive")
        elif args.local:
            _run_local(args.yaml, args.backend)
        elif args.remote:
            _run_remote(args.yaml)
        else:
            parser.error("specify a mode: --local | --remote")
    else:
        parser.print_help()
        sys.exit(1)
