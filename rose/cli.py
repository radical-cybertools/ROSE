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
    from rose.spec.runner import resolve_client_callables, run_learner

    async def _main():
        spec = load_spec(yaml_path)
        cfg = spec.config
        resolve_client_callables(cfg)  # fail on a bad source/callback before the backend starts
        engine = await rhapsody.get_backend(backend)
        asyncflow = await WorkflowEngine.create(engine)
        await run_learner(cfg, asyncflow)

    asyncio.run(_main())


def _run_remote(yaml_path: Path) -> None:
    from rose.remote import run_remote
    from rose.spec import load_spec
    from rose.spec.runner import resolve_client_callables

    spec = load_spec(yaml_path)
    if spec.config.remote.target is None:
        sys.exit(
            "--remote requires a 'remote.target' block in the spec "
            f"({yaml_path}) — see docs/user-guide/spec-api.md#remote-config"
        )
    # stream sources and on_model_ready run here, on the client, next to the YAML file
    resolve_client_callables(spec.config)  # fail before bootstrapping the HPC endpoint
    asyncio.run(run_remote(spec))


def _run_setup(wait_timeout: float | None) -> None:
    from rose.remote import run_setup_wizard

    if wait_timeout is None:
        print(
            "Using the default wait timeout of 300s for HPC job checks. "
            "To change this, re-run with: rose setup --wait-timeout SECONDS"
        )
        wait_timeout = 300.0

    ok = run_setup_wizard(wait_timeout=wait_timeout)
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(prog="rose")
    sub = parser.add_subparsers(dest="command")

    setup_p = sub.add_parser(
        "setup", help="Interactive first-time setup wizard for `rose run --remote`"
    )
    setup_p.add_argument(
        "--wait-timeout",
        type=float,
        default=None,
        help="Seconds to wait for HPC test jobs during setup (default: 300)",
    )

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
        default="dragon",
        help="Rhapsody backend name for --local (default: dragon)",
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
    elif args.command == "setup":
        _run_setup(args.wait_timeout)
    else:
        parser.print_help()
        sys.exit(1)
