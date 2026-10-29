"""Run a spec-built learner: shared by ``rose run --local`` and ``WorkflowSpec.workflow``.

Backend creation stays with the caller; this module has no rhapsody import.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

from .builder import LearnerBuilder
from .schema import StreamDef, WorkflowConfig


def _resolve(spec: str) -> Any:
    mod_path, fn_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), fn_name)


def client_specs(cfg: WorkflowConfig) -> list[str]:
    """'module:callable' strings that run on the client: stream sources and on_model_ready."""
    if cfg.stream is None:
        return []
    specs = [src.function for src in cfg.stream.sources]
    if cfg.stream.on_model_ready is not None:
        specs.append(cfg.stream.on_model_ready.function)
    return list(dict.fromkeys(specs))


def resolve_client_callables(cfg: WorkflowConfig, base_dir: str | Path | None = None) -> None:
    """Import the client-side callables now, so a typo fails before any backend or HPC job starts.

    Args:
        base_dir: Directory temporarily put on ``sys.path`` (the YAML file's directory).

    Raises:
        ValueError: Listing every callable that cannot be imported.
    """
    added = None
    if base_dir is not None and str(base_dir) not in sys.path:
        added = str(base_dir)
        sys.path.insert(0, added)
    try:
        failures = []
        for spec in client_specs(cfg):
            try:
                _resolve(spec)
            except (ImportError, AttributeError) as exc:
                failures.append(f"  {spec!r}: {exc}")
        if failures:
            raise ValueError(
                "stream callables are not importable on the client:\n" + "\n".join(failures)
            )
    finally:
        if added is not None and added in sys.path:
            sys.path.remove(added)


async def _build_stream(stream_def: StreamDef):
    from rose.stream import Stream

    sources = []
    for src in stream_def.sources:
        source = _resolve(src.function)(*src.args, **src.kwargs)
        if inspect.isawaitable(source):
            source = await source
        if not hasattr(source, "__aiter__"):
            raise TypeError(f"stream source {src.function!r} did not return an async iterator")
        sources.append(source)
    return Stream(
        batch_size=stream_def.batch_size,
        max_wait=stream_def.max_wait,
        conflate=stream_def.conflate,
        sources=sources,
    )


def _start_kwargs(cfg: WorkflowConfig, builder: LearnerBuilder) -> dict[str, Any]:
    start_kwargs: dict[str, Any] = {"max_iter": cfg.learner.max_iter}
    if cfg.learner.type == "parallel_active_learner":
        lcs = builder.build_learner_configs()
        if lcs is not None:
            start_kwargs["parallel_learners"] = len(lcs)
            start_kwargs["learner_configs"] = lcs
        else:
            start_kwargs["parallel_learners"] = cfg.learner.parallel_learners
    else:
        # Sequential learners accept initial_config — the same ROSE-native mechanism used by the
        # parallel path, giving tasks access to parameters and iteration via kwargs.
        ic = builder.build_learner_config()
        if ic is not None:
            start_kwargs["initial_config"] = ic
    return start_kwargs


async def run_learner(cfg: WorkflowConfig, asyncflow) -> None:
    """Build the learner (and stream) from ``cfg`` and run it to completion on ``asyncflow``.

    Shuts down ``asyncflow`` when done; the stream is closed first.
    """
    stream = None
    try:
        builder = LearnerBuilder(cfg, asyncflow)
        learner = builder.build()
        start_kwargs = _start_kwargs(cfg, builder)
        if cfg.stream is not None:
            stream = await _build_stream(cfg.stream)
            start_kwargs["stream"] = stream
            if cfg.stream.on_model_ready is not None:
                learner.on_model_ready(_resolve(cfg.stream.on_model_ready.function))

        async for state in learner.start(**start_kwargs):
            ready = "  model_ready" if state.model_ready else ""
            print(f"[iter {state.iteration}]  metric={state.metric_value}{ready}", flush=True)
    finally:
        if stream is not None:
            await stream.aclose()
        await asyncflow.shutdown()
