"""Run a spec-built learner: shared by ``rose run --local`` and ``WorkflowSpec.workflow``.

Backend creation stays with the caller; this module has no rhapsody import.
"""

from __future__ import annotations

import importlib
import inspect
import sys
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


def _client_path(cfg: WorkflowConfig) -> None:
    """Put the YAML file's directory on sys.path for the rest of the process.

    Client-side callables are imported from there, also lazily while the run is going.
    """
    source_dir = cfg._source_dir
    if cfg.stream is not None and source_dir is not None and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def resolve_client_callables(cfg: WorkflowConfig, keep_path: bool = True) -> None:
    """Import the client-side callables now, so a typo fails before any backend or HPC job starts.

    Args:
        keep_path: Keep the YAML dir on sys.path afterwards (for the run); False for a pure check.

    Raises:
        ValueError: Listing every callable that cannot be imported or is not callable.
    """
    before = list(sys.path)
    _client_path(cfg)
    failures = []
    try:
        for spec in client_specs(cfg):
            try:
                if not callable(_resolve(spec)):
                    failures.append(f"  {spec!r}: not callable")
            except Exception as exc:  # any import-time error of the user's module
                failures.append(f"  {spec!r}: {exc}")
    finally:
        if not keep_path:
            sys.path[:] = before
    if failures:
        raise ValueError(
            "stream callables are not importable on the client:\n" + "\n".join(failures)
        )


async def _build_stream(stream_def: StreamDef):
    from rose.stream import Stream

    sources = []
    try:
        for src in stream_def.sources:
            source = _resolve(src.function)(*src.args, **src.kwargs)
            if inspect.isawaitable(source):
                source = await source
            if not hasattr(source, "__aiter__"):
                close = getattr(source, "close", None)
                if callable(close):
                    close()  # e.g. a sync generator returned by mistake
                raise TypeError(f"stream source {src.function!r} did not return an async iterator")
            sources.append(source)
    except BaseException:
        for source in sources:  # close the sources built so far (sockets, files)
            try:
                if hasattr(source, "aclose"):
                    await source.aclose()
            except Exception as e:
                print(f"stream source cleanup failed: {e!r}")
        raise
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
    _client_path(cfg)
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
        try:
            if stream is not None:
                await stream.aclose()
        finally:
            await asyncflow.shutdown()
