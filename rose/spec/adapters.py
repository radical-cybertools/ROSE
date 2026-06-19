from __future__ import annotations

from collections.abc import Callable

from .schema import RemoteConfig, TaskDef


class TaskAdapterFactory:
    @staticmethod
    def make_closure(task_def: TaskDef, remote: RemoteConfig) -> Callable:
        td = dict(task_def.task_description or {})
        if task_def.type == "shell":
            return _make_shell_closure(task_def.command, td)
        return _make_python_closure(task_def.function, remote.pythonpath, td)

    @staticmethod
    def as_executable(task_def: TaskDef) -> bool:
        return task_def.type == "shell"

    @staticmethod
    def make_dispatch_closure(
        slot_name: str,
        task_defs: list[TaskDef],
        remote: RemoteConfig,
    ) -> Callable:
        """Dispatch closure for parallel learners: routes per-learner based on learner_id kwarg.

        task_description uses the first learner's value — asyncflow reads it once at registration.
        """
        td = dict(task_defs[0].task_description or {})
        if task_defs[0].type == "shell":
            return _make_shell_dispatch(
                {i: tdi.command for i, tdi in enumerate(task_defs)}, slot_name, td
            )
        return _make_python_dispatch(
            {i: tdi.function for i, tdi in enumerate(task_defs)},
            list(remote.pythonpath),
            slot_name,
            td,
        )


def _make_shell_closure(command: str, task_description: dict) -> Callable:
    _cmd = command
    _td = task_description

    async def _task(*args, task_description=_td, **kwargs) -> str:
        return _cmd.format_map(kwargs)

    _task.__name__ = "shell_task"
    return _task


def _make_python_closure(spec: str, remote_paths: list[str], task_description: dict) -> Callable:
    _spec = spec
    _paths = list(remote_paths)
    _td = task_description

    async def _task(*args, task_description=_td, **kwargs):
        import importlib as _il
        import inspect as _ins
        import sys as _sys

        for p in _paths:
            if p not in _sys.path:
                _sys.path.insert(0, p)
        mod_path, fn_name = _spec.rsplit(":", 1)
        fn = getattr(_il.import_module(mod_path), fn_name)
        result = fn(*args, **kwargs)
        return (await result) if _ins.iscoroutine(result) else result

    _task.__name__ = spec.split(":")[-1]
    return _task


def _make_shell_dispatch(cmds: dict[int, str], slot_name: str, task_description: dict) -> Callable:
    _cmds = dict(cmds)
    _td = task_description

    async def _dispatch(*args, task_description=_td, **kwargs) -> str:
        cmd = _cmds[kwargs.get("learner_id", 0)]
        return cmd.format_map(kwargs)

    _dispatch.__name__ = f"{slot_name}_dispatch"
    return _dispatch


def _make_python_dispatch(
    specs: dict[int, str], remote_paths: list[str], slot_name: str, task_description: dict
) -> Callable:
    _specs = dict(specs)
    _paths = list(remote_paths)
    _td = task_description

    async def _dispatch(*args, task_description=_td, **kwargs):
        import importlib as _il
        import inspect as _ins
        import sys as _sys

        lid = kwargs.pop("learner_id", 0)  # strip internal routing key before calling user fn
        spec = _specs[lid]
        for p in _paths:
            if p not in _sys.path:
                _sys.path.insert(0, p)
        mod_path, fn_name = spec.rsplit(":", 1)
        fn = getattr(_il.import_module(mod_path), fn_name)
        result = fn(*args, **kwargs)
        return (await result) if _ins.iscoroutine(result) else result

    _dispatch.__name__ = f"{slot_name}_dispatch"
    return _dispatch
