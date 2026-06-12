"""Unit tests for rose.spec.adapters — closure factories, no HPC needed."""
import asyncio

import pytest

from rose.spec.adapters import TaskAdapterFactory
from rose.spec.schema import RemoteConfig, TaskDef


def _remote() -> RemoteConfig:
    return RemoteConfig()


# ── Shell closure ─────────────────────────────────────────────────────────────

def test_shell_closure_returns_command():
    td = TaskDef(type="shell", command="echo hello")
    fn = TaskAdapterFactory.make_closure(td, _remote())
    result = asyncio.run(fn())
    assert result == "echo hello"



def test_shell_as_executable_true():
    td = TaskDef(type="shell", command="x")
    assert TaskAdapterFactory.as_executable(td) is True


# ── Python closure ────────────────────────────────────────────────────────────

def test_python_closure_calls_sync_function():
    td = TaskDef(type="python", function="os.path:join")
    fn = TaskAdapterFactory.make_closure(td, _remote())
    result = asyncio.run(fn("a", "b"))
    assert result == "a/b"


def test_python_closure_calls_async_function():
    async def _async_fn(x):
        return x * 2

    import types
    mod = types.ModuleType("_test_async_mod")
    mod.double = _async_fn
    import sys
    sys.modules["_test_async_mod"] = mod

    td = TaskDef(type="python", function="_test_async_mod:double")
    fn = TaskAdapterFactory.make_closure(td, _remote())
    result = asyncio.run(fn(21))
    assert result == 42

    del sys.modules["_test_async_mod"]


def test_python_as_executable_false():
    td = TaskDef(type="python", function="os:getcwd")
    assert TaskAdapterFactory.as_executable(td) is False


def test_python_closure_injects_remote_path(tmp_path):
    (tmp_path / "myutil.py").write_text("def add(a, b): return a + b\n")
    td = TaskDef(type="python", function="myutil:add")
    fn = TaskAdapterFactory.make_closure(td, RemoteConfig(pythonpath=[str(tmp_path)]))
    result = asyncio.run(fn(3, 4))
    assert result == 7


# ── Shell dispatch ────────────────────────────────────────────────────────────

def test_shell_dispatch_routes_by_learner_id():
    tds = [
        TaskDef(type="shell", command="cmd_0"),
        TaskDef(type="shell", command="cmd_1"),
    ]
    fn = TaskAdapterFactory.make_dispatch_closure("simulation", tds, _remote())
    assert asyncio.run(fn(learner_id=0)) == "cmd_0"
    assert asyncio.run(fn(learner_id=1)) == "cmd_1"


def test_shell_dispatch_defaults_to_zero():
    tds = [TaskDef(type="shell", command="cmd_default")]
    fn = TaskAdapterFactory.make_dispatch_closure("simulation", tds, _remote())
    assert asyncio.run(fn()) == "cmd_default"


# ── Python dispatch ───────────────────────────────────────────────────────────

def test_python_dispatch_routes_by_learner_id():
    tds = [
        TaskDef(type="python", function="os.path:basename"),
        TaskDef(type="python", function="os.path:dirname"),
    ]
    fn = TaskAdapterFactory.make_dispatch_closure("training", tds, _remote())
    assert asyncio.run(fn("/a/b/c", learner_id=0)) == "c"
    assert asyncio.run(fn("/a/b/c", learner_id=1)) == "/a/b"
