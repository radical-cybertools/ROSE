import asyncio
import importlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

from rose.al.active_learner import ParallelActiveLearner, SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig

from .client import ServiceClient
from .models import Workflow, WorkflowState

logger = logging.getLogger(__name__)


class WorkflowLoader:
    """Helper to load and run a Learner from a YAML workflow definition."""

    @staticmethod
    def load_yaml(path: str) -> dict[str, Any]:
        """Parse a workflow YAML file and return its contents as a dict."""
        try:
            import yaml

            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not found, falling back to JSON parsing")
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    @staticmethod
    def _import_function(path_str: str) -> Callable:
        """Import a callable from a dotted module path (e.g. 'pkg.module.func')."""
        try:
            module_name, func_name = path_str.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import function '{path_str}': {e}") from e

    @staticmethod
    def _create_script_task_factory(script_path: str) -> Callable:
        """Return an async task function that builds a shell command from the script path.

        The returned function collects string positional arguments and keyword arguments,
        assembles them into a command list, and returns the joined string for execution
        by the workflow engine (``as_executable=True``).
        """

        async def task_func(*args, **kwargs):
            cmd_parts = [script_path]
            for arg in args:
                if isinstance(arg, str):
                    cmd_parts.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, bool):
                    if v:
                        cmd_parts.append(f"--{k}")
                else:
                    cmd_parts.extend([f"--{k}", str(v)])
            return " ".join(cmd_parts)

        return task_func

    @classmethod
    def create_learner(
        cls, wf_id: str, workflow_def: dict[str, Any], asyncflow: WorkflowEngine
    ) -> tuple:
        """Instantiate and configure a Learner from a workflow definition dict.

        Returns:
            (learner, initial_config) tuple.
        """
        learner_def = workflow_def.get("learner", {})
        l_type = learner_def.get("type", "SequentialActiveLearner")
        l_path = learner_def.get("path")

        if l_path:
            try:
                module_name, class_name = l_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                learner_cls = getattr(module, class_name)
            except (ValueError, ImportError, AttributeError) as e:
                raise ImportError(f"Could not import learner class '{l_path}': {e}") from e
        elif l_type == "SequentialActiveLearner":
            learner_cls = SequentialActiveLearner
        elif l_type == "ParallelActiveLearner":
            learner_cls = ParallelActiveLearner
        else:
            raise ValueError(f"Unknown learner type '{l_type}' and no path provided.")

        learner = learner_cls(asyncflow)
        learner.learner_id = wf_id

        components = workflow_def.get("components", {})

        for name in ["simulation", "training", "active_learn", "criterion"]:
            comp_def = components.get(name)
            if not comp_def:
                continue

            ctype = comp_def.get("type", "script")
            cpath = comp_def.get("path")
            as_executable = ctype != "function"

            if ctype == "function":
                task_func = cls._import_function(cpath)
            else:
                task_func = cls._create_script_task_factory(cpath)

            if name == "simulation":
                logger.info(f"Registering simulation task for workflow {wf_id}")
                learner.simulation_task(as_executable=as_executable)(task_func)
            elif name == "training":
                logger.info(f"Registering training task for workflow {wf_id}")
                learner.training_task(as_executable=as_executable)(task_func)
            elif name == "active_learn":
                logger.info(f"Registering active_learn task for workflow {wf_id}")
                learner.active_learn_task(as_executable=as_executable)(task_func)
            elif name == "criterion":
                logger.info(f"Registering criterion task for workflow {wf_id}")
                threshold = comp_def.get("threshold", 0.0)
                metric = comp_def.get("metric", "CUSTOM")
                learner.as_stop_criterion(
                    metric_name=metric, threshold=threshold, as_executable=as_executable
                )(task_func)

        l_config = LearnerConfig()
        for name in ["simulation", "training", "active_learn", "criterion"]:
            comp_def = components.get(name)
            if comp_def and "config" in comp_def:
                c_config = comp_def["config"]
                t_config = TaskConfig(
                    args=tuple(c_config.get("args", ())), kwargs=c_config.get("kwargs", {})
                )
                setattr(l_config, name, t_config)

        return learner, l_config

    @staticmethod
    async def run_learner(
        learner,
        wf_def: dict[str, Any],
        initial_config: LearnerConfig | None,
        on_iteration,
    ) -> None:
        """Drive a learner's iteration loop, calling ``on_iteration(state)`` per step.

        Handles both SequentialActiveLearner and ParallelActiveLearner.
        ``on_iteration`` may be a regular function or a coroutine.
        """
        config = wf_def.get("config", {})
        learner_cfg = wf_def.get("learner", {})
        max_iter = config.get("max_iterations", learner_cfg.get("max_iterations", 10))

        if isinstance(learner, ParallelActiveLearner):
            parallel = config.get("parallel_learners", learner_cfg.get("parallel_learners", 2))
            configs = [initial_config] * parallel if initial_config else None
            async for state in learner.start(
                parallel_learners=parallel, max_iter=max_iter, learner_configs=configs
            ):
                result = on_iteration(state)
                if asyncio.iscoroutine(result):
                    await result
        else:
            async for state in learner.start(max_iter=max_iter, initial_config=initial_config):
                result = on_iteration(state)
                if asyncio.iscoroutine(result):
                    await result


class ServiceManager:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.service_root = Path.home() / ".rose" / "services" / str(job_id)
        self.requests_dir = self.service_root / "requests"
        self.registry_file = self.service_root / "registry.json"

        self.workflows: dict[str, Workflow] = {}
        self.engine: WorkflowEngine | None = None
        self._learner_tasks: dict[str, asyncio.Task] = {}
        self._shutdown = False

    async def initialize(self):
        """Setup directories and workflow engine."""
        self.requests_dir.mkdir(parents=True, exist_ok=True)

        backend = LocalExecutionBackend()
        self.engine = await WorkflowEngine.create(backend)
        logger.info(f"Service initialized at {self.service_root}")

    async def _process_requests(self):
        """Pick up JSON request files from requests_dir and dispatch them."""
        if not self.requests_dir.exists():
            return

        req_files = sorted(self.requests_dir.glob("*.json"), key=os.path.getmtime)

        for req_file in req_files:
            try:
                with open(req_file, encoding="utf-8") as f:
                    req = json.load(f)

                action = req.get("action")
                payload = req.get("payload", {})

                if action == "submit":
                    await self._handle_submit(req.get("id"), payload)
                elif action == "cancel":
                    await self._handle_cancel(payload)
                elif action == "shutdown":
                    logger.info("Shutdown request received via IPC")
                    self._shutdown = True

                req_file.unlink()

            except Exception:
                logger.exception(f"Error processing request {req_file}")
                try:
                    req_file.unlink()
                except Exception:
                    pass

    async def _handle_submit(self, req_id: str, payload: dict[str, Any]):
        wf_file = payload.get("workflow_file")
        if not wf_file:
            logger.error("No workflow file in submit payload")
            return

        wf_id = ServiceClient.get_wf_id(req_id)
        wf = Workflow(wf_id=wf_id, state=WorkflowState.INITIALIZING, workflow_file=wf_file)
        self.workflows[wf_id] = wf
        self._update_registry()

        try:
            wf_def = WorkflowLoader.load_yaml(wf_file)
            learner, initial_l_config = WorkflowLoader.create_learner(wf_id, wf_def, self.engine)
            wf.learner_instance = learner

            task = asyncio.create_task(self._run_learner(wf, wf_def, initial_l_config))
            self._learner_tasks[wf_id] = task

        except Exception as e:
            logger.exception(f"Failed to submit workflow {wf_id}")
            wf.state = WorkflowState.FAILED
            wf.error = str(e)
            self._update_registry()

    async def _handle_cancel(self, payload: dict[str, Any]):
        wf_id = payload.get("wf_id")
        wf = self.workflows.get(wf_id)
        if wf and wf.state in (
            WorkflowState.RUNNING,
            WorkflowState.INITIALIZING,
            WorkflowState.SUBMITTED,
        ):
            logger.info(f"Canceling workflow {wf_id}")
            if wf.learner_instance:
                wf.learner_instance.stop()
            task = self._learner_tasks.get(wf_id)
            if task and not task.done():
                task.cancel()
            wf.state = WorkflowState.CANCELED
            self._update_registry()

    async def _run_learner(
        self,
        wf: Workflow,
        wf_def: dict[str, Any],
        initial_l_config: LearnerConfig | None = None,
    ):
        """Driver loop for a single workflow."""
        wf.state = WorkflowState.RUNNING
        wf.start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting workflow {wf.wf_id} ({wf.workflow_file})")
        self._update_registry()

        try:

            def on_iteration(state):
                wf.stats = state.to_dict() if hasattr(state, "to_dict") else {"result": str(state)}
                logger.info(
                    f"Workflow {wf.wf_id} - learner {getattr(state, 'learner_id', '?')},"
                    f" iteration {getattr(state, 'iteration', '?')}"
                    f" (metric={getattr(state, 'metric_value', '?')})"
                )
                self._update_registry()

            await WorkflowLoader.run_learner(
                wf.learner_instance, wf_def, initial_l_config, on_iteration
            )
            wf.state = WorkflowState.COMPLETED
            logger.info(f"Workflow {wf.wf_id} completed successfully")

        except asyncio.CancelledError:
            wf.state = WorkflowState.CANCELED
            logger.info(f"Workflow {wf.wf_id} canceled")

        except Exception as e:
            wf.state = WorkflowState.FAILED
            wf.error = str(e)
            logger.exception(f"Workflow {wf.wf_id} failed")

        finally:
            wf.end_time = asyncio.get_event_loop().time()
            self._learner_tasks.pop(wf.wf_id, None)
            self._update_registry()

    def _update_registry(self):
        """Atomically write workflow registry to disk."""
        data = {wf_id: wf.to_dict() for wf_id, wf in self.workflows.items()}
        tmp_file = self.registry_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.registry_file)

    async def run(self):
        """Main service loop."""
        try:
            await self.initialize()
            logger.info("Service Manager Running")

            while not self._shutdown:
                await self._process_requests()
                await asyncio.sleep(0.1)
        finally:
            await self.shutdown()

    async def shutdown(self):
        self._shutdown = True
        logger.info("Service Shutting Down...")

        if self.workflows:
            logger.info(f"Stopping {len(self.workflows)} workflows")
            for wf in self.workflows.values():
                if wf.learner_instance:
                    wf.learner_instance.stop()

        if self._learner_tasks:
            logger.info(f"Canceling {len(self._learner_tasks)} learner tasks")
            lerner_tasks = list(self._learner_tasks.values())
            for lerner_task in lerner_tasks:
                if not lerner_task.done():
                    lerner_task.cancel()
            await asyncio.gather(*lerner_tasks, return_exceptions=True)
            self._learner_tasks.clear()
            logger.info("All learner tasks stopped")

        if self.engine:
            logger.info("Shutting down workflow engine")
            await self.engine.shutdown()
            self.engine = None
            logger.info("Workflow engine shut down")

        logger.info("Service shutdown complete")
