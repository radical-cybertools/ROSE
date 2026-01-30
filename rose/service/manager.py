import asyncio
import json
import os
import shutil
import importlib.util
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from concurrent.futures import ProcessPoolExecutor

from rose.al.active_learner import SequentialActiveLearner
from rose.learner import LearnerConfig, TaskConfig
from .models import Workflow, WorkflowState

logger = logging.getLogger(__name__)

class WorkflowLoader:
    """Helper to load a Learner from a YAML definition."""
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """Load YAML file (mocking yaml load with json for now or basic parsing if yaml lib not avail?
           User environment might have PyYAML. Assuming yaml is available or using json for simplicity if needed.
           The user request says 'workflow.yaml', so we should try to support YAML.
           If PyYAML is not installed, we might fallback or error. 
           Standard python doesn't have yaml.
        """
        # For this implementation, I will assume PyYAML is available as it's common in this stack,
        # or I will implement a very simple parser if restricted.
        # Given "ROSE" context, PyYAML is likely a dependency. 
        # But to be safe and depend only on stdlib as requested ("standard libraries only" was for IPC, but let's stick to it),
        # I will check if yaml module exists, otherwise parsing simple key-value or use JSON.
        # However, the user explicitly said "workflow.yaml". 
        # I will try to import yaml.
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            # Fallback: simpler parsing or expect JSON content in .yaml (not ideal)
            logger.warning("PyYAML not found, trying JSON parsing for workflow file")
            with open(path, "r") as f:
                return json.load(f)

    @staticmethod
    def _import_function(path_str: str) -> Callable:
        """Import a function from a module path string 'package.module.func'."""
        try:
            module_name, func_name = path_str.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import function '{path_str}': {e}")

    @staticmethod
    def _create_script_task_factory(script_path: str) -> Callable:
        """Create a task function that returns the script path + arguments.
        
        Args:
           script_path: The base command or script path.
        """
        async def task_func(*args, **kwargs):
            # Extract string arguments to append to the command.
            # Skip Task objects (dependencies).
            cmd_parts = [script_path]
            for arg in args:
                if isinstance(arg, str):
                    cmd_parts.append(arg)
            
            for k, v in kwargs.items():
                if isinstance(v, bool):
                     if v: cmd_parts.append(f"--{k}")
                else:
                     cmd_parts.append(f"--{k} {v}")
            
            print(f"Task command: {' '.join(cmd_parts)}")
            return " ".join(cmd_parts)
        return task_func

    @classmethod
    def create_learner(cls, wf_id: str, workflow_def: Dict[str, Any], asyncflow: WorkflowEngine) -> SequentialActiveLearner:
        """Create and configure a SequentialActiveLearner."""

        # 1. Create Learner
        learner = SequentialActiveLearner(asyncflow)
        learner.learner_id = wf_id # Using wf_id (str) might need adaptation if learner_id expects int in some places? 
        # rose/active_learner.py: learner_id (Optional[int]). 
        # I should probably hash the wf_id or just set it if it accepts Any?
        # The type hint says Optional[int]. Let's ignore type hint for a moment or hash it.
        learner.learner_id = hash(wf_id) 

        components = workflow_def.get("components", {})
        
        # 2. Register Components
        # Expecting structure:
        # components:
        #   simulation:
        #     type: function | script
        #     path: ...
        #     config: ...
        
        for name in ["simulation", "training", "active_learn", "criterion"]:
            comp_def = components.get(name)
            if not comp_def:
                continue
                
            ctype = comp_def.get("type", "script") # Default to script?
            cpath = comp_def.get("path")
            
            task_func = None
            as_executable = True
            
            if ctype == "function":
                task_func = cls._import_function(cpath)
                as_executable = False
            else:
                # Script
                task_func = cls._create_script_task_factory(cpath)
                as_executable = True
                
            # Register using the appropriate decorator
            if name == "simulation":
                print(f"Registering simulation task: {task_func}")
                learner.simulation_task(as_executable=as_executable)(task_func)
            elif name == "training":
                learner.training_task(as_executable=as_executable)(task_func)
            elif name == "active_learn":
                learner.active_learn_task(as_executable=as_executable)(task_func)
            elif name == "criterion":
                # Special handling for criterion
                threshold = comp_def.get("threshold", 0.0)
                metric = comp_def.get("metric", "CUSTOM")
                learner.as_stop_criterion(metric_name=metric, threshold=threshold, as_executable=as_executable)(task_func)

        # 3. Build initial LearnerConfig from component defs
        # This ensures that args/kwargs specified in YAML are used
        l_config = LearnerConfig()
        for name in ["simulation", "training", "active_learn", "criterion"]:
            comp_def = components.get(name)
            if comp_def and "config" in comp_def:
                c_config = comp_def["config"]
                t_config = TaskConfig(
                    args=tuple(c_config.get("args", ())),
                    kwargs=c_config.get("kwargs", {})
                )
                setattr(l_config, name, t_config)

        return learner, l_config

class ServiceManager:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.service_root = Path.home() / ".rose" / "services" / str(job_id)
        self.requests_dir = self.service_root / "requests"
        self.registry_file = self.service_root / "registry.json"
        
        self.workflows: Dict[str, Workflow] = {}
        self.engine: Optional[WorkflowEngine] = None
        self._learner_tasks: List[asyncio.Task] = []
        self._shutdown = False

    async def initialize(self):
        """Setup directories and backend."""
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AsyncFlow with Local Backend (or Slurm if needed, but 'service runs inside job')
        # If running inside a job, we usually use Resource='local.localhost' or similar to spawn tasks 
        # on the allocated resources.
        engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
        self.engine = await WorkflowEngine.create(engine)
        logger.info(f"Service initialized at {self.service_root}")

    async def _process_requests(self):
        """Pick up json files from requests_dir."""
        if not self.requests_dir.exists():
            return

        # Sort by mtime to process in order?
        req_files = sorted(self.requests_dir.glob("*.json"), key=os.path.getmtime)
        
        for req_file in req_files:
            try:
                with open(req_file, "r") as f:
                    req = json.load(f)
                
                action = req.get("action")
                payload = req.get("payload", {})
                
                if action == "submit":
                    await self._handle_submit(req.get("id"), payload)
                elif action == "cancel":
                    await self._handle_cancel(payload)
                
                # Remove request file after processing
                req_file.unlink()
                
            except Exception as e:
                logger.error(f"Error processing request {req_file}: {e}")
                # Move to failed_requests? Or just delete?
                # For now, delete to avoid loop
                try:
                    req_file.unlink()
                except:
                    pass

    async def _handle_submit(self, req_id: str, payload: Dict[str, Any]):
        wf_file = payload.get("workflow_file")
        if not wf_file:
            logger.error("No workflow file in submit payload")
            return

        # Use request ID as part of wf_id or generate new?
        # User goal: "assigned a unique workflow identifier (wf_id)"
        wf_id = f"wf.{req_id[:8]}" 
        
        wf = Workflow(wf_id=wf_id, state=WorkflowState.INITIALIZING, workflow_file=wf_file)
        self.workflows[wf_id] = wf
        self._update_registry()

        try:
            wf_def = WorkflowLoader.load_yaml(wf_file)
            learner, initial_l_config = WorkflowLoader.create_learner(wf_id, wf_def, self.engine)
            wf.learner_instance = learner
            
            # Merge with top-level config if needed (e.g. if we want to override via top-level)
            # For now, initial_l_config from components is primary.
            
            # Start the learner loop as a background task
            task = asyncio.create_task(self._run_learner(wf, wf_def.get("config", {}), initial_l_config))
            self._learner_tasks.append(task)
            
        except Exception as e:
            logger.error(f"Failed to submit workflow {wf_id}: {e}")
            wf.state = WorkflowState.FAILED
            wf.error = str(e)
            self._update_registry()

    async def _handle_cancel(self, payload: Dict[str, Any]):
        wf_id = payload.get("wf_id")
        wf = self.workflows.get(wf_id)
        if wf and wf.state in [WorkflowState.RUNNING, WorkflowState.INITIALIZING, WorkflowState.SUBMITTED]:
            logger.info(f"Canceling workflow {wf_id}")
            if wf.learner_instance:
                wf.learner_instance.stop() # Cooperative cancel
            wf.state = WorkflowState.CANCELED
            self._update_registry()

    async def _run_learner(self, wf: Workflow, config: Dict[str, Any], initial_l_config: Optional[LearnerConfig] = None):
        """Driver loop for a single workflow."""
        wf.state = WorkflowState.RUNNING
        wf.start_time = asyncio.get_event_loop().time()
        self._update_registry()
        
        try:
            max_iter = config.get("max_iterations", 0)
            
            async for state in wf.learner_instance.start(max_iter=max_iter, initial_config=initial_l_config):
                # Update stats
                wf.stats = state.to_dict()
                # If we want to support granular status updates, update registry here
                # (maybe throttled)
                self._update_registry()
            
            wf.state = WorkflowState.COMPLETED
            
        except Exception as e:
            logger.error(f"Workflow {wf.wf_id} failed: {e}")
            wf.state = WorkflowState.FAILED
            wf.error = str(e)
        finally:
            wf.end_time = asyncio.get_event_loop().time()
            self._update_registry()

    def _update_registry(self):
        """Dump registry to json."""
        data = {wf_id: wf.to_dict() for wf_id, wf in self.workflows.items()}
        tmp_file = self.registry_file.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.registry_file)

    async def run(self):
        """Main Service Loop."""
        await self.initialize()
        logger.info("Service Manager Running")
        
        while not self._shutdown:
            await self._process_requests()
            await asyncio.sleep(1) # Polling interval

    async def shutdown(self):
        self._shutdown = True
        logger.info("Service Shutting Down")
        
        # 1. Stop all learners
        for wf in self.workflows.values():
            if wf.learner_instance:
                wf.learner_instance.stop()
        
        # 2. Cancel and wait for learner tasks
        if self._learner_tasks:
            for task in self._learner_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._learner_tasks, return_exceptions=True)
            self._learner_tasks.clear()

        # 3. Shutdown engine
        if self.engine:
            await self.engine.shutdown()
            self.engine = None
