# flake8: noqa

from functools import wraps
from typing import Callable, Dict
from concurrent.futures import Future

import radical.utils as ru
import radical.pilot as rp

import typeguard
from data import InputFile, OutputFile

class Task(rp.TaskDescription):
    """
    Represents a task description by extending the `TaskDescription` class from `rp` (an external module).

    This class is primarily used to define and manage the details of a task, inheriting properties and methods 
    from the `rp.TaskDescription` base class. Additional arguments and keyword arguments can be passed to further 
    configure the task, which are then forwarded to the base class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the parent class constructor, if needed.
    **kwargs : dict
        Keyword arguments to configure the task. Passed directly to the `TaskDescription` initializer.

    Methods
    -------
    None. This class relies on inherited methods from `rp.TaskDescription`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs)


class ResourceEngine:
    """
    The ResourceEngine class is responsible for managing computing resources and creating
    sessions for executing tasks. It interfaces with a resource management framework to
    initialize sessions, manage task execution, and submit resources required for the workflow.

    Attributes:
        session (rp.Session): A session instance used to manage and track task execution,
            uniquely identified by a generated ID. This session serves as the primary context for
            all task and resource management within the workflow.

        task_manager (rp.TaskManager): Manages the lifecycle of tasks, handling their submission,
            tracking, and completion within the session.

        pilot_manager (rp.PilotManager): Manages computing resources, known as "pilots," which
            are dynamically allocated based on the provided resources. The pilot manager coordinates
            these resources to support task execution.

        resource_pilot (rp.Pilot): Represents the submitted computing resources as a pilot.
            This pilot is described by the `resources` parameter provided during initialization,
            specifying details such as CPU, GPU, and memory requirements.

    Parameters:
        resources (Dict): A dictionary specifying the resource requirements for the pilot,
            including details like the number of CPUs, GPUs, and memory. This dictionary
            configures the pilot to match the needs of the tasks that will be executed.

    Raises:
        Exception: If session creation, pilot submission, or task manager setup fails,
            the ResourceEngine will raise an exception, ensuring the resources are correctly
            allocated and managed.

    Example:
        ```python
        resources = {"cpu": 4, "gpu": 1, "memory": "8GB"}
        engine = ResourceEngine(resources)
        ```
    """

    @typeguard.typechecked
    def __init__(self, resources: Dict) -> None:
        try:
            self.session = rp.Session(uid=ru.generate_id('rose.session',
                                                         mode=ru.ID_PRIVATE))
            self.task_manager = rp.TaskManager(self.session)
            self.pilot_manager = rp.PilotManager(self.session)
            self.resource_pilot = self.pilot_manager.submit_pilots(rp.PilotDescription(resources))
            self.task_manager.add_pilots(self.resource_pilot)

            print('Resource Engine started successfully\n')

        except Exception:
            print('Resource Engine Failed to start, terminating\n')
            raise

        except (KeyboardInterrupt, SystemExit):
            # the callback called sys.exit(), and we can here catch the
            # corresponding KeyboardInterrupt exception for shutdown.  We also catch
            # SystemExit (which gets raised if the main threads exits for some other
            # reason).
            raise

    def state(self):
        return self.resource_pilot

    def task_state_cb(self, task, state):
        pass

    def shutdown(self) -> None:
        self.session.close(download=True)


class WorkflowEngine:
    """
    A WorkflowEngine manages and executes tasks within a Directed Acyclic Graph (DAG)
    structure, allowing for complex workflows where tasks may have dependencies. Each
    node in the DAG represents a distinct task, even if some nodes have identical labels.
    This allows for flexible, reusable task definitions in workflows with intricate dependencies.

    In this DAG, nodes (tasks) can have the same label or identifier, but they represent distinct
    entities within the workflow.

    Example:
        Consider a simple DAG representation:

        Nodes: A, A (two distinct nodes with identical labels)
        Edges: A → B, A → C

        Here, two nodes labeled 'A' are distinct instances connected to 'B' and 'C',
        each with its own role in the workflow.

    Attributes:
        engine (ResourceEngine): An instance of `ResourceEngine`, responsible for managing the
            runtime resources needed to execute tasks. This engine is agnostic to the specific
            runtime context (RCT).

        sequential_execution (bool): A flag indicating if tasks should be gathered and run in
            parallel wherever dependencies allow or sequentially. If `True`, tasks are executed
            in sequence according to dependencies. if `False` (default), it will run tasks
            concurrently when possible.

        tasks (dict): A dictionary storing task identifiers and associated task objects (futures).
            This enables tracking of task states and results as the workflow progresses.

        dependencies (dict): A dictionary that maps each task to its list of dependencies,
            enabling the engine to resolve dependencies before executing each task.

        task_manager: This attribute references the `task_manager` provided by the `ResourceEngine`,
            which handles the underlying task operations and states.
    """

    @typeguard.typechecked
    def __init__(self, engine: ResourceEngine,
                 sequential_execution: bool = False) -> None:

        self.tasks = {}
        self.engine = engine
        self.dependencies = {}
        self.task_manager = self.engine.task_manager
        self.sequential_execution = sequential_execution

        if not self.sequential_execution:
            print('Workflow engine will use conccurent tasks\
                   execution strategy when is possible!\n')
        else:
            print('Workflow engine will use sequential tasks\
                   execution strategy!\n')

    @typeguard.typechecked
    def __call__(self, func: Callable):
        """Use RoseEngine as a decorator to register workflow tasks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_descriptions = func(*args, **kwargs)
            task_descriptions['name'] = func.__name__
            task_descriptions['uid'] = self.__assign_task_uid()

            task_deps, input_files_deps, output_files_deps = self._detect_dependencies(args)

            task_descriptions['metadata'] = {'dependencies': task_deps,
                                             'input_files': input_files_deps,
                                             'output_files': output_files_deps}

            task_fut = Future()  # Create a Future object for this task
            task_fut.id = task_descriptions['uid'].split('task.')[1]

            # Store the future and task description in the tasks dictionary, keyed by UID
            self.tasks[task_descriptions['uid']] = {'future': task_fut,
                                                    'description': task_descriptions}
            self.dependencies[task_descriptions['uid']] = task_deps

            msg = f"Registered task '{task_descriptions['name']}' and id of {task_fut.id}"
            msg += f" with dependencies: {[dep['name'] for dep in task_deps]}"
            print(msg)

            if self.sequential_execution:
                self.run()

            return task_descriptions

        return wrapper

    def __assign_task_uid(self):
        uid = ru.generate_id('task.%(item_counter)06d',
                             ru.ID_CUSTOM, ns=self.engine.session.uid)
        return uid

    def link_data_deps(self, task_id, file_name=None):
        if not file_name:
            file_name = task_id

        data_deps = {'source': f"pilot:///{task_id}/{file_name}",
                     'target': f"task:///{file_name}", 'action': rp.TRANSFER}

        return data_deps

    def _detect_dependencies(self, possible_dependencies):

        dependencies = []
        input_files = []
        output_files = []

        for possible_dep in possible_dependencies:
            # it is a task deps
            if isinstance(possible_dep, rp.TaskDescription):
                dependencies.append(possible_dep)
            # it is input file needs to be obtained from somewhere
            elif isinstance(possible_dep, InputFile):
                input_files.append(possible_dep.filename)
            # it is output file needs to be obtained from the task folder
            elif isinstance(possible_dep, OutputFile):
                output_files.append(possible_dep.filename)

        return dependencies, input_files, output_files

    def clear(self):

        self.tasks.clear()
        self.dependencies.clear()

    def run(self):

        # Iteratively resolve dependencies and submit tasks when ready
        resolved = set()  # Track tasks that have been resolved
        unresolved = set(self.dependencies.keys())  # Start with all tasks unresolved

        while unresolved:
            to_submit = []  # Collect tasks to submit this round

            for task_uid in list(unresolved):
                if self.tasks[task_uid]['future'].done():
                    resolved.add(task_uid)
                    unresolved.remove(task_uid)
                    continue

                dependencies = self.dependencies[task_uid]
                # Check if all dependencies have been resolved
                if all(dep['uid'] in resolved for dep in dependencies):
                    task_desc = self.tasks[task_uid]['description']

                    input_staging = []

                    # Gather staging information for input files
                    for dep in dependencies:
                        dep_desc = self.tasks[dep['uid']]['description']
                        for output_file in dep_desc.metadata['output_files']:
                            if output_file in task_desc.metadata['input_files']:
                                input_staging.append(self.link_data_deps(dep['uid'], output_file))

                    # Add independent input files to input_staging: local file, https file
                    for input_file in task_desc.metadata['input_files']:
                        _data_target = [item['target'].split('/')[-1] for item in input_staging]
                        if input_file not in _data_target:
                            # FIXME: link_data_deps() must be able to link input files
                            input_staging.append({'source': input_file,
                                                  'target': f"task:///{input_file}",
                                                  'action': rp.TRANSFER})

                    task_desc.input_staging = input_staging

                    # Add the task to the submission list
                    to_submit.append(task_desc)
                    msg = f"Task '{task_desc.name}' ready to submit;"
                    msg += f" resolved dependencies: {[dep['name'] for dep in dependencies]}"
                    print(msg)

            if to_submit:
                # Submit collected tasks concurrently and track their futures
                self.submit(to_submit)

            # make sure to update dependencies records only when tasks are submitted/succeeded
            for task in to_submit:
                resolved.add(task.uid)
                unresolved.remove(task.uid)

    def submit(self, tasks):

        print(f'submitting {[t.name for t in tasks]} for execution')

        # This assumes `submit_tasks` can take a list of task descriptions
        submitted_tasks = self.task_manager.submit_tasks(tasks)

        # Wait for all tasks to complete
        self.task_manager.wait_tasks([task.uid for task in submitted_tasks])

        # Set the result for each future
        for task in submitted_tasks:
            task_fut = self.tasks[task.uid]['future']

            if task.state in [rp.FAILED, rp.CANCELED]:
                task_fut.set_exception(task.state)
                self.tasks[task.uid]['description'].stderr = task.stderr

            elif task.state == rp.DONE:
                task_fut.set_result(task.state)  # Set the result to the future
                self.tasks[task.uid]['description'].stdout = task.stdout

            print(f'Task {task.name} finished with state: {task.state}')
