# flake8: noqa
import queue
import time
import threading

from functools import wraps
from typing import Callable, Dict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor

import radical.utils as ru
import radical.pilot as rp

import typeguard
from .data import InputFile, OutputFile

class Task(rp.TaskDescription):
    """
    Represents a task description by extending the `TaskDescription` class from `rp`
    (an external module).

    This class is primarily used to define and manage the details of a task, inheriting
    properties and methods from the `rp.TaskDescription` base class. Additional arguments
    and keyword arguments can be passed to further configure the task, which are then
    forwarded to the base class.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the parent class constructor, if needed.
    **kwargs : dict
        Keyword arguments to configure the task. Passed directly to the `TaskDescription`
        initializer.

    Methods
    -------
    None. This class relies on inherited methods from `rp.TaskDescription`.
    """

    def __init__(self, **kwargs):
        # we pass only the kwargs as dict to the rp.TaskDescription
        # as it only allows from_dict
        super().__init__(from_dict=kwargs)


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
            are dynamically allocated based on the provided resources. The pilot manager
            coordinates these resources to support task execution.

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
            self._session = rp.Session(uid=ru.generate_id('rose.session',
                                                          mode=ru.ID_PRIVATE))
            self.task_manager = rp.TaskManager(self._session)
            self.pilot_manager = rp.PilotManager(self._session)
            self.resource_pilot = self.pilot_manager.submit_pilots(rp.PilotDescription(resources))
            self.task_manager.add_pilots(self.resource_pilot)

            print('Resource Engine started successfully\n')

        except Exception:
            print('Resource Engine Failed to start, terminating\n')
            raise

        except (KeyboardInterrupt, SystemExit) as e:
            # the callback called sys.exit(), and we can here catch the
            # corresponding KeyboardInterrupt exception for shutdown.  We also catch
            # SystemExit (which gets raised if the main threads exits for some other
            # reason).
            excp_msg = f'Resource engine failed internally, please check {self._session.path}'
            raise SystemExit(excp_msg) from e

    def state(self):
        """
        Retrieve the current state of the resource pilot.

        Returns:
            The current state of the resource pilot.
        """
        raise NotImplementedError

    def task_state_cb(self, task, state):
        """
        Callback function for handling task state changes.

        Args:
            task: The task object whose state has changed.
            state: The new state of the task.
        
        Note:
            This method is intended to be overridden or extended
            to perform specific actions when a task's state changes.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """
        Gracefully shuts down the session, downloading any necessary data.

        This method ensures that the session is properly closed and any
        required data is downloaded before finalizing the shutdown.

        Returns:
            None
        """
        print('Shutdown is triggered, terminating the resources gracefully')
        self._session.close(download=True)


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

        tasks (dict): A dictionary storing task identifiers and associated task objects (futures).
            This enables tracking of task states and results as the workflow progresses.

        dependencies (dict): A dictionary that maps each task to its list of dependencies,
            enabling the engine to resolve dependencies before executing each task.

        task_manager: This attribute references the `task_manager` provided by the
            `ResourceEngine`, which handles the underlying task operations and states.
    """

    @typeguard.typechecked
    def __init__(self, engine: ResourceEngine) -> None:
        self.tasks = {}
        self.engine = engine
        self.resolved = set()
        self.dependencies = {}
        self.unresolved = set()
        self.queue = queue.Queue()
        self.task_manager = self.engine.task_manager

        # Start the submission thread
        submission_thread = threading.Thread(target=self.submit, name='WFSubmitThread')
        submission_thread.daemon = True
        submission_thread.start()

        # Start the run method in a background thread
        run_thread = threading.Thread(target=self.run, name='WFRunThread')
        run_thread.daemon = True
        run_thread.start()

        self.task_manager.register_callback(self.callbacks)
        self._conccurent_wf_submitter = ThreadPoolExecutor()

    def as_async(self, func: Callable):
        """
        A decorator to run `blocking` function in a seperate thread.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Submit the function to the thread pool and return the Future
            future = self._conccurent_wf_submitter.submit(func, *args, **kwargs)
            return future  # The caller can wait for the future's result if needed

        return wrapper

    @typeguard.typechecked
    def __call__(self, func: Callable):
        """Use RoseEngine as a decorator to register workflow tasks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_descriptions = func(*args, **kwargs)
            task_descriptions['name'] = func.__name__
            task_descriptions['uid'] = self._assign_task_uid()

            task_deps, input_files_deps, output_files_deps = self._detect_dependencies(args)

            task_descriptions['metadata'] = {'dependencies': task_deps,
                                             'input_files': input_files_deps,
                                             'output_files': output_files_deps}

            task_fut = Future()  # Create a Future object for this task
            task_fut.id = task_descriptions['uid'].split('task.')[1]
            task_fut.task = task_descriptions

            # Store the future and task description in the tasks dictionary, keyed by UID
            self.tasks[task_descriptions['uid']] = {'future': task_fut,
                                                    'description': task_descriptions}
            self.dependencies[task_descriptions['uid']] = task_deps

            msg = f"Registered task '{task_descriptions['name']}' and id of {task_fut.id}"
            msg += f" with dependencies: {[dep['name'] for dep in task_deps]}"
            print(msg)

            return task_fut

        return wrapper

    @staticmethod
    def shutdown_on_failure(func: Callable):
        """
        Decorator that calls `shutdown` if an exception occurs in the decorated function.
        """
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                print('Internal failure is detected, shutting down the resource engine')
                self.engine.shutdown()  # Call shutdown on exception
                raise e
        return wrapper

    def _assign_task_uid(self):
        """
        Generates a unique identifier (UID) for a task.

        This method generates a custom task UID based on the format `task.%(item_counter)06d`
        and assigns it a session-specific namespace using the engine session UID.

        Returns:
            str: The generated unique identifier for the task.
        """
        uid = ru.generate_id('task.%(item_counter)06d',
                             ru.ID_CUSTOM, ns=self.engine._session.uid)
        return uid

    def link_explicit_data_deps(self, task_id, file_name=None):
        """
        Creates a dictionary linking explicit data dependencies between tasks.

        This method defines the source and target for data transfer between tasks, 
        using the provided task ID and optional file name. If no file name is provided, 
        the task ID is used as the file name.

        Args:
            task_id (str): The task ID to link data dependencies from.
            file_name (str, optional): The file name to be used in the data dependency. 
                                        Defaults to None, in which case the task_id is used.

        Returns:
            dict: A dictionary containing the data dependencies, including source, 
                target, and transfer action.
        """
        if not file_name:
            file_name = task_id

        data_deps = {'source': f"pilot:///{task_id}/{file_name}",
                     'target': f"task:///{file_name}", 'action': rp.LINK}

        return data_deps

    def link_implicit_data_deps(self, src_task):
        """
        Generates commands to link implicit data dependencies for a source task.

        This method creates shell commands to copy files from the source task's sandbox
        to the current task's sandbox, excluding task-related files. The commands are 
        returned as a list of Python shell commands to execute in the task environment.

        Args:
            src_task (Task): The source task whose files are to be copied.

        Returns:
            list: A list of shell commands to link implicit data dependencies between tasks.
        """
        cmd1 = f'export SRC_TASK_ID={src_task.uid}'
        cmd2 = f'export SRC_TASK_SANDBOX="$RP_PILOT_SANDBOX/$SRC_TASK_ID"'

        cmd3 = '''files=$(cd "$SRC_TASK_SANDBOX" && ls | grep -ve "^$SRC_TASK_ID")
                  for f in $files
                  do 
                     ln -sf "$SRC_TASK_SANDBOX/$f" "$RP_TASK_SANDBOX"
                  done'''

        commands = [cmd1, cmd2, cmd3]

        return commands

    def _detect_dependencies(self, possible_dependencies):
        """
        Detects and categorizes possible dependencies into tasks, input files, and output files.

        This method iterates over a list of possible dependencies and classifies them into three
          categories:
        - Tasks that are instances of `Future` with a `task` attribute (task dependencies).
        - Input files that are instances of `InputFile` (files required by the task).
        - Output files that are instances of `OutputFile` (files produced by the task).

        Args:
            possible_dependencies (list): A list of possible dependencies, which can include
            tasks, input files, and output files.

        Returns:
            tuple: A tuple containing three lists:
                - `dependencies`: A list of tasks that need to be completed.
                - `input_files`: A list of input file names that need to be fetched.
                - `output_files`: A list of output file names that need to be retrieved
                   from the task folder.
        """
        dependencies = []
        input_files = []
        output_files = []

        for possible_dep in possible_dependencies:
            # it is a task deps
            if isinstance(possible_dep, Future) and hasattr(possible_dep, 'task'):
                possible_dep = possible_dep.task
                dependencies.append(possible_dep)
            # it is input file needs to be obtained from somewhere
            elif isinstance(possible_dep, InputFile):
                input_files.append(possible_dep.filename)
            # it is output file needs to be obtained from the task folder
            elif isinstance(possible_dep, OutputFile):
                output_files.append(possible_dep.filename)

        return dependencies, input_files, output_files

    def _clear(self):
        """
        clear worfklow tasks and their deps
        """

        self.tasks.clear()
        self.dependencies.clear()

    @shutdown_on_failure
    def run(self):
        """Background method to resolve dependencies and submit tasks."""

        while True:
            # Continuously try to resolve dependencies and submit tasks as they become ready
            self.unresolved = set(self.dependencies.keys())  # Start with all tasks unresolved

            if not self.unresolved:
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage in the loop
                continue

            to_submit = []  # Collect tasks to submit in each iteration

            for task_uid in list(self.unresolved):
                if self.tasks[task_uid]['future'].done():
                    self.resolved.add(task_uid)
                    self.unresolved.remove(task_uid)
                    continue

                if self.tasks[task_uid]['future'].running():
                    continue

                dependencies = self.dependencies[task_uid]
                # Check if all dependencies have been resolved and are done
                if all(dep['uid'] in self.resolved and self.tasks[dep['uid']]['future'].done() for dep in dependencies):
                    task_desc = self.tasks[task_uid]['description']

                    input_staging = []

                    # Gather staging information for input files
                    for dep in dependencies:
                        dep_desc = self.tasks[dep['uid']]['description']

                        # implicit data dependencies
                        if not dep_desc.metadata.get('output_files'):
                            task_desc.pre_exec.extend(self.link_implicit_data_deps(dep_desc))

                        # explicit data dependencies
                        for output_file in dep_desc.metadata['output_files']:
                            if output_file in task_desc.metadata['input_files']:
                                input_staging.append(self.link_explicit_data_deps(dep['uid'],
                                                                                output_file))

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
                self.queue.put(to_submit)
                for t in to_submit:
                    self.tasks[t.uid]['future'].set_running_or_notify_cancel()
                    self.resolved.add(t.uid)
                    self.unresolved.remove(t.uid)

            time.sleep(0.5)  # Small delay to prevent excessive CPU usage in the loop

    def callbacks(self, task, state):
        """
        Callback function to handle task state changes and set results or exceptions.

        This method is called when the state of a task changes. It updates the corresponding 
        `future` in the task dictionary with either the task's result or an exception, 
        depending on the final state of the task.

        Args:
            task (rp.Task): The task whose state has changed.
            state (str): The current state of the task (e.g., rp.DONE, rp.FAILED, rp.CANCELED).

        Raises:
            Exception: If the task has failed or was canceled, an exception is set on the future.
        """
        task_fut = self.tasks[task.uid]['future']

        if state == rp.DONE:
            print(f'{task.uid} is DONE')
            task_fut.set_result(task.stdout)

        elif state in [rp.FAILED, rp.CANCELED]:
            print(f'{task.uid} is FAILED')
            task_fut.set_exception(Exception(task.stderr))

    @shutdown_on_failure
    def submit(self):
        """
        Submits tasks from the queue for execution in a loop until an error occurs.

        This method repeatedly checks the task queue for new tasks and submits them for 
        execution using the `task_manager.submit_tasks()` method. If the queue is empty, 
        the method waits briefly before checking again. If any exceptions are encountered 
        during the submission process, they are caught and logged.

        The method will keep running until it encounters an error or is explicitly stopped.

        Raises:
            Exception: If any exception occurs while submitting tasks, it is logged.
        """
        while True:
            try:
                tasks = self.queue.get(timeout=1)
                print(f'submitting {[t.name for t in tasks]} for execution')
                self.task_manager.submit_tasks(tasks)
            except queue.Empty:
                time.sleep(0.5)
