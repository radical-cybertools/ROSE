import copy
import typeguard

import radical.utils as ru
import radical.pilot as rp

from functools import wraps
from concurrent.futures import Future
from data import InputFile, OutputFile
from typing import Callable, Dict, List


class ResourceEngine:
    
    @typeguard.typechecked
    def __init__(self, resources:Dict) -> None:
        try:
            self.session = rp.Session(uid=ru.generate_id('rose.session',
                                                         mode=ru.ID_PRIVATE))
            self.task_manager = rp.TaskManager(self.session)
            self.pilot_manager = rp.PilotManager(self.session)
            self.resource_pilot = self.pilot_manager.submit_pilots(rp.PilotDescription(resources))
            self.task_manager.add_pilots(self.resource_pilot)
        
        except Exception as e:
            print(f'RoseEngine Failed to start due to {e}, terminating')
            self.shutdown()

    def state(self):
        return self.resource_pilot


    def shutdown(self) -> None:
        self.session.close(download=True)


class WorkflowEngine:
    '''
    In a Directed Acyclic Graph (DAG), nodes can have identical labels
    or properties, but they must be distinct entities within the graph.

    Example:
    Consider the following DAG representation:

    Nodes: A, A (identical labels, but distinct instances)
    Edges: A → B, A → C
    In this case, you have two nodes labeled A, but they are different
    entities connected to B and C.
    '''
    
    @typeguard.typechecked
    def __init__(self, engine:ResourceEngine) -> None:
        self.tasks = {}        # Dictionary to store task futures and their descriptions
        self.engine = engine   # Runtime engine and theortically it should be agnostic from RCT
        self.dependencies = {} # Dictionary to track dependencies for each task
        self.task_manager = self.engine.task_manager
        self.workflows_book = []


    @typeguard.typechecked
    def __call__(self, func:Callable):
        """Use RoseEngine as a decorator to register workflow tasks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_descriptions = func(*args, **kwargs)
            task_descriptions['name'] = func.__name__
            task_descriptions['uid'] = ru.generate_id('task.%(item_counter)06d',
                                                      ru.ID_CUSTOM, ns=self.engine.session.uid)

            task_deps, input_files_deps, output_files_deps = self._detect_dependencies(args)

            task_descriptions['metadata'] = {'dependencies': task_deps,
                                             'input_files': input_files_deps,
                                             'output_files': output_files_deps}

            task_fut = Future()  # Create a Future object for this task
            self.tasks[task_fut] = task_descriptions  # Store task description with Future as key
            self.dependencies[task_descriptions['uid']] = task_deps

            # print(f"Registered task '{task_descriptions['name']}' with dependencies: {[dep['name'] for dep in dependencies]}")

            return task_descriptions

        return wrapper


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
        executed = set()  # Track tasks that have been successfully executed
        unresolved = set(self.dependencies.keys())  # Start with all tasks unresolved

        self.workflows_book.append(copy.copy(self.tasks))

        print(f'Now resolving and executing workflow-{len(self.workflows_book)}\n')

        while unresolved:
            to_submit = []  # Collect tasks to submit this round

            for task_uid in list(unresolved):
                dependencies = self.dependencies[task_uid]
                # Check if all dependencies have been resolved
                if all(dep['uid'] in resolved for dep in dependencies):
                    task_desc = next(t for fut, t in self.tasks.items() if t['uid'] == task_uid)

                    input_staging = []
                    
                    # Gather staging information for input files
                    for dep in dependencies:
                        dep_desc = next(t for t in self.tasks.values() if t['uid'] == dep['uid'])
                        for output_file in dep_desc['metadata']['output_files']:
                            if output_file in task_desc['metadata']['input_files']:
                                input_staging.append(self.link_data_deps(dep['uid'], output_file))

                    # Add independent input files to input_staging like local file, https file and so on
                    for input_file in task_desc['metadata']['input_files']:
                        if input_file not in [item['target'].split('/')[-1] for item in input_staging]:
                            # FIXME: link_data_deps() must be able to link input files
                            input_staging.append({'source': input_file,
                                                  'target': f"task:///{input_file}",
                                                  'action': rp.TRANSFER})

                    task_desc.input_staging = input_staging

                    # Add the task to the submission list
                    to_submit.append((task_desc, task_uid))
                    #print(f"Task '{task_desc['name']}' ready to submit; resolved dependencies: {[dep['name'] for dep in dependencies]}")

            if to_submit:
                # Submit collected tasks concurrently and track their futures
                self.submit(to_submit)

            # make sure to update dependencies records only when tasks are submitted/succeeded
            for t in to_submit:
                tuid = t[1]
                resolved.add(tuid)
                unresolved.remove(tuid)


    def submit(self, tasks):
        # Submit tasks in one go to the engine
        #if len(tasks) > 1:
        #    print(f'Executing {[t[0]["name"] for t in tasks]} conccurently')
        #else:
        #    print(f'Executing {[t[0]["name"] for t in tasks]}')

        # Submit the list of tasks
        task_futures = [next(fut for fut, desc in self.tasks.items() if desc['uid'] == task_uid) for _, task_uid in tasks]

        # This assumes `submit_tasks` can take a list of task descriptions
        submitted_tasks = self.task_manager.submit_tasks([task_desc for task_desc, _ in tasks])

        # Wait for all tasks to complete
        self.task_manager.wait_tasks([task.uid for task in submitted_tasks])

        # Set the result for each future
        for task_fut, task in zip(task_futures, submitted_tasks):
            if task.state in [rp.FAILED, rp.CANCELED]:
                print(f'{task.name} is Failed and has error of {task.stderr}')
                task_fut.set_exception('failed')
            elif task.state == rp.DONE:
                print(f'{task.name} is Done and has output of {task.stdout}')
                task_fut.set_result('Done')  # Set the result to the future
