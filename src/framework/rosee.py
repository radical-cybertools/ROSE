import copy
import time
import radical.pilot as rp
import radical.utils as ru

from functools import wraps
from concurrent.futures import Future
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

class RoseEngine:
    def __init__(self, resources) -> None:
        try:
        
            self.session = rp.Session()
            self.task_manager = rp.TaskManager(self.session)
            self.pilot_manager = rp.PilotManager(self.session)
            self.resource_pilot = self.pilot_manager.submit_pilots(rp.PilotDescription(resources))
            self.task_manager.add_pilots(self.resource_pilot)
        
        except Exception as e:
            print(f'RoseEngine Failed to start due to {e}, terminating')
            self.shutdown()

    def state(self):
        return self.resource_pilot


    def shutdown(self):
        self.session.close(download=True)


class RoseWorkflow:
    def __init__(self, engine):
        self.tasks = []
        self.main = []
        #self.tasks_book = {}
        self.engine = engine
        self.task_manager = self.engine.task_manager

    def __call__(self, func):
        """Use RoseEngine as a decorator to register workflow tasks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Run the function to generate tasks and add them to self.tasks
            #task_fut = Future()

            task_descriptions = func(*args, **kwargs)
            task_descriptions['name'] = func.__name__
            task_descriptions['uid'] = ru.generate_id('task.%(item_counter)06d',
                                                      ru.ID_CUSTOM, ns=self.engine.session.uid)
            _main = []
            for possible_task in args:
                # main task that has dependencies of dependencies
                if isinstance(possible_task, rp.TaskDescription):
                    if possible_task in self.tasks:
                        if possible_task not in self.main:
                            self.tasks.remove(possible_task)
                            _main.append(possible_task)

            self.main.extend(_main)

            task_descriptions['metadata'] = {'args': args, 'kwargs':kwargs}
            self.tasks.append(task_descriptions)

            #self.tasks_book.update({task_fut: task_descriptions})

            return task_descriptions

        return wrapper
    

    def set_data_deps(self, src_task, target_task):

        target_task.input_staging.append({'source': f'pilot://{src_task.uid}/{src_task.uid}.out', 
                                          'target': f'task://{src_task.uid}.out', 
                                          'action': rp.TRANSFER})
    

    def clear(self):
        self.main.clear()
        self.tasks.clear()


    def run(self):

        #total_workflow_tasks = copy.copy(self.tasks)

        def _submit():
            done_workflow_tasks = 0
            depending_on_task_outputs = []
            
            submitted_tasks = set()  # Track tasks that have been submitted to avoid duplicates
            completed_tasks = set()  # Track completed tasks

            while self.tasks:  # Process tasks as long as there are any remaining
                task = self.tasks[0]  # Peek at the first task in the list without popping
                print(f'processing task {task.name}')

                # Collect dependencies for the task
                dependee = []
                for arg in task['metadata']['args']:
                    parent_task = arg
                    if isinstance(parent_task, rp.TaskDescription):
                        if parent_task.uid not in completed_tasks:
                            print(f'found unresolved deps for task {task.name}: {parent_task.name}')
                            dependee.append(parent_task)
                        else:
                            print('setting data deps as task is done already')
                            self.set_data_deps(parent_task, task)
                    else:
                        raise TypeError(f'Depndee tasks must be of type {rp.TaskDescription}')

                # Check if all dependencies are complete
                if dependee:
                    # Submit and wait on dependencies only if they haven't been submitted already
                    unsolved_dependencies = [t for t in dependee if t.uid not in submitted_tasks]
                    if unsolved_dependencies:
                        print(f'executing {[t.name for t in unsolved_dependencies]}')
                        done_or_failed_tasks = self.task_manager.submit_tasks(unsolved_dependencies)
                        self.task_manager.wait_tasks([t.uid for t in done_or_failed_tasks])

                        for t in done_or_failed_tasks:
                            if t.state in [rp.FAILED, rp.CANCELED]:
                                raise Exception(f'Task {t.uid} failed, workflow will fail too')

                            completed_tasks.add(t.uid)  # Mark as complete
                            submitted_tasks.add(t.uid)  # Mark as submitted
                        
                            # Add stage-in requirements from the completed task
                            print(f'{t.name} is finished, adding its output to {task.name}')
                            self.set_data_deps(t, task)

                        done_workflow_tasks += len(done_or_failed_tasks)
                        print(f'tasks {[t.name for t in  done_or_failed_tasks]} are finished at {time.time()}')

                # Submit the main task if dependencies are resolved and it hasn't been submitted
                if task.uid not in submitted_tasks:
                    print(f'executing task {task.name}')
                    task.input_staging = depending_on_task_outputs
                    task['metadata'].clear()  # Clear metadata to prevent serialization issues

                    main_task = self.task_manager.submit_tasks(task)
                    self.task_manager.wait_tasks(main_task.uid)

                    if main_task.state == rp.FAILED:
                        raise Exception(f'Task {main_task.uid} failed due to: {main_task.stderr if main_task.stderr else main_task.exception}')

                    if main_task.state == rp.DONE:
                        done_workflow_tasks += 1
                        completed_tasks.add(main_task.uid)  # Mark main task as completed
                        submitted_tasks.add(main_task.uid)  # Mark main task as submitted
                        print(f'task {task.name} is finished at {time.time()}')

                    # Remove the completed task from the list
                    self.tasks.pop(0)
                else:
                    # Move to the next task if dependencies aren't ready
                    self.tasks.append(self.tasks.pop(0))

            if len(completed_tasks) == len(submitted_tasks):
                print(f"Workflow completed with {len(completed_tasks)} tasks out of {len(submitted_tasks)}")
            else:
                raise RuntimeError('workflows failed due to unresolved or unfinshed tasks')

        with ThreadPoolExecutor() as submitter:
            workflow_future = submitter.submit(_submit)
            workflow_future.result()
