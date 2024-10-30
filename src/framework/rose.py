import radical.pilot as rp

from data import InputFile, OutputFile
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor


class SimulationTask(rp.TaskDescription):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'SimulationTask'
        for val in args:
            if isinstance(val, InputFile):
                kwargs['stage_in'] = val
            
            if isinstance(val, OutputFile):
                kwargs['stage_out'] = val

        super().__init__(kwargs)


class TrainingTask(rp.TaskDescription):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'TrainingTask'
        for val in args:
            if isinstance(val, InputFile):
                kwargs['stage_in'] = val
            
            if isinstance(val, OutputFile):
                kwargs['stage_out'] = val

        super().__init__(kwargs)

class ActiveLearnTask(rp.TaskDescription):
    def __init__(self, *args, **kwargs):
        kwargs['name'] ='ActiveLearnTask'
        for val in args:
            if isinstance(val, InputFile):
                kwargs['stage_in'] = val
            
            if isinstance(val, OutputFile):
                kwargs['stage_out'] = val

        super().__init__(kwargs)

class RoseResource(rp.PilotDescription):
    def __init__(self, **kwargs):
        super().__init__(kwargs)


class RoseWorkflow():
    def __init__(self):
        pass

    def _callbacks(self, tasks):
        pass

class ParalleWorkflow(RoseWorkflow):
    def __init__(self):
        pass

# maybe serial Workflow should be the base?
class SerialWorkflow(RoseWorkflow):
    def __init__(self, simulation_task, training_task,
                 activelearn_task, max_iterations=10):
        
        if not all(isinstance(task, rp.TaskDescription) for task in [simulation_task,
                                                                     training_task,
                                                                     activelearn_task]):
            raise TypeError(f"All workflow tasks must be of type '{rp.TaskDescription}' ")

        self.workflows_book = {}
        self.workflows_counter = 0
        self.runtime_state = 'idle'
        self.simulation_task = simulation_task
        self.training_task = training_task
        self.activelearn_task = activelearn_task

        self.max_iterations = max_iterations

        super().__init__()
    

    def _callbacks(self, task, state):
        pass

    
    def run(self, resources:rp.PilotDescription, replicas=1):

        submitted_workflows = None

        if not isinstance(resources, rp.PilotDescription):
            raise TypeError(f'resources must be of type {rp.PilotDescription}')

        if self.runtime_state != 'running':
            session =  rp.Session()
            task_manager = rp.TaskManager(session)
            pilot_manager = rp.PilotManager(session)

            resource_pilot = pilot_manager.submit_pilots(resources)
            task_manager.add_pilots(resource_pilot)

            self.runtime_state = 'running'

        try:
            def _submit():

                done_tasks = []
                workflow_id = f'wf.{self.workflows_counter}'

                for _ in range(self.max_iterations):
                    for t in [self.simulation_task, self.training_task, self.activelearn_task]:
                        t['name'] = workflow_id
                        task = task_manager.submit_tasks(t)
                        print(f'{task.uid} from {workflow_id} was submitted for execution')
                        task_manager.wait_tasks(task.uid)

                        if task.state in [rp.FAILED, rp.CANCELED]:
                            error = task.exception if not task.stderr else task.stderr
                            self.workflows_book[workflow_id]['workflow_state'] = rp.FAILED
                            print(f'{task.uid} from {workflow_id} failed with {error}, thus workflow execution failed as well')
                            return

                        elif task.state == rp.DONE:
                            done_tasks.append(task)
                            print(f'{task.uid} from {workflow_id} finshed succefully')

                if all(t.state == rp.DONE for t in done_tasks):
                    self.workflows_book[workflow_id]['workflow_state'] = rp.DONE

            with ThreadPoolExecutor() as submitter:
                # Fire off workflows and don't wait for results
                for _ in range(replicas):
                    print(f'workflow wf.{self.workflows_counter} is submitted')
                    workflow_future = submitter.submit(_submit)
                    self.workflows_book.update({f'wf.{self.workflows_counter}': {'workflow_state': rp.INITIAL,
                                                                                 'workflow_future': workflow_future,
                                                                                 'workflow_tasks':[self.simulation_task,
                                                                                                   self.training_task,
                                                                                                   self.activelearn_task]}})
                    self.workflows_counter +=1

            submitted_workflows = [wf['workflow_future'] for wf in self.workflows_book.values()]

            print('Waiting for all workflows to finish')

            [f.result() for f in as_completed(submitted_workflows)]


        except Exception as e:
            raise e

        except (KeyboardInterrupt, SystemExit):
            # the callback called sys.exit(), and we can here catch the
            # corresponding KeyboardInterrupt exception for shutdown.  We also catch
            # SystemExit (which gets raised if the main threads exits for some other
            # reason).
            raise

        finally:
            # always clean up the session, no matter if we caught an exception or
            # not.  This will kill all remaining pilots.
            print('finalize')
            session.close(download=True)

