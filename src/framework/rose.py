import os
import radical.entk as re
import radical.utils as ru
import radical.pilot as rp

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor


class SimulationTask(rp.TaskDescription):
    def __init__(self, **kwargs):
        kwargs['name'] = 'SimulationTask'
        super().__init__(kwargs)

class TrainingTask(rp.TaskDescription):
    def __init__(self, **kwargs):
        kwargs['name'] = 'TrainingTask'
        super().__init__(kwargs)

class ActiveLearnTask(rp.TaskDescription):
    def __init__(self, **kwargs):
        kwargs['name'] ='ActiveLearnTask'
        super().__init__(kwargs)

class RoseResource(rp.PilotDescription):
    def __init__(self, **kwargs):
        super().__init__(kwargs)


class RoseWorkflow():
    def __init__(self):
        pass

class SerialWorkflow(RoseWorkflow):
    def __init__(self, simulation_task, training_task,
                 activelearn_task, max_iterations=10):
        
        if not all(isinstance(task, rp.TaskDescription) for task in [simulation_task,
                                                                     training_task,
                                                                     activelearn_task]):
            raise TypeError(f"All workflow tasks must be of type '{rp.TaskDescription}' ")
        
        self.state = 'idle'
        self.simulation_task = simulation_task
        self.training_task = training_task
        self.activelearn_task = activelearn_task

        self.max_iterations = max_iterations

        super().__init__()

    
    def run(self, resources:rp.PilotDescription, replicas=1):

        submitted_workflows = None

        if self.state == 'idle':
            self.state = 'running'

        elif self.state == 'running':
            print(f'{self.__class__.__name__} is still running')
            return

        if not isinstance(resources, rp.PilotDescription):
            raise TypeError(f'resources must be of type {rp.PilotDescription}')

        try:
            session =  rp.Session()
            task_manager = rp.TaskManager(session)
            pilot_manager = rp.PilotManager(session)

            resource_pilot = pilot_manager.submit_pilots(resources)

            task_manager.add_pilots(resource_pilot)

            def _submit(n=1):

                for phase in range(self.max_iterations):

                    simulation_task = task_manager.submit_tasks(self.simulation_task)
                    
                    task_manager.wait_tasks(self.simulation_task.uid)

                    training_task = task_manager.submit_tasks(self.training_task)

                    task_manager.wait_tasks(self.training_task.uid)

                    activelearn_task = task_manager.submit_tasks(self.activelearn_task)

                    task_manager.wait_tasks(self.activelearn_task.uid)

            with ThreadPoolExecutor() as submitter:
                # Fire off workflows and don't wait for results
                submitted_workflows = [submitter.submit(_submit, i) for i in range(replicas)]

            for future in as_completed(submitted_workflows):
                print(f"Result: {future.result()}")


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

