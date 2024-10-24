from rose import SerialWorkflow, RoseResource
from rose import SimulationTask, ActiveLearnTask, TrainingTask

sim_task = SimulationTask(executable='/bin/echo I am simulation task')
train_task = TrainingTask(executable='/bin/echo I am Training task')
active_learn_task = ActiveLearnTask(executable='/bin/echo I am AL task')

serial_wf = SerialWorkflow(simulation_task=sim_task,
                           training_task=train_task,
                           activelearn_task=active_learn_task)


resources = RoseResource(resource='local.localhost')

serial_wf.run(resources=resource)