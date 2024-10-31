from data import InputFile, OutputFile
from roseee import RoseWorkflow, RoseEngine
from rose import SimulationTask, ActiveLearnTask, TrainingTask

engine = RoseEngine({'resource': 'local.localhost'})
flow = RoseWorkflow(engine=engine)
code_path = 'python3 /home/aymen/RADICAL/ROSE/src/framework/samples/'

@flow
def sim(*args):
    return SimulationTask(executable=code_path+'simulate.py')

@flow
def train(*args):
    return TrainingTask(executable=code_path+'train.py')

@flow
def acl(*args):
    return ActiveLearnTask(executable=code_path+'active_learn.py')


for i in range(5):
    acl(train(sim(OutputFile('data_simulated.npz')
                  ),
              InputFile('data_simulated.npz'),
              OutputFile('accuracy_score.txt')
              ),
    InputFile('accuracy_score.txt')
    )

    flow.run()
    flow.tasks.clear()
    flow.dependencies.clear()

engine.shutdown()
