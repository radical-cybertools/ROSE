from data import InputFile, OutputFile
from roseee import ResourceEngine, WorkflowEngine
from rose import SimulationTask, ActiveLearnTask, TrainingTask

engine = ResourceEngine({'resource': 'local.localhost'})
flow = WorkflowEngine(engine=engine)
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


@flow
def downaload_and_echo(*args):
    return ActiveLearnTask(executable=f"/bin/cat sample3.txt")

for i in range(5):
    acl(train(sim(OutputFile('data_simulated.npz')
                  ),
              InputFile('data_simulated.npz'),
              OutputFile('accuracy_score.txt')
              ),
    InputFile('accuracy_score.txt')
    )

    flow.run()
    flow.clear()


# downaload_and_echo(InputFile('https://filesamples.com/samples/document/txt/sample3.txt'))
# flow.run()

engine.shutdown()
