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


def active_learning_cycle(iterations=5, accuracy_threshold=0.95):
    
    accuracy = 0.0
    
    for i in range(iterations):
        
        print(f"\nActive learning iteration {i + 1}")

        # Step 1: Generate or simulate data
        simulated_data = sim(OutputFile('data_simulated.npz'))
        

        acl(train(InputFile('data_simulated.npz'),  # <-- Data Dependency
                  OutputFile('accuracy_score.txt'), # <-- Data Dependency
                  simulated_data),                  # <-- Task Dependency
            InputFile('accuracy_score.txt')         # <-- Data Dependency
            ) 

        flow.run()

        import random
        accuracy = random.random()
        print(f"Iteration {i + 1}: Accuracy = {accuracy}")

        if accuracy >= accuracy_threshold:
            print("Accuracy threshold met, stopping active learning loop.")
            break

    engine.shutdown()

    
    print("Active learning cycle completed.")



active_learning_cycle()
