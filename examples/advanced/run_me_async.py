import os
import sys
import time
import asyncio
from rose.learner import ActiveLearner
from rose.metrics import MODEL_ACCURACY
from radical.flow import Task, ResourceEngine

async def main():
    engine = ResourceEngine({'runtime': 30,
                             'resource': 'local.localhost'})

    learner = ActiveLearner(engine)
    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @learner.simulation_task
    async def simulation(*args):
        return Task(executable=f'{code_path}/simulation.py')

    # Define and register the training task
    @learner.training_task
    async def training(*args):
        return Task(executable=f'{code_path}/training.py')

    # Define and register the active learning task
    @learner.active_learn_task
    async def active_learn(*args):
        return Task(executable=f'{code_path}/active_learn.py')

    # Defining the stop criterion with a metric (MSE in this case)
    @learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=0.99)
    async def check_accuracy(*args):
        return Task(executable=f'{code_path}/check_accuracy.py')

    async def teach(i):
        # 3 iterations of active learn
        print(f'Starting AL workflow-{i} at {time.time()}')
        for acl_iter in range(3):
            print(f'Starting Iteration-{acl_iter}')
            simul = simulation()
            train = training(simul)
            active = active_learn(simul, train)
            check_result = await check_accuracy(active)

            should_stop, metric_val = learner.check_stop_criterion(check_result)

            if should_stop:
                print(f'Accuracy ({metric_val}) met the threshold, breaking...')
                break

    # Run workflows concurrently
    results = await asyncio.gather(*[teach(i) for i in range(2)])
    print('all done')
    engine.shutdown()

if __name__ == '__main__':
    asyncio.run(main())


