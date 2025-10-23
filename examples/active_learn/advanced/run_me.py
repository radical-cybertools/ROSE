import os
import sys
import asyncio

from rose.learner import Learner
from rose.metrics import MODEL_ACCURACY

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ConcurrentExecutionBackend

from concurrent.futures import ThreadPoolExecutor

try:
    import numpy, sklearn
except ImportError:
    print("\nRun 'pip install numpy scikit-learn' to use this example.\n")
    sys.exit(1)

async def custom_al():

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = Learner(asyncflow)
    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @learner.simulation_task
    async def simulation(*args):
        return f'{code_path}/simulation.py'

    # Define and register the training task
    @learner.training_task
    async def training(*args):
        return f'{code_path}/training.py'

    # Define and register the active learning task
    @learner.active_learn_task
    async def active_learn(*args):
        return f'{code_path}/active_learn.py'

    # Defining the stop criterion with a metric (MSE in this case)
    @learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=0.99)
    async def check_accuracy(*args):
        return f'{code_path}/check_accuracy.py'

    async def teach():
        # 10 iterations of active learn
        for acl_iter in range(10):
            print(f'Starting Iteration-{acl_iter}')
            sim = simulation() # <-- this returns a future
            train = training(sim) # <-- wait for sim task first
            active = active_learn(sim, train) # <-- wait for sim and train

            # wait for active learn task and obtain result once done
            should_stop, metric_val = await check_accuracy(active)

            if should_stop:
                print(f"Accuracy ({metric_val}) met user's threshold, breaking...")
                break

    await teach()
    await learner.shutdown()

if __name__ == "__main__":
    asyncio.run(custom_al())
