import os
import sys
import asyncio

from rose.al.selector import AlgorithmSelector
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

try:
    import numpy, sklearn
except ImportError:
    print("\nRun 'pip install numpy scikit-learn' to use this example.\n")
    sys.exit(1)

async def select_algorithm():
    engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
    asyncflow = await WorkflowEngine.create(engine)
    als = AlgorithmSelector(asyncflow)

    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @als.simulation_task
    async def simulation(*args):
        return f'{code_path}/sim.py'

    # Define and register the training task
    @als.training_task
    async def training(*args):
        return f'{code_path}/train.py'

    # Define and register Multiple AL tasks
    @als.active_learn_task(name='algo_1')
    async def active_learn_1(*args):
        return f'{code_path}/active_1.py'

    @als.active_learn_task(name='algo_2')
    async def active_learn_2(*args):
        return f'{code_path}/active_2.py'

    # Defining the stop criterion with a metric (MSE in this case)
    @als.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.01)
    async def check_mse(*args):
        return f'{code_path}/check_mse.py'

    # Start the teaching process
    await als.teach_and_select(max_iter=4)
    await als.shutdown()

if __name__ == "__main__":
    asyncio.run(select_algorithm())
