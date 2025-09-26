import os
import sys
import asyncio

from rose.metrics import MEAN_SQUARED_ERROR_MSE
from rose.al.active_learner import SequentialActiveLearner

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend


async def rose_al():

    engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
    asyncflow = await WorkflowEngine.create(engine)

    acl = SequentialActiveLearner(asyncflow)
    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @acl.simulation_task
    async def simulation(*args):
        return f'{code_path}/sim.py'

    # Define and register the training task
    @acl.training_task
    async def training(*args):
        return f'{code_path}/train.py'

    # Define and register the active learning task
    @acl.active_learn_task
    async def active_learn(*args):
        return f'{code_path}/active.py'

    # Defining the stop criterion with a metric (MSE in this case)
    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
    async def check_mse(*args):
        return f'{code_path}/check_mse.py'

    # Start the teaching process
    await acl.teach()
    await acl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_al())