import asyncio
import os
import sys

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import RadicalExecutionBackend

from rose.al.active_learner import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE


async def rose_al():
    engine = await RadicalExecutionBackend({"resource": "local.localhost"})
    asyncflow = await WorkflowEngine.create(engine)

    acl = SequentialActiveLearner(asyncflow)
    code_path = f"{sys.executable} {os.getcwd()}"

    # Define and register the simulation task
    @acl.simulation_task
    async def simulation(*args, task_description={"shell": True}):
        return f"{code_path}/sim.py"

    # Define and register the training task
    @acl.training_task
    async def training(*args, task_description={"shell": True}):
        return f"{code_path}/train.py"

    # Define and register the active learning task
    @acl.active_learn_task
    async def active_learn(*args, task_description={"shell": True}):
        return f"{code_path}/active.py"

    # Defining the stop criterion with a metric (MSE in this case)
    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
    async def check_mse(*args, task_description={"shell": True}):
        return f"{code_path}/check_mse.py"

    # Start the active learning process
    async for state in acl.start():
        print(f"Iteration {state.iteration}: metric={state.metric_value}")

    await acl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_al())
