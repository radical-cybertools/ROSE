# This is a *draft* example of the StreamingActiveLearner
# It cuts some corners in order to deliver a bare-bones test.
#     Namely: StreamingActiveLearner currently requires a stop criterion.
#         so, I have a dummy one that always returns true.
#         (and technically the stop criterion is usually executable, but
#         did this for brevity)
#
#     All tasks are function tasks. Currently, the only way to fetch state in
#         the model is if they are function tasks returning dictionaries. If these were
#         executable tasks, the developer would be responsible for their own data transfer
#         management.
#
# A more "real" example of the StreamingActiveLearner would eventually replace
# this quick test script.

import asyncio
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.streaming_learner import StreamingActiveLearner
from rose.learner import IterationState
from rose.metrics import MEAN_SQUARED_ERROR_MSE


async def rose_al():
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    acl = StreamingActiveLearner(asyncflow, batch_size=2)

    # Define and register the simulation task
    @acl.simulation_task(as_executable=False)
    async def simulation(window, *args):
        print(f"Start simulation: {window}")
        return {"train_window": window}

    # Define and register the training task
    @acl.training_task(as_executable=False)
    async def training(train_window, *args):
        print(f"Start training: {train_window}")
        out = []
        for i in train_window:
            out.append(i * 2)
        return out

    # Define and register the active learning task
    @acl.active_learn_task(as_executable=False)
    async def active_learn(from_train, *args):
        print(f"Start learning: {from_train}")
        out = []
        for i in from_train:
            out.append(i * 3)
        return {"sum": sum(out)}

    # Defining the stop criterion with a metric (MSE in this case)
    @acl.as_stop_criterion(
        as_executable=False, metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1
    )
    async def check_mse(*args):
        return 0.01

    # dummy data source
    async def data_source():
        for i in range(10):
            yield i
            await asyncio.sleep(1)

    acl.attach_source(data_source())

    # model callback
    async def on_model_callback(state: IterationState):
        print(f"Publish: {state.sum}")

    acl.on_model_ready(on_model_callback)

    # start learner
    async for _ in acl.start():
        await asyncio.sleep(0)

    acl.stop()

    await acl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_al())
