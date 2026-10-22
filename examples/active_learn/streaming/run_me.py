# Bare-bones example of a SequentialActiveLearner driven by a Stream.
#
# Each window of streamed items triggers one simulation -> training ->
# active-learning iteration; the window is the first argument of the
# simulation task. Without a stop criterion, every iteration's model is
# published via on_model_ready.
#
# All tasks are function tasks: the callback state is filled from their dict
# results. With executable tasks, the developer is responsible for passing
# model references (e.g. a checkpoint path) themselves.

import asyncio
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend

from rose import Stream
from rose.al.active_learner import SequentialActiveLearner
from rose.learner import IterationState


async def rose_al():
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    acl = SequentialActiveLearner(asyncflow)

    # Define and register the simulation task
    @acl.simulation_task(as_executable=False)
    async def simulation(window, *args):
        print(f"Start simulation: {window}")
        return {"train_window": window}

    # Define and register the training task
    @acl.training_task(as_executable=False)
    async def training(sim_result, *args):
        train_window = sim_result["train_window"]
        print(f"Start training: {train_window}")
        out = []
        for i in train_window:
            out.append(i * 2)
        return {"trained": out}

    # Define and register the active learning task
    @acl.active_learn_task(as_executable=False)
    async def active_learn(sim_result, train_result, *args):
        from_train = train_result["trained"]
        print(f"Start learning: {from_train}")
        out = []
        for i in from_train:
            out.append(i * 3)
        return {"sum": sum(out)}

    # dummy data source
    async def data_source():
        for i in range(10):
            yield i
            await asyncio.sleep(1)

    stream = Stream(batch_size=2, sources=data_source())

    # model callback
    async def on_model_callback(state: IterationState):
        print(f"Publish: {state.sum}")

    acl.on_model_ready(on_model_callback)

    # start learner
    async for _ in acl.start(stream=stream):
        await asyncio.sleep(0)

    await acl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_al())
