import os
import sys
import asyncio

from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ConcurrentExecutionBackend

from rose.metrics import GREATER_THAN_THRESHOLD
from rose.rl.reinforcement_learner import SequentialReinforcementLearner


async def rose_rl():

    engine = await ConcurrentExecutionBackend(
        ThreadPoolExecutor()
        )
    asyncflow = await WorkflowEngine.create(engine)
    rl = SequentialReinforcementLearner(asyncflow)
    code_path = f'{sys.executable} {os.getcwd()}'
    data_path = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_path, exist_ok=True)

    # Define and register the environment task
    @rl.environment_task
    async def environment(*args):
        return f'{code_path}/environment.py {data_path} 0.1 5 experience_bank.pkl'

    # Define and register the policy update task
    @rl.update_task
    async def update(*args):
        return f'{code_path}/update.py {data_path}'

    @rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=200, operator=GREATER_THAN_THRESHOLD)
    async def check_reward(*args):
        return f'{code_path}/check_reward.py {data_path}'


    await rl.learn()
    await rl.shutdown()


if __name__ == "__main__":
    asyncio.run(rose_rl())
