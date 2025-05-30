import os
import sys
import time

verbose  = os.environ.get('RADICAL_PILOT_VERBOSE', 'REPORT')
os.environ['RADICAL_PILOT_VERBOSE'] = verbose

import radical.pilot as rp
import radical.utils as ru

from rose.learner import ReinforcementLearner
from rose.engine import Task, ResourceEngine
from rose.metrics import GREATER_THAN_THRESHOLD

engine = ResourceEngine({'runtime': 30,
                         'resource': 'local.localhost'})
rl = ReinforcementLearner(engine)
code_path = f'{sys.executable} {os.getcwd()}'
data_path = os.path.join(os.getcwd(), 'data')
os.makedirs(data_path, exist_ok=True)

# Define and register the environment task
@rl.environment_task
def environment(shard, *args):
    return Task(executable=f'{code_path}/environment.py {shard} {data_path}', arguments=args)

# Define and register the policy update task
@rl.update_task
def update(*args):
    return Task(executable=f'{code_path}/update.py {data_path}', arguments=args)

@rl.utility_task
def merge(*args):
    return Task(executable=f'{code_path}/merge_memory.py {data_path}', arguments=args)

@rl.as_stop_criterion(metric_name='MODEL_REWARD', threshold=100, operator=GREATER_THAN_THRESHOLD)
def check_reward(*args):
    return Task(executable=f'{code_path}/check_reward.py {data_path}', arguments=args)

# Custom teaching loop with reinforcement Learning
rewards = []
def learn():
    for iter in range(20):
        envs = []
        for i in range(5):
            env = environment(i)
            envs.append(env)
        [env.result() for env in envs]
        mrg = merge()
        upd = update(mrg)
        should_stop, reward = check_reward(upd);
        rewards.append(reward)
        if should_stop:
            break
# invoke the custom/user-defined teach() method
learn()
engine.shutdown()
