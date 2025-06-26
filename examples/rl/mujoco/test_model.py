from dm_control import suite
from policy import ReinforcePolicy
import pickle
import numpy as np

def run_test_episode(env, policy):
    time_step = env.reset()
    total_reward = 0.0
    for _ in range(500):
        state = time_step.observation["position"].tolist() + time_step.observation["velocity"].tolist()
        action = policy.select_action(state, deterministic=True)
        time_step = env.step(action)
        total_reward += time_step.reward or 0.0
        if time_step.last():
            break
    return total_reward

if __name__ == "__main__":
    with open("trained_policy.pkl", "rb") as f:
        policy = pickle.load(f)

    env = suite.load(domain_name="cartpole", task_name="swingup")
    reward = run_test_episode(env, policy)
    print(f"REINFORCE policy reward: {reward:.2f}")
