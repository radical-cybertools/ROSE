import numpy as np
import os
from dm_control import suite
from policy import ReinforcePolicy
from rose.rl.experience import Experience, ExperienceBank
import pickle

def run_cartpole_episode(env, policy_fn, max_steps=200):
    time_step = env.reset()
    experiences = []

    for _ in range(max_steps):
        state = time_step.observation["position"].tolist() + time_step.observation["velocity"].tolist()
        action = policy.select_action(state, deterministic=False)
        time_step = env.step(action)

        next_state = time_step.observation["position"].tolist() + time_step.observation["velocity"].tolist()
        reward = time_step.reward or 0.0
        done = time_step.last()

        exp = Experience(state, action, reward, next_state, done)
        experiences.append(exp)

        if done:
            break

    return experiences

if __name__ == "__main__":
    env = suite.load(domain_name="cartpole", task_name="swingup")

    policy_path = "trained_reinforce_policy.pkl"
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy = pickle.load(f)
        print("Loaded existing policy for simulation.")
    else:
        policy = ReinforcePolicy(state_dim=5)
        print("No policy found, using random weights.")

    memory = ExperienceBank(max_size=1000)

    for episode in range(20):
        episode_memory = run_cartpole_episode(env, policy)
        memory.add_batch(episode_memory)
        print(f"Collected {len(episode_memory)} steps in episode {episode}")

    memory.save(bank_file="replay_memory.pkl")