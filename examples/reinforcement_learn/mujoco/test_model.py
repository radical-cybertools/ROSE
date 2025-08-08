import os
import argparse
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RL policy performance')
    parser.add_argument('--data-dir', type=str, default='.', 
                       help='Directory to load policy file from (default: current directory)')
    parser.add_argument('--policy-file', type=str, default='trained_policy.pkl',
                       help='Name of the policy file to load (default: trained_policy.pkl)')
    
    args, unknown = parser.parse_known_args()

    
    # Construct full file path
    policy_path = os.path.join(args.data_dir, args.policy_file)
    
    # Load policy
    try:
        with open(policy_path, "rb") as f:
            policy = pickle.load(f)
    except FileNotFoundError: 
        exit(1)

    env = suite.load(domain_name="cartpole", task_name="swingup")
    reward = []
    for _ in range(10):
        reward.append(run_test_episode(env, policy))
    mean_reward = np.mean(reward)
    print(f"{mean_reward:.2f}")
