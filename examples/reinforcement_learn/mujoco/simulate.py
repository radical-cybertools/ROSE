import argparse
import os
import pickle

from dm_control import suite
from policy import ReinforcePolicy

from rose.rl.experience import Experience, ExperienceBank


def run_cartpole_episode(env, policy, max_steps=200):
    time_step = env.reset()
    experiences = []

    for _ in range(max_steps):
        state = (
            time_step.observation["position"].tolist() + time_step.observation["velocity"].tolist()
        )
        action = policy.select_action(state, deterministic=False)
        time_step = env.step(action)

        next_state = (
            time_step.observation["position"].tolist() + time_step.observation["velocity"].tolist()
        )
        reward = time_step.reward or 0.0
        done = time_step.last()

        exp = Experience(state, action, reward, next_state, done)
        experiences.append(exp)

        if done:
            break

    return experiences


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simulate RL policy and collect experience data")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--policy-file", type=str, default="trained_reinforce_policy.pkl")
    parser.add_argument("--memory-file", type=str, default="replay_memory.pkl")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--memory-size", type=int, default=1000)

    args, unknown = parser.parse_known_args()

    # Construct full file paths
    policy_path = os.path.join(args.data_dir, args.policy_file)
    memory_path = os.path.join(args.data_dir, args.memory_file)

    env = suite.load(domain_name="cartpole", task_name="swingup")

    # Load policy
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy = pickle.load(f)
        print(f"Loaded existing policy from: {policy_path}")
    else:
        policy = ReinforcePolicy(state_dim=5)
        print(f"No policy found at {policy_path}, using random weights.")

    memory = ExperienceBank(max_size=args.memory_size)

    print(f"Running {args.episodes} episodes with max {args.max_steps} steps each")
    for episode in range(args.episodes):
        episode_memory = run_cartpole_episode(env, policy, max_steps=args.max_steps)
        memory.add_batch(episode_memory)
        print(f"Episode {episode}: collected {len(episode_memory)} steps")

    print(f"Saving experience bank to: {memory_path}")
    memory.save(bank_file=memory_path)
