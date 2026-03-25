import argparse
import os
import pickle

from policy import ReinforcePolicy

from rose.rl.experience import ExperienceBank

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--memory-file", type=str, default="replay_memory.pkl")
    parser.add_argument("--policy-file", type=str, default="trained_policy.pkl")

    args, unknown = parser.parse_known_args()

    # Construct full file paths
    memory_path = os.path.join(args.data_dir, args.memory_file)
    policy_path = os.path.join(args.data_dir, args.policy_file)

    print(f"Loading replay memory from: {memory_path}")
    memory = ExperienceBank.load(memory_path)

    # Load existing policy if it exists
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy = pickle.load(f)
        print(f"Loaded existing policy from: {policy_path}")
    else:
        policy = ReinforcePolicy(state_dim=5)
        print("Initialized new policy.")

    for epoch in range(200):
        for _ in range(5):
            samples = memory.sample(64)
            policy.update(samples)
        print(f"Epoch {epoch}: updated policy with {len(samples)} samples")

    # Save the trained policy to the specified path
    with open(policy_path, "wb") as f:
        pickle.dump(policy, f)
    print(f"Saved trained policy to {policy_path}")
