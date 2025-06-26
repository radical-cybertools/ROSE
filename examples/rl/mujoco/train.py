import numpy as np
import os
from policy import ReinforcePolicy
from rose.rl.experience import Experience, ExperienceBank
import pickle

if __name__ == "__main__":
    memory = ExperienceBank.load("replay_memory.pkl")
    policy_path = "trained_policy.pkl"

    # Load existing policy if it exists
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy = pickle.load(f)
        print("Loaded existing policy.")
    else:
        policy = ReinforcePolicy(state_dim=5)
        print("Initialized new policy.")
    
    for epoch in range(20):
        for i in range(5):
            samples = memory.sample(64)
            policy.update(samples)
        print(f"Epoch {epoch}: updated policy with {len(samples)} samples")

    with open("trained_policy.pkl", "wb") as f:
        pickle.dump(policy, f)