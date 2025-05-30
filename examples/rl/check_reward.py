import gym
import torch
import numpy as np
import sys
import os
from model import QNetwork

def reward(work_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    ENV_NAME = "CartPole-v1"
    MODEL_PATH = os.path.join(work_dir, "dqn_model.pth")
    EPISODES = 10
    RENDER = False

    # Load environment and model
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = QNetwork(state_size, action_size, seed=0).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    episode_rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if RENDER:
                env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    env.close()
    mean_reward = np.mean(episode_rewards)
    print(f"{mean_reward:.2f}")

if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    reward(work_dir)
