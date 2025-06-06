
import gym
import torch
import pickle
import numpy as np
import sys
import os
import math
from collections import deque, namedtuple
from model import QNetwork

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def episode(memory_file, work_dir=".", epsilon=0.1, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    ENV_NAME = "CartPole-v1"
    MODEL_PATH = os.path.join(work_dir, "dqn_model.pth")
    MEMORY_PATH = os.path.join(work_dir, memory_file)
    MAX_MEMORY_SIZE = int(1e5)

    # ReplayBuffer logic
    def load_memory(path, max_size):
        try:
            with open(path, 'rb') as f:
                memory = pickle.load(f)
            print("Loaded memory.")
        except FileNotFoundError:
            memory = deque(maxlen=max_size)
            print("Initialized new memory.")
        return memory

    def save_memory(memory, path):
        with open(path, 'wb') as f:
            pickle.dump(memory, f)

    # Environment interaction
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = QNetwork(state_size, action_size, seed=0).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded existing model.")
    except FileNotFoundError:
        print("No existing model found. Starting fresh.")
    model.eval()

    memory = load_memory(MEMORY_PATH, MAX_MEMORY_SIZE)

    for epoch in range(epochs):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.random() < epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action  = model(state_tensor).argmax().item()
            else:
                action = env.action_space.sample()
                    
            next_state, reward, done, _, _ = env.step(action)
            memory.append(Experience(state, action, reward, next_state, done))
            state = next_state

    save_memory(memory, MEMORY_PATH)
    print(f"Saved memory with {len(memory)} experiences.")

    env.close()

if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    memory_file = sys.argv[2] if len(sys.argv) > 2 else "replay_memory.pkl"
    episode(memory_file, work_dir)
