
import gym
import torch
import pickle
import numpy as np
from collections import deque, namedtuple
from model import QNetwork

def episode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    ENV_NAME = "CartPole-v1"
    MODEL_PATH = "dqn_model.pth"
    MEMORY_PATH = "replay_memory.pkl"
    MAX_MEMORY_SIZE = int(1e5)
    EPISODES = 10

    # ReplayBuffer logic
    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
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

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, _ = env.step(action)
            memory.append(Experience(state, action, reward, next_state, done))
            state = next_state

    save_memory(memory, MEMORY_PATH)
    print(f"Saved memory with {len(memory)} experiences.")

    env.close()

if __name__ == "__main__":
    episode()
