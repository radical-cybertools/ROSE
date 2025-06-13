
import gym
import torch
import pickle
import numpy as np
import sys
import os
import math
from collections import deque, namedtuple
from model import QNetwork
from rose.experience import Experience, ExperienceBank, create_experience

def episode(work_dir=".", epsilon=0.1, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    ENV_NAME = "CartPole-v1"
    MODEL_PATH = os.path.join(work_dir, "dqn_model.pth")
    MAX_MEMORY_SIZE = int(1e5)

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

    memory = ExperienceBank(max_size=MAX_MEMORY_SIZE)

    for epoch in range(epochs):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if np.random.random() < epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()
            else:
                action = env.action_space.sample()
                    
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Create and add experience
            experience = create_experience(state, action, reward, next_state, done)
            memory.add(experience)
            
            state = next_state
        
        print(f"Epoch {epoch + 1}/{epochs} - Episode reward: {episode_reward}")

    # Save memory
    memory.save()
    print(f"Saved memory with {len(memory)} experiences.")

    env.close()

if __name__ == "__main__":
    episode()
