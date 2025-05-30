
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import random
import gym
import os
import sys
from collections import deque, namedtuple
from model import QNetwork

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def update(work_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENV_NAME = "CartPole-v1"
    
    # Config
    MODEL_PATH = os.path.join(work_dir, "dqn_model.pth")
    MEMORY_PATH = os.path.join(work_dir, "replay_memory.pkl")
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 1e-3
    EPOCHS = 100

    # Load memory
    with open(MEMORY_PATH, 'rb') as f:
        memory = pickle.load(f)

    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize model
    model = QNetwork(state_size, action_size, seed=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded existing model.")
    except FileNotFoundError:
        print("No existing model found. Starting fresh.")

    # Training loop
    for epoch in range(EPOCHS):
        batch = random.sample(memory, BATCH_SIZE)
        states = torch.FloatTensor([e.state for e in batch]).to(device)
        actions = torch.LongTensor([[e.action] for e in batch]).to(device)
        rewards = torch.FloatTensor([[e.reward] for e in batch]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
        dones = torch.BoolTensor([[e.done] for e in batch]).to(device)

        q_values = model(states).gather(1, actions)
        with torch.no_grad():
            q_next = model(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + (1 - dones.float()) * GAMMA * q_next

        loss = F.mse_loss(q_values, q_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    update(work_dir)
