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
from rose.rl.experience import Experience, ExperienceBank
from rose.uq import UQScorer
from rose.metrics import PREDICTIVE_ENTROPY

def update(work_dir=".", memory_file="experience_bank.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENV_NAME = "CartPole-v1"
    
    # Config
    MODEL_PATH = os.path.join(work_dir, "dqn_model.pth")
    BATCH_SIZE = 64
    GAMMA = 0.99
    LR = 1e-3
    EPOCHS = 100
    ENTROPY_COEF = 0.01  # weight for entropy regularization

    # Load memory bank
    memory_path = os.path.join(work_dir, memory_file)
    memory = ExperienceBank.load(memory_path)

    if len(memory) < BATCH_SIZE:
        print(f"Not enough experiences for training. Need at least {BATCH_SIZE}, got {len(memory)}")
        return
    
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
        # Sample batch from experience bank
        batch = memory.sample(BATCH_SIZE, replace=True)
        
        states = torch.FloatTensor([exp.state for exp in batch]).to(device)
        actions = torch.LongTensor([[exp.action] for exp in batch]).to(device)
        rewards = torch.FloatTensor([[exp.reward] for exp in batch]).to(device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(device)
        dones = torch.BoolTensor([[exp.done] for exp in batch]).to(device)

        q_preds = model(states)
        with torch.no_grad():
            q_next = model(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + (1 - dones.float()) * GAMMA * q_next

        uq = UQScorer(task_type='classification')
        _, uq_metric = uq.select_top_uncertain(q_preds, k=10, metric=PREDICTIVE_ENTROPY)
        
        uq_metric = uq_metric.unsqueeze(1)
        for i, q_values in enumerate(q_preds):
            q_selected = q_values.gather(1, actions)
            loss = F.mse_loss(q_selected, q_targets) - ENTROPY_COEF * uq_metric.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    update(work_dir)
