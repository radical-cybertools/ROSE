import os
import pickle
import torch
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import Trainer

# Configuration
BOARD_SIZE = 9
MODEL_PATH = 'models/current_model.pth'
DATA_PATH = 'data/selfplay_data.pkl'
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Load self-play data
with open(DATA_PATH, 'rb') as f:
    game_data = pickle.load(f)

# Initialize network
network = AlphaZeroNet(board_size=BOARD_SIZE)
network.load_state_dict(torch.load(MODEL_PATH))

# Train network
trainer = Trainer(network, learning_rate=LEARNING_RATE)
trainer.train(game_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save updated model
torch.save(network.state_dict(), MODEL_PATH)
