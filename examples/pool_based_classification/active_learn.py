import torch
import torch.nn.functional as F
import numpy as np
import os

# Updated file names
MODEL_FILE = "model.pt"
TRAIN_IMAGES_FILE = "train_images.npy"
UNLABELED_FILE = "unlabeled_indices.npy"
OUTPUT_SELECTED = "selected_samples.npy"
QUERY_SIZE = 200  # Number of samples to query

class CNNModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_unlabeled_data():
    # Load full dataset images
    if not os.path.exists(TRAIN_IMAGES_FILE):
        raise FileNotFoundError("Train dataset files not found! Run `simulation.py` first.")

    train_images = np.load(TRAIN_IMAGES_FILE)  # Shape: (50000, 3, 32, 32)

    # Check if `unlabeled_indices.npy` exists
    if not os.path.exists(UNLABELED_FILE):
        raise FileNotFoundError("Missing 'unlabeled_indices.npy'. Run `simulation.py` or `active_learn.py` first.")

    unlabeled_indices = np.load(UNLABELED_FILE)  # Indices of unlabeled samples

    # Extract unlabeled images
    unlabeled_images = train_images[unlabeled_indices]  # Should be (num_unlabeled, 3, 32, 32)

    # Debugging: Print shapes before fixing
    print(f"🔍 Loaded unlabeled images shape: {unlabeled_images.shape}")

    # Ensure correct shape (N, C, H, W)
    if unlabeled_images.shape[1] != 3:  # Check if channels are in the right position
        unlabeled_images = np.transpose(unlabeled_images, (0, 3, 1, 2))  # Convert (N, H, W, C) → (N, C, H, W)

    # Debugging: Print final shape
    print(f"✅ Fixed unlabeled images shape: {unlabeled_images.shape}")

    # Convert to PyTorch tensors
    images = torch.tensor(unlabeled_images, dtype=torch.float32)

    return images, unlabeled_indices

def select_samples():
    # Load the trained model
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load unlabeled data
    images, unlabeled_indices = load_unlabeled_data()
    images = images.to(device)

    # Get uncertainty scores
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)  # Maximum class probability
        uncertainty = 1 - confidence  # Least confidence method

    # Rank samples by uncertainty (highest uncertainty first)
    sorted_indices = sorted(range(len(uncertainty)), key=lambda i: uncertainty[i], reverse=True)
    
    # Select top `QUERY_SIZE` most uncertain samples
    selected_indices = [unlabeled_indices[i] for i in sorted_indices[:QUERY_SIZE]]

    # Save selected indices
    np.save(OUTPUT_SELECTED, selected_indices)
    print(f"Selected {QUERY_SIZE} most uncertain samples saved to {OUTPUT_SELECTED}")

if __name__ == "__main__":
    select_samples()

