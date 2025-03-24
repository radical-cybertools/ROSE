import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Updated file names
TRAIN_IMAGES_FILE = "train_images.npy"
TRAIN_LABELS_FILE = "train_labels.npy"
LABELS_FILE = "labeled_indices.npy"
MODEL_FILE = "model.pt"

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_labeled_data():
    # Load full dataset (images and labels)
    if not os.path.exists(TRAIN_IMAGES_FILE) or not os.path.exists(TRAIN_LABELS_FILE):
        raise FileNotFoundError("❌ Train dataset files not found! Run `simulation.py` first.")

    train_images = np.load(TRAIN_IMAGES_FILE)  # Shape: (N, 32, 32, 3)
    train_labels = np.load(TRAIN_LABELS_FILE)  # Shape: (N,)
    # Debugging: Print shapes before processing
    print(f"🔍 Loaded train images shape: {train_images.shape}")
    print(f"🔍 Loaded train labels shape: {train_labels.shape}")

    # Check if `labeled_indices.npy` exists, else create it
    if not os.path.exists(LABELS_FILE):
        print("⚠️ No labeled_indices.npy found! Initializing random labeled samples.")
        INIT_LABELED = 1000  # Initial labeled samples
        TOTAL_SAMPLES = len(train_images)
        labeled_indices = np.random.choice(TOTAL_SAMPLES, INIT_LABELED, replace=False)
        np.save(LABELS_FILE, labeled_indices)
    else:
        labeled_indices = np.load(LABELS_FILE)

    # Extract labeled samples
    labeled_images = train_images[labeled_indices]  # Shape: (INIT_LABELED, 32, 32, 3)
    labeled_labels = train_labels[labeled_indices]  # Shape: (INIT_LABELED,)

    # Ensure correct tensor shape (N, C, H, W) for PyTorch
    # labeled_images = np.transpose(labeled_images, (0, 3, 1, 2))  # Convert (N, H, W, C) → (N, C, H, W)

   
    # Debugging: Print shapes after fixing
    print(f"✅ Processed labeled images shape: {labeled_images.shape}")
    # Convert to PyTorch tensors
    images = torch.tensor(labeled_images, dtype=torch.float32)  # Shape: (N, 3, 32, 32)
    labels = torch.tensor(labeled_labels, dtype=torch.long)  # Shape: (N,)

    return images, labels

def train_model():
    images, labels = load_labeled_data()

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("🚀 Training Model...")
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ Model saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()

