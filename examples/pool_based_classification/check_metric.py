import torch
import numpy as np
import torch.nn.functional as F
from training import CNNModel  # Import your CNN model from training.py
import os

# Updated file names
MODEL_FILE = "model.pt"
TEST_IMAGES_FILE = "test_images.npy"
TEST_LABELS_FILE = "test_labels.npy"
LABELED_FILE = "labeled_indices.npy"
METRIC_LOG = "metrics_log.txt"  # Log file to store accuracy per iteration

def evaluate_model():
    """Loads the trained model and evaluates test accuracy."""
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Model file not found! Run `training.py` first.")

    # Load test data
    test_images = np.load(TEST_IMAGES_FILE)  # Shape: (N, 32, 32, 3)
    test_labels = np.load(TEST_LABELS_FILE)

    # Convert to PyTorch tensors
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_images = np.transpose(test_images, (0, 2, 3, 1))
    # Create DataLoader
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    # Compute test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    # Log accuracy over iterations
    print(accuracy)

if __name__ == "__main__":
    evaluate_model()

