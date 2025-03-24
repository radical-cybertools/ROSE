import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

DATASET = "CIFAR10"
DATA_DIR = "."

# File paths
TRAIN_IMAGES_FILE = "train_images.npy"
TRAIN_LABELS_FILE = "train_labels.npy"
LABELED_FILE = "labeled_indices.npy"
UNLABELED_FILE = "unlabeled_indices.npy"

TEST_IMAGES_FILE = "test_images.npy"
TEST_LABELS_FILE = "test_labels.npy"

INIT_LABELED = 1000  # Number of initially labeled samples

def load_data():
    """Loads CIFAR-10 or MNIST datasets and returns train and test sets."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_class = torchvision.datasets.CIFAR10 if DATASET == "CIFAR10" else torchvision.datasets.MNIST

    train_set = dataset_class(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = dataset_class(root=DATA_DIR, train=False, download=True, transform=transform)

    return train_set, test_set  # ✅ Ensure both train and test sets are returned


def save_data():
    train_set, test_set = load_data()

    # Convert dataset to NumPy arrays
    # train_images = np.array([img.numpy() for img, _ in train_set])  # Convert images to NumPy
    # train_labels = np.array([label for _, label in train_set])  # Convert labels to NumPy

    # Convert train dataset to NumPy arrays
    train_images = np.array([img.numpy() for img, _ in train_set])
    train_labels = np.array([label for _, label in train_set])

    # Convert test dataset to NumPy arrays
    test_images = np.array([img.numpy() for img, _ in test_set])
    test_labels = np.array([label for _, label in test_set])
    
    test_images = np.transpose(test_images, (0, 3, 1, 2))  # Convert (N, H, W, C)

    print(train_images.shape, test_images.shape)

    # Save images and labels
    np.save(TRAIN_IMAGES_FILE, train_images)
    np.save(TRAIN_LABELS_FILE, train_labels)
    np.save(TEST_IMAGES_FILE, test_images)
    np.save(TEST_LABELS_FILE, test_labels)

    print(f"✅ Test images saved to {TEST_IMAGES_FILE}")
    print(f"✅ Test labels saved to {TEST_LABELS_FILE}")


    # Create labeled and unlabeled indices
    total_samples = len(train_set)
    labeled_indices = np.random.choice(total_samples, INIT_LABELED, replace=False)
    unlabeled_indices = np.setdiff1d(np.arange(total_samples), labeled_indices)

    # Save labeled and unlabeled indices
    np.save(LABELED_FILE, labeled_indices)
    np.save(UNLABELED_FILE, unlabeled_indices)

    print(f"✅ Train images saved to {TRAIN_IMAGES_FILE}")
    print(f"✅ Train labels saved to {TRAIN_LABELS_FILE}")
    print(f"✅ Initialized {INIT_LABELED} labeled samples.")
    print(f"✅ Initialized {len(unlabeled_indices)} unlabeled samples.")

if __name__ == "__main__":
    save_data()

