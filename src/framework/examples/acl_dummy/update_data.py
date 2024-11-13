import numpy as np

def update_labeled_data():
    # Load datasets and queried indices
    X_labeled = np.load("X_labeled.npy")
    y_labeled = np.load("y_labeled.npy")
    X_unlabeled = np.load("X_unlabeled.npy")
    y_unlabeled = np.load("y_unlabeled.npy")
    queried_indices = np.load("queried_indices.npy")

    # Simulate querying by adding selected samples to the labeled set
    X_new = X_unlabeled[queried_indices]
    y_new = y_unlabeled[queried_indices]

    # Update labeled data
    X_labeled = np.concatenate((X_labeled, X_new))
    y_labeled = np.concatenate((y_labeled, y_new))

    # Remove queried samples from unlabeled data
    X_unlabeled = np.delete(X_unlabeled, queried_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, queried_indices, axis=0)

    # Save updated datasets
    np.save("X_labeled.npy", X_labeled)
    np.save("y_labeled.npy", y_labeled)
    np.save("X_unlabeled.npy", X_unlabeled)
    np.save("y_unlabeled.npy", y_unlabeled)

if __name__ == "__main__":
    update_labeled_data()

