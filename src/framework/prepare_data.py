import numpy as np
from sklearn.datasets import load_iris
import random

def prepare_data(initial_size=5):
    data = load_iris()
    X, y = data.data, data.target

    # Save the full dataset for evaluation
    np.save("X.npy", X)
    np.save("y.npy", y)


    labeled_indices = random.sample(range(len(X)), initial_size)
    unlabeled_indices = list(set(range(len(X))) - set(labeled_indices))

    # Save labeled and unlabeled data
    np.save("X_labeled.npy", X[labeled_indices])
    np.save("y_labeled.npy", y[labeled_indices])
    np.save("X_unlabeled.npy", X[unlabeled_indices])
    np.save("y_unlabeled.npy", y[unlabeled_indices])

if __name__ == "__main__":
    prepare_data()

