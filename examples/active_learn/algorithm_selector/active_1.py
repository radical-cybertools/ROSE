# acl.py
import pickle

import numpy as np


def complicated_function(x):
    return (
        0.3 * np.sin(1.5 * np.pi * x**2)
        + 0.2 * np.cos(2 * np.pi * x**3)
        + 0.5 * np.exp(-0.5 * x)
        + 0.1 * np.tanh(0.2 * (x - 0.5))
        + 0.3 * (x**3)
    )


def acl(input_file="train_output.pkl", output_file="acl_output.pkl"):
    with open(input_file, "rb") as f:
        labeled_data, model = pickle.load(f)

    X = np.random.uniform(low=0.0, high=1.0, size=500)
    y = complicated_function(X)
    X_input = X.reshape(-1, 1)
    y_pred = model.predict(X_input)
    uncertainty = np.abs(y_pred - y)

    uncertain_indices = uncertainty.argsort()[-100:]  # Top 100 most uncertain samples
    X_selected = X[uncertain_indices]

    with open(output_file, "wb") as f:
        pickle.dump((labeled_data, X_selected), f)

    return output_file


if __name__ == "__main__":
    acl()  # Running the active learning task
