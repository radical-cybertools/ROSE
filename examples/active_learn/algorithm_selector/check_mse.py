# check.py
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error


def complicated_function(x):
    return (
        0.3 * np.sin(1.5 * np.pi * x**2)
        + 0.2 * np.cos(2 * np.pi * x**3)
        + 0.5 * np.exp(-0.5 * x)
        + 0.1 * np.tanh(0.2 * (x - 0.5))
        + 0.3 * (x**3)
    )


def check(input_file="train_output.pkl"):
    # Load the model after active learning
    with open(input_file, "rb") as f:
        labeled_data, model = pickle.load(f)

    X_eval = np.random.uniform(low=0.0, high=1.0, size=500)
    y_eval = complicated_function(X_eval)
    X_eval = X_eval.reshape(-1, 1)
    y_pred_eval = model.predict(X_eval)
    mse_eval = mean_squared_error(y_eval, y_pred_eval)
    print(mse_eval)


if __name__ == "__main__":
    check()  # Running the check task
