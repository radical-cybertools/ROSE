"""Shared criterion function for the parallel learners example.

The stop criterion is defined once for all parallel learners. It checks the most recently updated
model files (whichever suffix is available) and returns the mean MSE across both learners.
"""

import glob
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error


def check_mse(*args, **kwargs):
    model_files = sorted(glob.glob("model_*.pkl"))
    if not model_files:
        return 1.0  # no models yet — return a high metric so training continues

    rng = np.random.default_rng()
    X_val = rng.random((50, 1))
    y_val = 2 * X_val + 1 + rng.normal(0, 0.1, (50, 1))

    scores = []
    for path in model_files:
        with open(path, "rb") as f:
            state = pickle.load(f)
        scores.append(mean_squared_error(y_val, state["model"].predict(X_val)))

    return float(np.mean(scores))
