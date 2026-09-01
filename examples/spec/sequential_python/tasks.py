"""Python task functions for the sequential spec example.

All data passes in-memory — no file I/O needed for python-type tasks.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def simulate(*args, **kwargs):
    rng = np.random.default_rng(seed=42)
    X = rng.random((100, 1))
    y = 2 * X + 1 + rng.normal(0, 0.1, (100, 1))
    X_pool = rng.random((200, 1))
    return {"X": X, "y": y, "X_pool": X_pool}


def train(sim_result, **kwargs):
    model = LinearRegression()
    model.fit(sim_result["X"], sim_result["y"])
    return model


def active_learn(sim_result, model, **kwargs):
    X_pool = sim_result["X_pool"]
    uncertainty = np.abs(model.predict(X_pool) - model.predict(X_pool).mean())
    top_idx = uncertainty.flatten().argsort()[-10:]
    X_new = X_pool[top_idx]
    y_new = 2 * X_new + 1 + np.random.normal(0, 0.1, X_new.shape)
    return {
        "X": np.vstack([sim_result["X"], X_new]),
        "y": np.vstack([sim_result["y"], y_new]),
        "X_pool": np.delete(X_pool, top_idx, axis=0),
    }


def check_mse(*args, **kwargs):
    rng = np.random.default_rng()
    X_val = rng.random((50, 1))
    y_val = 2 * X_val + 1 + rng.normal(0, 0.1, (50, 1))
    # retrieve the model from the most recent training result
    for a in args:
        if hasattr(a, "predict"):
            return float(mean_squared_error(y_val, a.predict(X_val)))
    return 0.05
