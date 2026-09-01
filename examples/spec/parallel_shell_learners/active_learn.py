#!/usr/bin/env python3
"""Active learning task: select uncertain samples and extend the dataset.

Usage: python active_learn.py --label <suffix>
Reads:  model_<suffix>.pkl
Writes: sim_<suffix>.pkl (updated), model_<suffix>.pkl (updated)
"""

import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
args = parser.parse_args()

with open(f"model_{args.label}.pkl", "rb") as f:
    state = pickle.load(f)

model = state["model"]
data = state["data"]
X, y, X_pool = data["X"], data["y"], data["X_pool"]

uncertainty = np.abs(model.predict(X_pool) - model.predict(X_pool).mean()).flatten()
top_idx = uncertainty.argsort()[-10:]
X_new = X_pool[top_idx]
y_new = 2 * X_new + 1 + np.random.normal(0, 0.1, X_new.shape)

X_updated = np.vstack([X, X_new])
y_updated = np.vstack([y, y_new])
X_pool_upd = np.delete(X_pool, top_idx, axis=0)

model.fit(X_updated, y_updated)

updated_data = {"X": X_updated, "y": y_updated, "X_pool": X_pool_upd}
with open(f"sim_{args.label}.pkl", "wb") as f:
    pickle.dump(updated_data, f)
with open(f"model_{args.label}.pkl", "wb") as f:
    pickle.dump({"model": model, "data": updated_data}, f)

print(f"model_{args.label}.pkl")
