#!/usr/bin/env python3
"""Simulation task: generate labeled + unlabeled data.

Usage: python sim.py --label <suffix>
Writes: sim_<suffix>.pkl
"""

import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
args = parser.parse_args()

rng = np.random.default_rng()
X = rng.random((100, 1))
y = 2 * X + 1 + rng.normal(0, 0.1, (100, 1))
X_pool = rng.random((200, 1))

output = f"sim_{args.label}.pkl"
with open(output, "wb") as f:
    pickle.dump({"X": X, "y": y, "X_pool": X_pool}, f)

print(output)
