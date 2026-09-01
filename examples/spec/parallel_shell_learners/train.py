#!/usr/bin/env python3
"""Training task: fit the chosen model on sim_<label>.pkl.

Usage: python train.py --label <suffix> --model linear|ridge
Reads:  sim_<suffix>.pkl
Writes: model_<suffix>.pkl
"""
import argparse
import pickle

from sklearn.linear_model import LinearRegression, Ridge

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--model", choices=["linear", "ridge"], required=True)
args = parser.parse_args()

with open(f"sim_{args.label}.pkl", "rb") as f:
    data = pickle.load(f)

model = LinearRegression() if args.model == "linear" else Ridge(alpha=1.0)
model.fit(data["X"], data["y"])

output = f"model_{args.label}.pkl"
with open(output, "wb") as f:
    pickle.dump({"model": model, "data": data}, f)

print(output)
