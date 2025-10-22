# active_learn.py
import json
from pathlib import Path
import numpy as np
import argparse


def active_learn(home_dir, pool_suffix, samples_suffix, uq_suffix, learner_name):

    pool_file = Path(home_dir, learner_name + pool_suffix)
    samples_file = Path(home_dir, learner_name + samples_suffix)
    uq_file = Path(home_dir, learner_name + uq_suffix)

    if not pool_file.is_file():
        print(f"Pool file {pool_file} does not exist.")
        return
    if not uq_file.is_file():
        print(f"UQ file {uq_file} does not exist.")
        return
    if not samples_file.is_file():
        print(f"Samples file {samples_file} does not exist.")
        return
    
    with open(pool_file, 'r') as f:
        pool_idx = json.load(f)
    with open(samples_file, 'r') as f:
        labeled_idx = json.load(f)
    with open(uq_file, 'r') as f:
        top_idx = json.load(f)

    # Map best_idx_local to actual pool indices
    pool_idx = np.array(pool_idx)
    selected_idx = pool_idx[top_idx]

    # Add to labeled set
    labeled_idx = np.concatenate([labeled_idx, selected_idx])

    # Remove from pool
    mask = np.ones(len(pool_idx), dtype=bool)
    mask[top_idx] = False
    pool_idx = pool_idx[mask]

    with open(samples_file, 'w') as f:
        json.dump(labeled_idx.tolist(), f)
    with open(pool_file, 'w') as f:
        json.dump(pool_idx.tolist(), f)
    print('Active learner picked {} indices for next iteration.'.format(len(labeled_idx)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    parser.add_argument('--home_dir', type=str, help='Home directory for the project')
    args = parser.parse_args()

    active_learn(args.home_dir, "_pool.json", "_samples.json", "_uq_selection.json", learner_name=args.learner_name)
