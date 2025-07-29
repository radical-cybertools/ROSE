# active_learn.py
import json
from pathlib import Path
import numpy as np
import argparse
from rose.uq import UQMetrics

QUERY_SIZE = 10
HOME_PATH = '/anvil/scratch/x-mgoliyad1/ROSE/examples/uq_active_learn'

def active_learn(pool_file, samples_file, uq_metric, task_type, prediction_dir, learner_name):

    
    with open(Path(HOME_PATH, learner_name + pool_file), 'r') as f:
        pool_idx = json.load(f)
    with open(Path(HOME_PATH, learner_name + samples_file), 'r') as f:
        labeled_idx = json.load(f)

    all_preds = []
    predict_dir = Path(HOME_PATH, prediction_dir)

    # All predictions for the current pipeline are stored in a single directory. UQMetrics uses all of them to calculate uncertainty.
    for file in predict_dir.iterdir():
        if file.is_file():
            if file.suffix == '.npy':
                preds = np.load(file, allow_pickle=True)
                all_preds.append(np.vstack(preds))
    all_preds = np.array(all_preds)

    uq = UQMetrics(task_type=task_type)
    top_idx_local, uq_metric = uq.select_top_uncertain(all_preds, n_instances=QUERY_SIZE, strategy=uq_metric)

    # Map best_idx_local to actual pool indices
    pool_idx = np.array(pool_idx)
    selected_idx = pool_idx[top_idx_local]

    # Add to labeled set
    labeled_idx = np.concatenate([labeled_idx, selected_idx])

    # Remove from pool
    mask = np.ones(len(pool_idx), dtype=bool)
    mask[top_idx_local] = False
    pool_idx = pool_idx[mask]

    with open(Path(HOME_PATH, learner_name + samples_file), 'w') as f:
        json.dump(labeled_idx.tolist(), f)
    with open(Path(HOME_PATH, learner_name + pool_file), 'w') as f:
        json.dump(pool_idx.tolist(), f)
    print(np.mean(uq_metric))

if __name__ == "__main__":
    #uq_metric
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--uq_metric', type=str, help='Type of UQ metrics to use')
    parser.add_argument('--task_type', type=str, help='Type of training task')
    parser.add_argument('--prediction_dir', type=str, help='Directory for predictions')
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    args = parser.parse_args()

    active_learn("_pool.json", "_samples.json", uq_metric=args.uq_metric, task_type=args.task_type, prediction_dir=args.prediction_dir, learner_name=args.learner_name)
   