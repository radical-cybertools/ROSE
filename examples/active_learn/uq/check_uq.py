import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rose.uq import UQScorer, register_uq


@register_uq("custom_uq")
def confidence_score(self, mc_preds):
    """
    Custom classification metric: 1 - max predicted probability.
    Lower max prob = higher uncertainty.
    """
    mc_preds, _ = self._validate_inputs(mc_preds)
    mean_probs = np.mean(mc_preds, axis=0)  # [n_instances, n_classes]
    max_prob = np.max(mean_probs, axis=1)
    return 1.0 - max_prob


def check_uq(home_dir, predict_dir, learner_name, query_size, uq_metric_name, task_type):
    prediction_dir = Path(home_dir, predict_dir)

    all_preds = []
    all_files = []

    for file in prediction_dir.iterdir():
        if file.is_file() and file.suffix == ".npy":
            preds = np.load(file, allow_pickle=True)
            # If preds is already an array, just append it
            if isinstance(preds, np.ndarray):
                all_preds.append(preds)
            else:  # If it's a list of arrays, stack first
                all_preds.append(np.vstack(preds))
            all_files.append(file)

    # Combine all predictions safely if they share the same shape
    try:
        all_preds = np.concatenate(all_preds, axis=0)
    except ValueError:
        all_preds = np.array(all_preds, dtype=object)  # fallback to ragged array

    if len(all_preds) == 0:
        print(sys.float_info.max)
    else:
        uq = UQScorer(task_type=task_type)
        top_idx_local, uq_metric = uq.select_top_uncertain(
            all_preds, k=query_size, metric=uq_metric_name
        )

        with open(Path(home_dir, learner_name + "_uq_selection.json"), "w") as f:
            json.dump(top_idx_local.tolist(), f)
        print(np.mean(uq_metric))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument("--predict_dir", type=str, help="Directory for predictions")
    parser.add_argument("--query_size", type=int, help="Size of the query for uncertainty sampling")
    parser.add_argument(
        "--uq_metric_name",
        type=str,
        help="Name of the uncertainty quantification metric",
    )
    parser.add_argument(
        "--task_type", type=str, help="Type of the task for uncertainty quantification"
    )
    parser.add_argument("--learner_name", type=str, help="Name of the learner")
    parser.add_argument("--home_dir", type=str, help="Home directory for the project")
    args = parser.parse_args()

    check_uq(
        args.home_dir,
        args.predict_dir,
        args.learner_name,
        query_size=args.query_size,
        uq_metric_name=args.uq_metric_name,
        task_type=args.task_type,
    )
