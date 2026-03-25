import argparse
import pickle

import numpy as np


def sim(n_labeled=100, n_unlabeled=100, n_features=1, output_file="sim_output.pkl"):
    # Generate initial labeled data
    X = np.random.rand(n_labeled, n_features)
    y = (
        2 * X + 1 + np.random.normal(0, 0.1, (n_labeled, n_features))
    )  # Linear relationship with noise
    labeled_data = (X, y)

    # Generate unlabeled data
    X_unlabeled = np.random.rand(n_unlabeled, n_features)
    unlabeled_data = X_unlabeled

    # Save both to file
    with open(output_file, "wb") as f:
        pickle.dump((labeled_data, unlabeled_data), f)

    print(f"Simulation completed. Data saved to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate labeled and unlabeled data for active learning."
    )
    parser.add_argument(
        "--n_labeled",
        type=int,
        default=100,
        help="Number of labeled samples (default: 100)",
    )
    parser.add_argument(
        "--n_unlabeled",
        type=int,
        default=100,
        help="Number of unlabeled samples (default: 100)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=1,
        help="Number of features per sample (default: 1)",
    )
    parser.add_argument(
        "--output_file", type=str, default="sim_output.pkl", help="Output file name"
    )

    args = parser.parse_args()

    sim(
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        n_features=args.n_features,
        output_file=args.output_file,
    )
