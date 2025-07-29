# simulation.py
import logging
import json
import argparse
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

HOME_PATH = '/anvil/scratch/x-mgoliyad1/ROSE/examples/uq_active_learn'

def simulate(samples_file, pool_file, n_labeled, learner_name):
    logging.basicConfig(level=logging.INFO)
    transform = transforms.Compose([transforms.ToTensor()])

    data_dir = Path(HOME_PATH, 'data')
    try:
        mnist_train = datasets.MNIST(root=data_dir,
                                    train=True,
                                    transform=transform,
                                    download=False  # Do NOT download again
                                )
    except:          
        logging.info("Loading MNIST dataset...")      
        mnist_train = datasets.MNIST(root=data_dir, 
                                     train=True, 
                                     download=True, 
                                     transform=transform)

    # Split small labeled set + large pool
    indices = np.arange(len(mnist_train))
    np.random.seed(42)
    np.random.shuffle(indices)

    X_labeled_idx = indices[:n_labeled]
    X_pool_idx = indices[n_labeled:]

    with open(Path(HOME_PATH, learner_name + samples_file), 'w') as f:
        json.dump(X_labeled_idx.tolist(), f)
    with open(Path(HOME_PATH, learner_name + pool_file), 'w') as f:
        json.dump(X_pool_idx.tolist(), f)

    logging.info(f"Selected {n_labeled} initial samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--n_labeled', type=int, help='Number of labels used for initial training')
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    args = parser.parse_args()

    simulate("_samples.json", "_pool.json", n_labeled=args.n_labeled, learner_name=args.learner_name)

