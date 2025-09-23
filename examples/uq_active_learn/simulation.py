# simulation.py
import logging
import json
import argparse
import numpy as np
from pathlib import Path

def simulate(home_dir, samples_file, pool_file, train_batch, learner_name):


    try:
        from torchvision import datasets, transforms
        data_dir = Path(home_dir, 'mnist_data')
        logging.basicConfig(level=logging.INFO)
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root=data_dir,
                                    train=True,
                                    transform=transform,
                                    download=False  # Do NOT download again
                                )
        indices = np.arange(len(mnist_train))
    except:    
        # In case of any error, create dummy indices      
        indices = np.arange(1000)

    np.random.seed(42)
    np.random.shuffle(indices)

    X_labeled_idx = indices[:train_batch]
    X_pool_idx = indices[train_batch:]

    with open(Path(home_dir, learner_name + samples_file), 'w') as f:
        json.dump(X_labeled_idx.tolist(), f)
    with open(Path(home_dir, learner_name + pool_file), 'w') as f:
        json.dump(X_pool_idx.tolist(), f)

    logging.info(f"Selected {train_batch} initial samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--train_batch', type=int, help='Number of labels used for initial training')
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    parser.add_argument('--home_dir', type=str, help='Home directory for the project')
    args = parser.parse_args()

    simulate(args.home_dir, "_samples.json", "_pool.json", train_batch=args.train_batch, learner_name=args.learner_name)
