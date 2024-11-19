# sim.py
import numpy as np
import pickle

def sim(output_file='sim_output.pkl'):
    # Generate initial labeled data for simulation
    X = np.random.rand(100, 1)  # 100 samples, 1 feature
    y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))  # Linear relationship with noise

    # Save the initial labeled data
    labeled_data = (X, y)

    # Generate additional unlabeled data for active learning
    X_unlabeled = np.random.rand(100, 1)  # 100 additional unlabeled samples
    unlabeled_data = X_unlabeled

    with open(output_file, 'wb') as f:
        pickle.dump((labeled_data, unlabeled_data), f)

    print(f"Simulation completed. Data saved to {output_file}")
    return output_file


if __name__ == "__main__":
    sim()  # Running the simulation task
