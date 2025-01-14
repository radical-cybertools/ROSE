# sim.py
import numpy as np
import pickle
import os

def complicated_function(x):
    return (
        0.3 * np.sin(1.5 * np.pi * x**2) +
        0.2 * np.cos(2 * np.pi * x**3) +
        0.5 * np.exp(-0.5 * x) +
        0.1 * np.tanh(0.2 * (x - 0.5)) +
        0.3 * (x**3)
        )

def sim(input_file='acl_output.pkl', output_file='sim_output.pkl'):
    labeled_data = None
    unlabeled_data = np.linspace(0, 1, 100)
    if os.path.isfile(input_file):
        with open(input_file, 'rb') as f:
            (labeled_data, unlabeled_data) = pickle.load(f)
    y = complicated_function(unlabeled_data)

    if labeled_data is not None:
        x1, y1 = labeled_data
        print("x1 = ", x1)
        print("unlabeled_data = ", unlabeled_data)
        x = np.concatenate([x1, unlabeled_data])
        y = np.concatenate([y1, y])
        labeled_data = (x, y)
    else:
        labeled_data = (unlabeled_data, y)
        
    print("labeled_data = ", labeled_data)
    with open(output_file, 'wb') as f:
        pickle.dump(labeled_data, f)

    print(f"Simulation completed. Data saved to {output_file}")
    return output_file


if __name__ == "__main__":
    sim()  # Running the simulation task
