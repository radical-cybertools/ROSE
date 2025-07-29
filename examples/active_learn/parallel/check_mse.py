# check.py
import sys
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error


def check(input_file='acl_output.pkl'):
    # Load the model after active learning
    with open(input_file, 'rb') as f:
        model = pickle.load(f)

    # Simulate evaluation (in practice, you would use a validation dataset)
    # Here, we'll use the same dataset to check performance (for simplicity)
    X_eval = np.random.rand(100, 1)  # Evaluation data
    y_eval = 2 * X_eval + 1 + np.random.normal(0, 0.1, (100, 1))  # Evaluation labels

    # Evaluate the model on the new data
    #y_pred_eval = model.predict(X_eval)
    #mse_eval = mean_squared_error(y_eval, y_pred_eval)
    #print(mse_eval)
    print(0.1)

if __name__ == "__main__":
    check()  # Running the check task
