# check_mse.py
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error


def check(input_file="acl_output.pkl"):
    # Load the model after active learning
    with open(input_file, "rb") as f:
        model = pickle.load(f)

    # Load the original simulation data to get the correct feature dimensions
    with open("sim_output.pkl", "rb") as f:
        (X_labeled_orig, y_labeled_orig), _ = pickle.load(f)

    # Get the number of features from the original training data
    n_features = X_labeled_orig.shape[1]
    n_output_features = y_labeled_orig.shape[1] if y_labeled_orig.ndim > 1 else 1

    # Create evaluation data with the correct number of features
    X_eval = np.random.rand(100, n_features)  # Match the feature dimension

    # Generate evaluation labels with the same structure as training data
    if n_output_features == n_features:
        # Multi-output case: each output corresponds to input features
        y_eval = 2 * X_eval + 1 + np.random.normal(0, 0.1, X_eval.shape)
    else:
        # Single output case: aggregate the features somehow
        y_eval = np.mean(2 * X_eval + 1, axis=1, keepdims=True) + np.random.normal(0, 0.1, (100, 1))

    # Evaluate the model on the new data
    y_pred_eval = model.predict(X_eval)

    # Ensure shapes match for MSE calculation
    if y_eval.shape != y_pred_eval.shape:
        # Handle potential shape mismatches
        if y_pred_eval.ndim == 1 and y_eval.ndim == 2:
            y_pred_eval = y_pred_eval.reshape(-1, 1)
        elif y_eval.ndim == 1 and y_pred_eval.ndim == 2:
            y_eval = y_eval.reshape(-1, 1)

    mse_eval = mean_squared_error(y_eval, y_pred_eval)
    # Return the MSE for the framework to use
    print(mse_eval)  # Print the final result


if __name__ == "__main__":
    mse = check()  # Running the check task
