# active.py
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error


def acl(input_file="train_output.pkl", output_file="acl_output.pkl"):
    # Load model and data
    with open(input_file, "rb") as f:
        model = pickle.load(f)

    with open("sim_output.pkl", "rb") as f:
        (X_labeled, y_labeled), X_unlabeled = pickle.load(f)

    print(
        f"Original shapes - X_labeled: {X_labeled.shape}, "
        f"y_labeled: {y_labeled.shape}, X_unlabeled: {X_unlabeled.shape}"
    )

    # Predict on the unlabeled data to find the most uncertain samples
    y_pred_unlabeled = model.predict(X_unlabeled)

    # Calculate uncertainty - handle multi-dimensional y properly
    if y_labeled.ndim > 1 and y_labeled.shape[1] > 1:
        # For multi-output case, use mean across features
        mean_y = np.mean(y_labeled, axis=0)
        uncertainty = np.mean(np.abs(y_pred_unlabeled - mean_y), axis=1)
    else:
        # For single output case
        uncertainty = np.abs(y_pred_unlabeled.flatten() - np.mean(y_labeled))

    # Select the most uncertain data points (e.g., top 10 most uncertain)
    n_select = min(10, len(X_unlabeled))
    uncertain_indices = uncertainty.argsort()[-n_select:]
    X_selected = X_unlabeled[uncertain_indices]

    # Generate labels for selected data - match the original y structure
    if y_labeled.shape[1] == X_labeled.shape[1]:
        # Multi-output regression case
        y_selected = 2 * X_selected + 1 + np.random.normal(0, 0.1, X_selected.shape)
    else:
        # Single output case - create appropriate shape
        y_selected = np.mean(2 * X_selected + 1, axis=1, keepdims=True) + np.random.normal(
            0, 0.1, (X_selected.shape[0], 1)
        )

    print(f"Selected shapes - X_selected: {X_selected.shape}, y_selected: {y_selected.shape}")

    # Ensure shapes are compatible for stacking
    if y_labeled.shape[1] != y_selected.shape[1]:
        if y_labeled.shape[1] == 1:
            # Convert y_selected to single output
            y_selected = np.mean(y_selected, axis=1, keepdims=True)
        elif y_selected.shape[1] == 1:
            # Convert y_labeled to multi-output (shouldn't happen in your case)
            y_selected = np.repeat(y_selected, y_labeled.shape[1], axis=1)

    print(
        f"Final shapes before stacking - X_labeled: {X_labeled.shape}, "
        f"X_selected: {X_selected.shape}"
    )
    print(
        f"Final shapes before stacking - y_labeled: {y_labeled.shape}, "
        f"y_selected: {y_selected.shape}"
    )

    # Add selected uncertain data to labeled set
    X_labeled = np.vstack([X_labeled, X_selected])
    y_labeled = np.vstack([y_labeled, y_selected])

    # Retrain model with updated labeled data
    model.fit(X_labeled, y_labeled)

    # Evaluate retrained model
    y_pred = model.predict(X_labeled)
    mse = mean_squared_error(y_labeled, y_pred)

    print(f"Active Learning completed. MSE: {mse:.4f}")

    # Save the updated model and the new labeled data
    with open(output_file, "wb") as f:
        pickle.dump(model, f)

    return output_file, mse


if __name__ == "__main__":
    acl()  # Running the active learning task
