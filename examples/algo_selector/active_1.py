# acl.py
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

def acl(input_file='train_output.pkl', output_file='acl_output.pkl'):
    # Load model and data
    with open(input_file, 'rb') as f:
        model = pickle.load(f)
    
    with open('sim_output.pkl', 'rb') as f:
        (X_labeled, y_labeled), X_unlabeled = pickle.load(f)

    # Predict on the unlabeled data to find the most uncertain samples
    y_pred_unlabeled = model.predict(X_unlabeled)
    uncertainty = np.abs(y_pred_unlabeled - np.mean(y_labeled))  # Example of uncertainty: deviation from mean label

    # Select the most uncertain data points (e.g., top 10% most uncertain)
    uncertain_indices = uncertainty.argsort()[-10:]  # Top 10 most uncertain samples
    X_selected = X_unlabeled[uncertain_indices]
    y_selected = 2 * X_selected + 1 + np.random.normal(0, 0.1, X_selected.shape)  # Simulate labels for selected data

    # Ensure X_selected is 2D (flatten if necessary)
    if X_selected.ndim == 3:
        X_selected = X_selected.reshape(X_selected.shape[0], -1)  # Flatten to 2D if needed

    # Similarly for y_selected if necessary
    if y_selected.ndim == 3:
        y_selected = y_selected.reshape(y_selected.shape[0], -1)

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
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)

    return output_file, mse

if __name__ == "__main__":
    acl()  # Running the active learning task
    time.sleep(30)
