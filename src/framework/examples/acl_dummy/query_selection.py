import numpy as np
import joblib

def select_queries(batch_size=5):
    # Load model and unlabeled data
    model = joblib.load("model.joblib")
    X_unlabeled = np.load("X_unlabeled.npy")

    # Predict probabilities for the unlabeled data
    probabilities = model.predict_proba(X_unlabeled)
    confidence = np.max(probabilities, axis=1)

    # Select indices of samples with lowest confidence
    uncertain_indices = np.argsort(confidence)[:batch_size]
    
    # Save indices of samples to query
    np.save("queried_indices.npy", uncertain_indices)

if __name__ == "__main__":
    select_queries()
