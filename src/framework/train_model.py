import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib  # To save the model

def train_model():
    # Load labeled data
    X_labeled = np.load("X_labeled.npy")
    y_labeled = np.load("y_labeled.npy")

    # Train model
    model = RandomForestClassifier()
    model.fit(X_labeled, y_labeled)

    # Save the trained model
    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    train_model()

