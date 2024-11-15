import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_iris


def evaluate_model():
    # Load full dataset and trained model
    X, y = np.load("X.npy"), np.load("y.npy")  # Full dataset
    model = joblib.load("model.joblib")

    # Predict and evaluate accuracy
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Save accuracy to a file
    #with open("accuracy.txt", "w") as f:
    #    f.write(f"Accuracy: {accuracy:.2f}\n")
    print(f"Accuracy: {accuracy:.2f}\n")

if __name__ == "__main__":
    # First, save the full dataset (for one-time use)
    data = load_iris()
    np.save("X.npy", data.data)
    np.save("y.npy", data.target)

    evaluate_model()

