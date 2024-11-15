import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the trained model

def train():
    # Hard-coded file paths
    input_file = "simulation_data.json"
    model_file = "trained_model.pkl"
    metrics_file = "train_metrics.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    X = [[item["feature"]] for item in data]
    y = [item["label"] for item in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {"accuracy": accuracy}
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {metrics_file}")

if __name__ == "__main__":
    train()

