import json
from sklearn.linear_model import LogisticRegression
import joblib

def active_learn():
    # Hard-coded file paths
    input_file = "simulation_data.json"
    model_file = "trained_model.pkl"
    updated_data_file = "updated_simulation_data.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    model = joblib.load(model_file)

    # Simulate adding more labeled data
    unlabeled_data = [{"feature": i} for i in range(100, 120)]
    new_data = [{"feature": d["feature"], "label": model.predict([[d["feature"]]])[0]} for d in unlabeled_data]

    updated_data = data + new_data
    with open(updated_data_file, "w") as f:
        json.dump(updated_data, f)

    print(f"Updated data saved to {updated_data_file}")

if __name__ == "__main__":
    active_learn()

