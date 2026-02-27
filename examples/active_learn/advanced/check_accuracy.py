# check_stop.py
import json
import pickle

from sklearn.metrics import accuracy_score


def check_stop(samples_file, model_file):
    # Load samples
    with open(samples_file) as f:
        samples = json.load(f)
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Evaluate model
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)


if __name__ == "__main__":
    check_stop("samples.json", "model.pkl")
