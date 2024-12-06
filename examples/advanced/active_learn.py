# active_learn.py
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

def active_learn(dataset_file, samples_file, model_file, updated_samples_file):
    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load dataset and current samples
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    with open(samples_file, 'r') as f:
        current_samples = json.load(f)
    
    # Extract remaining data
    current_texts = {s["text"] for s in current_samples}
    remaining_samples = [s for s in dataset if s["text"] not in current_texts]
    remaining_texts = [s["text"] for s in remaining_samples]
    
    if not remaining_texts:
        print("No remaining samples.")
        return

    # Predict probabilities for remaining samples
    probs = model.predict_proba(remaining_texts)
    uncertainties = np.abs(probs[:, 0] - 0.5)
    
    # Select most uncertain samples
    uncertain_indices = uncertainties.argsort()[:10]
    new_samples = [remaining_samples[i] for i in uncertain_indices]
    
    # Update samples
    current_samples.extend(new_samples)
    with open(updated_samples_file, 'w') as f:
        json.dump(current_samples, f)

if __name__ == "__main__":
    active_learn("dataset.json", "samples.json", "model.pkl", "samples.json")

