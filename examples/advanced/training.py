# training.py
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle

def train_model(samples_file, model_file):
    # Load samples
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    
    # Create pipeline: Vectorizer + Logistic Regression
    model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(texts, labels)
    
    # Save model to file
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model("samples.json", "model.pkl")

