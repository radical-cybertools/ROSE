# training.py
import time
import argparse

def train_model(home_dir, samples_file, model_name, learner_name, epochs=10):

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        from pathlib import Path
        from models import MC_Dropout_CNN, BayesianNN, MC_Dropout_MLP, elbo_loss
        import json

        model_file = Path(home_dir, f'{model_name}.pkl')
        transform = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)

        with open(Path(home_dir, learner_name + samples_file), 'r') as f:
            labeled_idx = json.load(f)
        loader = DataLoader(Subset(full_train, labeled_idx), batch_size=64, shuffle=True)
        
        # Recreate model architecture
        if model_name == 'MC_Dropout_CNN':
            model = MC_Dropout_CNN()  # Or your model class
        elif model_name == 'BayesianNN':
            model = BayesianNN()
        elif model_name == 'MC_Dropout_MLP':
            model = MC_Dropout_MLP()
        else:
            print(f"Model {model_name} not recognized. Please use BayesianNN, MC_Dropout_CNN or MC_Dropout_MLP.")
            return
        
        # Load weights from previous iteration of training
        try:
            model.load_state_dict(torch.load(model_file))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            pass

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            model.train()
            total_loss = 0
            for X, y in loader:
                device = next(model.parameters()).device
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(X)
                
                if 'bayesian' in model_name.lower():
                    # Use ELBO loss for Bayesian models
                    kl = model.kl_loss()
                    loss = elbo_loss(output, y, kl, kl_weight=1.0 / len(loader))
                else:
                    loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Train loss: {total_loss:.4f}")

        # Example: save model weights
        torch.save(model.state_dict(), model_file)
    except:
        for _ in range(100):
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--model_name', type=str, help='Name of the model used for training')
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--home_dir', type=str, help='Home directory for the project')
    args = parser.parse_args()

    print(args)

    train_model(home_dir=args.home_dir, samples_file="_samples.json", model_name=args.model_name, learner_name=args.learner_name, epochs=args.epochs)
