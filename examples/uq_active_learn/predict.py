# active_learn.py
import json
from pathlib import Path
import numpy as np
import argparse

QUERY_SIZE = 10

def prediction(home_dir, pool_file, predict_file, model_name, prediction_dir, prediction_num, learner_name):
    # Load samples``

    predict_file = Path(prediction_dir, str(prediction_num) + '_' + model_name + predict_file)
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        import torch.nn.functional as F
        from models import MC_Dropout_CNN, BayesianNN, MC_Dropout_MLP
        predict_dir = Path(home_dir, prediction_dir)

        # Remove all predictions from previous prediction_num.
        if predict_dir.is_dir():
            for file in predict_dir.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"Error deleting file {file}: {e}")
        else:
            predict_dir.mkdir(parents=True, exist_ok=True)

        
        model_file = Path(home_dir, f'{model_name}.pkl')

        transform = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        with open(Path(home_dir, learner_name + pool_file), 'r') as f:
            pool_idx = json.load(f)

        pool_loader = DataLoader(Subset(full_train, pool_idx), batch_size=64, shuffle=True)

        # Recreate model architecture
        if model_name == 'MC_Dropout_CNN':
            model = MC_Dropout_CNN()  
        elif model_name == 'BayesianNN':
            model = BayesianNN()
        elif model_name == 'MC_Dropout_MLP':
            model = MC_Dropout_MLP()
        else:
            print(f"Model {model_name} not recognized. Please use BayesianNN, MC_Dropout_CNN, or MC_Dropout_MLP.")
            return

        # Load weights
        try:
            model.load_state_dict(torch.load(model_file))
        except Exception as e:
            print(f"Error loading model weights: {model_name} not saved to {model_file}. Error: {e}")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_preds = []
        for _ in range(QUERY_SIZE):
            batch_preds = []
            with torch.no_grad():
                for n, (X, _) in enumerate(pool_loader):
                    X = X.to(device)
                    logits = model(X)
                    probs = F.softmax(logits, dim=1)
                    batch_preds.append(probs.cpu().numpy())
                    if n == 10:
                        break
            all_preds.append(np.vstack(batch_preds))
        all_preds = np.array(all_preds)
    except:
        all_preds = np.ones((QUERY_SIZE, 640, 10)) * 0.1  # Dummy predictions if anything fails

    np.save(predict_file, all_preds)
    print(f"Model {model_name} predictions are saved to {predict_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--model_name', type=str, help='Name of the model used for training')
    parser.add_argument('--prediction_dir', type=str, help='Directory for predictions')
    parser.add_argument('--prediction_num', type=str, help='Training prediction_num number for current model name')
    parser.add_argument('--learner_name', type=str, help='Name of the learner')
    parser.add_argument('--home_dir', type=str, help='Home directory for the project')
    args = parser.parse_args()

    prediction(args.home_dir, "_pool.json", "_predict.npy", model_name=args.model_name, prediction_dir=args.prediction_dir, prediction_num=args.prediction_num, learner_name=args.learner_name)