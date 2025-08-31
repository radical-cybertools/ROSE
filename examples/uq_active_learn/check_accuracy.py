# check_stop.py
import argparse
import  random


def check_stop(home_dir, model_name):
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        from models import MC_Dropout_CNN, BayesianNN, MC_Dropout_MLP
        from pathlib import Path
        model_file = Path(home_dir, model_name + "_model.pt")
        transform = transforms.Compose([transforms.ToTensor()])

        data_dir = Path(home_dir, 'mnist_data')
        try:
            mnist_test = datasets.MNIST(root=data_dir,
                                        train=False,
                                        transform=transform,
                                        download=False  # Do NOT download again
                                    )
        except:           
            mnist_test = datasets.MNIST(root=data_dir, 
                                        train=False, 
                                        download=True, 
                                        transform=transform)

        test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

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
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_file))
        except Exception as e:
            print(f"Error loading model weights: {model_name} not saved to {model_file}. Error: {e}")
            return

        model.eval()
        correct = 0
        tot = 0
        with torch.no_grad():
            for n, (X, y) in enumerate(test_loader):
                logits = model(X)
                pred = logits.argmax(dim=1)
                tot += y.size(0)
                correct += (pred == y).sum().item()
                if n == 10:
                    break
        acc = correct / tot 

        #Return accuracy to pipeline executor.
        print(acc)
    except:
        print(random.random())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction argument parser")
    parser.add_argument('--model_name', type=str, help='Name of the model used for training')
    parser.add_argument('--home_dir', type=str, help='Home directory for the project')
    args = parser.parse_args()
    check_stop(args.home_dir, "_model.pt", model_name=args.model_name)