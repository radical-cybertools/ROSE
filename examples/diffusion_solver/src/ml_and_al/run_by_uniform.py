# %%
import os
import torch
import trainers
import csv
import argparse
import numpy as np
import random
import util 
import matplotlib.pyplot as plt

def argument_parser():
    parser = argparse.ArgumentParser(description="Run Nonparametric Bayesian Architecture Learning")
    parser.add_argument('--use-cuda', action='store_false', 
                        help="Use CPU or GPU")
    parser.add_argument("--prior_temp", type=float, default=1.,
                        help="Temperature for Concrete Bernoulli from prior")
    parser.add_argument("--temp", type=float, default=.5,                 
                        help="Temperature for Concrete Bernoulli from posterior")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon to select the activated layers")
    parser.add_argument("--truncation_level", type=int, default=10,
                        help="K+: Truncation for Z matrix")
    parser.add_argument("--a_prior", type=float, default=2,
                        help="a parameter for Beta distribution")
    parser.add_argument("--b_prior", type=float, default=1.,
                        help="b parameter for Beta distribution")
    parser.add_argument("--kernel", type=int, default=5,
                        help="Kernel size. Default is 3.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples of Z matrix")
    parser.add_argument("--epochs", type=int, default=50,                
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-6,
                        help="Coefficient of weight decay.")
    parser.add_argument("--batch_size", type=float, default=1,
                        help="Batch size.")
    parser.add_argument("--max_width", type=int, default=1,
                        help="Dimension of hidden representation.")
    parser.add_argument('--func', type=str, default="acquisition function",
                        help='random/entropy/diversity')
    parser.add_argument('--sub', type=str, default="entropy sub function",
                        help='loss/std')
    parser.add_argument('--lwt', type=str, default="linear",
                        help='linear/log')
    parser.add_argument('--net', type=str, default="simplecnn",
                        help='simplecnn/unetab')
    parser.add_argument("--es", type=int, default=0,
                        help="Early stop")
    parser.add_argument("--by_dist", type=int, default=0,
                        help="Test by distance")
    parser.add_argument("--card", type=int, default=3,
                        help="GPU card")
    parser.add_argument("--param_size", type=int, default=7,
                        help="Param size (7 or 6)")
    return parser.parse_known_args()[0]

args = argument_parser()
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(args)
# %%
cuda_device_idx = str(args.card)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_idx
print("CUDA", cuda_device_idx)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_loc = r'./diffusion-ai-results/log-transform'
if not os.path.isdir(save_loc):
    os.mkdir(save_loc)

load_loc = r"/home/pb8294/data"
dataset_name = 'TwoSourcesRdm'

if not os.path.isdir(save_loc):
    os.mkdir(save_loc)
save_loc = os.path.join(save_loc,dataset_name)
BATCH_SIZE = 50
NUM_WORKERS = 8
lr = 0.0001

def assign_indices_based_on_intervals(distances):
    # Define the intervals. In this example, we split [0, 1] into 10 equal intervals.
    num_intervals = 10
    interval_width = 1 / num_intervals
    intervals = [(i * interval_width, (i + 1) * interval_width) for i in range(num_intervals)]
    
    # Initialize a list to store the assigned indices
    assigned_indices = []
    
    # Assign an index to each distance based on the interval it falls into
    for distance in distances:
        assigned_index = None
        for i, (start, end) in enumerate(intervals):
            if start <= distance <= end:
                assigned_index = i
                break
        assigned_indices.append(assigned_index)
    
    return assigned_indices

train_data, test_data, val_data = util.loaders.generateDatasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=0.01,
                                                                  s=100,
                                                                  transformation="linear").getDatasets()

test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                                shuffle=False, num_workers=NUM_WORKERS)

# indices = []

# distances = []
# for i, data in enumerate(test_loader):
#     dist = np.array(data[-1]).reshape(-1, 1)
#     if i == 0:
#         distances = dist
#     else:
#         distances = np.vstack((distances, dist))


# np.save("distances_npy", distances)
actual_indices = np.arange(4000)
distances = np.load("distances_npy.npy")
counts, bins = np.histogram(distances, 100)
print(bins, bins.shape)
distances = distances.reshape(-1, 1)
print(actual_indices)
empty_diction = {}
for i in range(100):
    empty_diction[i] = []
    
indices = []
for ind, dist in enumerate(distances):
    for i, bn in enumerate(bins):
        if bins[i] < dist and dist <= bins[i+1] and len(empty_diction[i]) < 10:
            empty_diction[i].append(ind)

all_values = []
for val in empty_diction.values():
    all_values += val
all_values = np.array(all_values) 
print(all_values, all_values.shape)
# exit(0)
selected_indices = actual_indices[all_values]
selected_distances = distances[all_values]
print(selected_indices)

# np.random.shuffle(actual_indices)
# distances = distances[actual_indices]
plt.hist(distances, 100)
plt.hist(selected_distances, 100, color="red")
plt.savefig("zz.png")
print(min(distances), max(distances))
print(actual_indices)
exit(0)

indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9:[]}
for i in range(10):
    indexes[i] = actual_indices[indices.flatten() == i]
    print(len(indexes[i]))
# print(indexes)
trainer = trainers.DeepDiffusionTrainer(device=device, std_tr=0.01, s=100)

path_0drop = {"random": "", "std": "", "loss": "", "diverse": "", "diverse_param6": "", "diverse_param7": ""}

# WORKING LOCATIONS
path_0drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-30/"
# path_0drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-58/"
path_0drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/"
path_0drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-40-58/"
path_0drop["diverse_param6"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/"
path_0drop["diverse_param7"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-02-59/"

path_0drop["std_bb"] = "/home/pb8294/Projects/DeepDiffusionSolver//diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-07-11-54-25/"
path_0drop["rnd_bb"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-12-06-45/"


keys = ["rnd_bb"]
for k in keys:
    arr = np.zeros((16, 10))
    for index_i, portion in enumerate(range(1000, 17000, 1000)):
        for index_j, i in enumerate(range(10)):
            print(k, index_i, index_j, len(indexes[i]))
            test_subset = torch.utils.data.Subset(test_data, indexes[i])
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128,
                                                        shuffle=True, num_workers=NUM_WORKERS)
            diff_solver = util.NNets.SimpleCNN100().to(device)
            load_saved_model = torch.load(path_0drop[k] + "diffusion-model-{}.pt".format(portion))
            diff_solver.load_state_dict(load_saved_model['model_state_dict'])
            

            arr[index_i, index_j] = trainer.test_diff_error(diff_solver, test_loader, trainer.loss, device, ada=0)
            del diff_solver
            # print(arr)

    np.save("./distwise_{}".format(k), arr)
exit(0)



# %%