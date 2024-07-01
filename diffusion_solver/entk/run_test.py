# %%
import os
import torch
import trainers
import csv
import argparse
import numpy as np
import random
import util
from util.plotter import plotSamp, myPlots, plotSampRelative
import matplotlib.pyplot as plt
from src.util import plot_network_mask


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
    return parser.parse_known_args()[0]

args = argument_parser()
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
cuda_device_idx = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_idx


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# save_loc = r'./diffusion-ai-results/log-transform'

# CNN feature extractor
# save_loc = "/home/pb8294/Documents/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2022-03-20-14-36-04/"

# Adaptive 1.1, 10
save_loc = "/home/pb8294/Documents/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2022-03-20-14-46-28/"

# Adaptive 1.1, 2
# save_loc = "/home/pb8294/Documents/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2022-03-20-17-15-25/"


if not os.path.isdir(save_loc):
    os.mkdir(save_loc)

load_loc = r"/home/pb8294/Documents/Projects"
dataset_name = 'TwoSourcesRdm'
BATCH_SIZE = 50
NUM_WORKERS = 8
lr = 0.0001

# %%
train_loader, test_loader = util.loaders.generateDatasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=0.01,
                                                                  s=100,
                                                                  transformation='linear').getDataLoaders()
theModel = util.NNets.AdaptiveConvNet(1, 1, 1, 3, args, device).to(device); adaptive = 1
# theModel = util.NNets.SimpleClas().to(device); adaptive = 0
theModel.load_state_dict(torch.load(save_loc + "diffusion-model.pt")['model_state_dict'])
# a_k, b_k = theModel.structure_sampler.get_variational_params()
# print(a_k)
# print(b_k)
# np.savetxt(save_loc + "a_k.txt", a_k.detach().cpu().numpy())
# np.savetxt(save_loc + "b_k.txt", b_k.detach().cpu().numpy())
# exit(0)

plt.close("all")
plt.figure(dpi=300)
ax = plt.gca()
plot_network_mask(ax, theModel, ylabel=True, sz=7)
plt.savefig(save_loc + "/acquired_final_model.png")

# plotDiff(save_loc, 1, device, error_list, test_error, testloader, diff, epoch, transformation="linear", bash=False)
diction = {"transformation": "linear"}
print(diction['transformation'])
plotSamp(theModel, test_loader, diction, device, save_loc, "plots", n=1, adaptive=adaptive)
plotSampRelative(theModel, test_loader, diction, device, save_loc, "plot_rel", adaptive=adaptive)
plotter_c = myPlots(adaptive=adaptive)
# test_csv = np.loadtxt(save_loc + "test_error.csv", dtype="float")
# train_csv = np.loadtxt(save_loc + "train_error.csv", dtype="float")
test_csv = np.genfromtxt(save_loc + "test_error.csv", delimiter=',')
train_csv = np.genfromtxt(save_loc + "train_error.csv", delimiter=',')
print(type(test_csv), type(train_csv))
plotter_c.plotDiff(save_loc, "1", device, train_csv, test_csv, test_loader, theModel, 0)


