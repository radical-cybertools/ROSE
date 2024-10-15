# %%
import os
import torch
import trainers
import csv
import argparse
import numpy as np
import random

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
                        help='random/entropy/diversity/tod')
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
    parser.add_argument('--exp', type=str, default="exp1",
                        help='loss/std')
    parser.add_argument('--mname', type=str, default="mname2",
                        help='loss/std')
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
save_loc = r'/scratch/pb_data/results/diffusion-ai-results/log-transform'
if not os.path.isdir(save_loc):
    os.mkdir(save_loc)

load_loc = r"/scratch/pb_data"
dataset_name = 'All'

if not os.path.isdir(save_loc):
    os.mkdir(save_loc)
save_loc = os.path.join(save_loc,dataset_name)
BATCH_SIZE = 16
NUM_WORKERS = 4
lr = 0.0001

# %%
trainer = trainers.DeepDiffusionTrainer(device=device, std_tr=0.01, s=512)

# %%
error_list, test_error, test_error_plus, test_error_minus, save_loc = \
        trainer.train_diff_solver512(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS,
                                  dataset_name=dataset_name, transformation=args.lwt, args=args)

# %%
with open(os.path.join(save_loc, 'train_error.csv'), 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(error_list)
with open(os.path.join(save_loc, 'test_error.csv'), 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(test_error)
if len(test_error_plus):
    with open(os.path.join(save_loc, 'test_N+1_error.csv'), 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(test_error_plus)
if len(test_error_minus):
    with open(os.path.join(save_loc, 'test_N-1_error.csv'), 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(test_error_minus)

# %%



