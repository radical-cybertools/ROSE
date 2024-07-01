import argparse
from copy import deepcopy
import csv
import os
import random
from select import select

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import util
import time
from src.util import plot_network_mask
from torch.utils.data import Subset
from util.plotter import myPlots

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
    parser.add_argument("--initial", type=int, default=1, help="Initial training (1) / Subsequent training (0)")
    parser.add_argument("--portion", type=int, default=1, help="Active learning portion out of total acquisition rounds (e.g. 1 out of 16)")
    return parser.parse_known_args()[0]


def get_tod_uncertainty(diff_solver, cod_model, pool_loader, device):
    diff_solver.eval()
    cod_model.eval()
    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (i, data) in enumerate(pool_loader):
            x = data[0].to(device)
            # print(x.shape)
            y = data[1].to(device)
            yhat = diff_solver(x)
            yhat_cod = cod_model(x)

            pred_loss = (yhat - yhat_cod).pow(2).sum(dim=(1, 2, 3)) / 2   
            uncertainty = torch.cat((uncertainty, pred_loss), dim=0)

    return uncertainty.cpu()

def active_run(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, dataset_name, transformation='linear', args=None):
    # ESSENTIAL VARIABLES
    portion = args.portion
    epochs = args.epochs

    # DEVICE / CUDA SETUP        
#    cuda_device_idx = str(args.card)
#    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_idx
#    print("CUDA GPU Card Number: ", cuda_device_idx)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # CREATING AND SETTING UP SAVE LOCATIONS
    from datetime import datetime
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    save_loc = os.path.join(save_loc, "UNET", "train_net")#datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    print('Save location:', save_loc)

    
    # CREATING DATA SPLITS 
    train_data, test_data = util.loaders.generate1to20Datasets(PATH=load_loc,
                                                                datasetName=dataset_name,
                                                                batch_size=BATCH_SIZE,
                                                                num_workers=NUM_WORKERS,
                                                                std_tr=0.01,
                                                                s=512,
                                                                transformation=transformation).getDatasets()

    all_indices = np.arange(len(train_data))
    r = np.random.RandomState(12345)
    r.shuffle(all_indices)
    init_indices = np.load(load_loc + "labelled_indices.npy")
    valid_indices = np.load(load_loc + "valid_indices.npy")
    collective_avoid_indices = np.vstack((init_indices.reshape(-1, 1), valid_indices.reshape(-1, 1)))
    pool_indices = all_indices[~np.isin(all_indices, collective_avoid_indices)]

    labelled_idx = deepcopy(init_indices)
    init_dataset = Subset(train_data, labelled_idx)
    valid_dataset = Subset(train_data, valid_indices)
    pool_dataset = Subset(train_data, pool_indices)
    train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)
    pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=NUM_WORKERS)

    print("Data sizes -> Training: {}, Validation: {}, Testing: {}, Pool: {}".format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), len(pool_loader.dataset)))

    diff_solver = util.NNets.UNet().to(device)
    if args.func == "tod":
        cod_model = util.NNets.UNet().to(device)
        for param in cod_model.parameters():
            param.detach_()

    if portion > 1 and args.func == "tod":
        print("Loaded TOD model: {}".format(save_loc + "/diffusion-model-{}.pt".format(portion - 1)))
        load_cod_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion - 1))
        cod_model.load_state_dict(load_cod_model['model_state_dict'])
    else:
        print("No Previous saved model. First Active learning session...")

    if args.func == "random":
        print("Random data acquisition ...")
        k_indices = np.arange(len(pool_indices))
        r.shuffle(k_indices)
        select_indices = pool_indices[k_indices[:1000]]
        # labelled_idx = np.append(labelled_idx, select_indices)
        # pool_indices = np.delete(pool_indices, k_indices[:1000])
        # init_dataset = Subset(train_data, labelled_idx)
        # train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE,
        #                                     shuffle=True, num_workers=NUM_WORKERS)
    elif args.func == "tod":
        print("TOD data acquisition ... ")
        tod_st_time = time.time()
        load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
        diff_solver.load_state_dict(load_saved_model['model_state_dict'])
        uncertainty = get_tod_uncertainty(diff_solver, cod_model, pool_loader, device=device)

        k_indices = torch.argsort(uncertainty, descending=True).numpy()
        select_indices = pool_indices[k_indices[:1000]]
        
        # labelled_idx = np.append(labelled_idx, select_indices)
        # pool_indices = np.delete(pool_indices, k_indices[:1000])
        
        # init_dataset = Subset(train_data, labelled_idx)
        # pool_dataset = Subset(train_data, pool_indices)
        
        # train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        # pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        tod_et_time = time.time()
        print("TOD Time: {}".format(tod_et_time - tod_st_time))

    labelled_idx = np.vstack((labelled_idx.reshape(-1, 1), select_indices.reshape(-1, 1)))
    np.save(save_loc + "/labelled_indices", labelled_idx)


if __name__ == "__main__":
    args = argument_parser()
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Define variables
#Tianle: Don't forget the '/' at last. Maybe we should use os.path.join()
    load_loc = r"/lus/eagle/projects/RECUP/twang/rose/diffusion_solver/data/"
    dataset_name = "20SourcesRdm"
    save_loc = r"/lus/eagle/projects/RECUP/twang/rose/diffusion_solver/exp/"
    lr = 0.0001
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    active_run(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, dataset_name=dataset_name, args=args)
