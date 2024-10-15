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

@staticmethod
def my_loss(output, target, alph=1, w=1, w2 = 2000):
    loss = torch.mean((1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph))
    return loss


def save_model(save_loc, model, opt, error_list, epoch, model_name="diffusion-model.pt"):
    file_name = os.path.join(save_loc, model_name)
    torch.save({'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_sate_dict': opt.state_dict(),
                'loss'               : error_list}, file_name)
    return

@staticmethod
def test_diff_error(neural_net, loader, criterion, device):
    neural_net.eval()

    with torch.no_grad():
        error = 0.0
        for i, data in enumerate(loader):
            x = data[0].to(device)
            y = data[1].to(device)
            
            yhat = neural_net(x)
            err = criterion(yhat, y)
            
            error += err.item()
    neural_net.train()
    return error / (i + 1)

def train_net(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, snap=25,
                          dataset_name='20SourcesRdm', transformation='linear', args=None):

        # ESSENTIAL VARIABLES
        portion = args.portion
        epochs = args.epochs

        # DEVICE / CUDA SETUP        
#        cuda_device_idx = str(args.card)
#        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_idx
#        print("CUDA GPU Card Number: ", cuda_device_idx)
#        print("TIANLE DEBUG: CUDA GPU Card Number: ", cuda_device_idx, flush=True)
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
                                                                  batch_size=16,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=0.01,
                                                                  s=512,
                                                                  transformation=transformation).getDatasets()

        if args.initial == 1:
            print("Network Initialization Phase. Creating initial data split ...")
            all_indices = np.arange(len(train_data))
            r = np.random.RandomState(12345)
            r.shuffle(all_indices)
            error_list = []
            test_error, test_error_plus, test_error_minus = [], [], []
            
            init_size = 1000
            init_indices = all_indices[:init_size]
            valid_indices = all_indices[init_size:init_size+1000]
            np.save(load_loc + "/labelled_indices", init_indices)
            np.save(load_loc + "/valid_indices", valid_indices)
        elif args.initial == 0:
            print("Network Training Phase. Getting the labelled indices ...")
            init_indices = np.load(load_loc + "labelled_indices.npy")
            valid_indices = np.load(load_loc + "valid_indices.npy")

        labelled_idx = deepcopy(init_indices)
        init_dataset = Subset(train_data, labelled_idx)
        valid_dataset = Subset(train_data, valid_indices)
        train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                shuffle=True, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,
                                                shuffle=False, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16,
                                                shuffle=False, num_workers=NUM_WORKERS)
        print("Data sizes -> Training: {}, Validation: {}, Testing: {}".format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))

        # INITIALIZING NETWORK, LOSS AND OPTIMIZERS
        diff_solver = util.NNets.UNet().to(device)
        criterion = my_loss
        opt = optim.Adam(diff_solver.parameters(), lr=lr)

        # SETTING UP VARIABLES FOR INTERMEDIATE RESULT STORAGE
        error_list = []
        test_error, test_error_plus, test_error_minus = [], [], []
        val_error = []

        best_loss = np.Inf
        early_stop = 0
        patience = 50
        # if portion > 12000:
        for epoch in range(epochs):
            st_time = time.time()
            print("Epoch", epoch)
            error = 0.0
            for (i, data) in enumerate(train_loader):
                diff_solver.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diff_solver(x)
                err = criterion(yhat, y)
                err.backward()
                opt.step()
                error += err.item()
            error_list.append(error / (i + 1))

            test_error.append(test_diff_error(diff_solver, test_loader, criterion, device))

            with open(os.path.join(save_loc, 'train_error{}.csv'.format(portion)), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(error_list)

            with open(os.path.join(save_loc, 'test_error{}.csv'.format(portion)), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(test_error)

            ve = test_diff_error(diff_solver, val_loader, criterion, device)
            val_error.append(ve)
            with open(os.path.join(save_loc, 'valid_error{}.csv'.format(portion)), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(val_error)
            

            if args.es == 1:
                # pass
                if ve < best_loss:
                    print("Validation loss reduced {} --> {}. Saving model ...".format(best_loss, ve))
                    best_loss = ve
                    # save_model(save_loc, diff_solver, opt, error_list, epoch)
                    save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))
                    early_stop = 0
                else:
                    early_stop += 1
                    if early_stop >= patience:
                        print("Patience broken. No improvement in training")
                        break
            else:
                save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))

                # CODE TO SAVE MODEL AFTER snap EPOCHS
                # if epoch % snap == snap - 1:
                #     save_model(save_loc, diff_solver, opt, error_list, epoch)

                
            et_time = time.time()
            print("Epoch time: {}".format(et_time - st_time))
            
        
        plotter_c = myPlots(adaptive=0)
        plotter_c.plotDiff(save_loc, "diff_plots", device, error_list, test_error, test_loader, diff_solver, epoch=portion)
        print('TRAINING FINISHED')


        

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
    load_loc = r"/lus/eagle/projects/RECUP/twang/rose/diffusion_solver/data/"
    dataset_name = "20SourcesRdm"
    save_loc = r"/lus/eagle/projects/RECUP/twang/rose/diffusion_solver/exp/"
    lr = 0.0001
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    train_net(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, dataset_name=dataset_name, args=args)
