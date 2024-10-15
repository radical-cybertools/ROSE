from copy import deepcopy
import csv
import os

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
import util
import time
from src.util import plot_network_mask
from torch.utils.data import Subset
from util.plotter import myPlots

def save_model(save_loc, model, opt, error_list, epoch):
    file_name = os.path.join(save_loc, 'diffusion-model.pt')
    torch.save({'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_sate_dict': opt.state_dict(),
                'loss'               : error_list}, file_name)
    return


class Train:

    def __init__(self, device, std_tr, s, custom_loss=None):
        self.device = device
        self.std_tr = std_tr
        self.s = s

        if custom_loss is None:
            self.loss = self.my_loss
        else:
            self.loss = custom_loss

    @staticmethod
    def my_loss(output, target, alph=1, w=1, w2 = 2000):
#         loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/w) * torch.abs((output - target)**alph))
        # print(output.shape, target.shape, w, w2)
        loss = torch.mean((1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph))
#         dict["w"] = w
#         dict["w2"] = w2
#         dict["alph"] = alph
        return loss
#     @staticmethod
#     def my_loss(output, target, alph=2, w=1):
#         loss = torch.mean(torch.exp(-(torch.ones_like(output) - output) / w) * torch.abs((output - target) ** alph))
#         return loss

    @staticmethod
    def test_diff_error(neural_net, loader, criterion, device, ada=0):
        neural_net.eval()
        with torch.no_grad():
            error = 0.0
            for i, data in enumerate(loader):
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = neural_net(x)
                if ada == 1:
                    err = criterion(yhat.mean(0), y)
                else:
                    err = criterion(yhat, y)
                error += err.item()
        neural_net.train()
        return error / (i + 1)

    def train_diff_solver(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=100, snap=25,
                          dataset_name='TwoSourcesRdm', transformation='linear', args=None):
        from datetime import datetime
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        save_loc = os.path.join(save_loc, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        print('Save location:', save_loc)
        
        train_data, test_data, val_data = util.loaders.generateDatasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=self.std_tr,
                                                                  s=self.s,
                                                                  transformation=transformation).getDatasets()
        all_indices = np.arange(len(train_data))
        r = np.random.RandomState(12345)
        r.shuffle(all_indices)
        error_list = []
        test_error, test_error_plus, test_error_minus = [], [], []
        
        ada = 0
        if ada == 1:
            diff_solver = util.NNets.AdaptiveConvNet(1, 1, 1, 3, args, self.device).to(self.device)
        else:
            diff_solver = util.NNets.SimpleCNN100().to(self.device)
        print(diff_solver)
        # x = torch.rand((1, 1, 100, 100))
        # x = x.float().to(self.device)
        # print(diff_solver(x, 10).shape)
        # # # diff_solver(x, 1)
        # print(diff_solver); exit()
        criterion = self.loss

        opt = optim.Adam(diff_solver.parameters(), lr=lr)
        init_size = 1000
        pool_size = len(train_data) - init_size
        init_indices = all_indices[:init_size]
        pool_indices = all_indices[init_size:]
        
        # disp_log_msg("Dataset Split for AL", "Init size: {}, Pool size: {}, Train size: {}, Repeated Indices: {}".format(len(init_indices), len(init_indices), len(init_indices), check_uniq_indices(init_indices, pool_indices)))
        init_dataset = Subset(train_data, init_indices)
        pool_dataset = Subset(train_data, pool_indices)
        labelled_idx = deepcopy(init_indices)

        # pool_dataset, valid_dataset, init_dataset = torch.utils.data.random_split(train_data, [pool_size, valid_size, init_size])
        
        train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=128,
                                                shuffle=True, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                                shuffle=False, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128,
                                                shuffle=False, num_workers=NUM_WORKERS)
        pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32,
                                                shuffle=False, num_workers=NUM_WORKERS)
        name_plus, name_minus = util.get_Nplus_Nminus(dataset_name)

        test_loader_plus, test_loader_minus = None, None
        r = np.random.RandomState(12345)
        for portion in range(1000, 17000, 1000):
            

            # if name_plus is not None:
            #     b_size = max(BATCH_SIZE, 1)
            #     _, test_loader_plus = util.loaders.generateDatasets(PATH=load_loc,
            #                                                         datasetName=name_plus,
            #                                                         batch_size=b_size,
            #                                                         num_workers=NUM_WORKERS,
            #                                                         std_tr=self.std_tr, s=self.s,
            #                                                         transformation=transformation).getDataLoaders()
            # if name_minus is not None:
            #     b_size = max(BATCH_SIZE, 1)
            #     _, test_loader_minus = util.loaders.generateDatasets(PATH=load_loc,
            #                                                          datasetName=name_minus,
            #                                                          batch_size=b_size,
            #                                                          num_workers=NUM_WORKERS,
            #                                                          std_tr=self.std_tr, s=self.s,
            #                                                          transformation=transformation).getDataLoaders()
            error_list = []
            test_error, test_error_plus, test_error_minus = [], [], []
            # acc, accTrain = [], []

            # todo: add plots back
            for epoch in range(epochs):
                ada = 0
                if ada == 1:
                    diff_solver = util.NNets.AdaptiveConvNet(1, 1, 1, 3, args, self.device).to(self.device)
                else:
                    diff_solver = util.NNets.SimpleCNN100().to(self.device)

                criterion = self.loss

                opt = optim.Adam(diff_solver.parameters(), lr=lr)

                st_time = time.time()
                print("Epoch", epoch)
                error = 0.0
                for (i, data) in enumerate(train_loader):
                    diff_solver.zero_grad()
                    x = data[0].to(self.device)
                    y = data[1].to(self.device)
                    yhat = diff_solver(x)
                    if ada == 1:
                        err = diff_solver.estimate_ELBO(criterion, yhat, y, len(train_loader.dataset))
                    else:
                        err = criterion(yhat, y)
                    err.backward()
                    opt.step()
                    error += err.item()
                error_list.append(error / (i + 1))

                test_error.append(self.test_diff_error(diff_solver, test_loader, criterion, self.device, ada=ada))
                if test_loader_plus is not None:
                    test_error_plus.append(self.test_diff_error(diff_solver, test_loader_plus, criterion, self.device))
                if test_loader_minus is not None:
                    test_error_minus.append(self.test_diff_error(diff_solver, test_loader_minus, criterion, self.device))

                with open(os.path.join(save_loc, 'train_error{}.csv'.format(portion)), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(error_list)

                with open(os.path.join(save_loc, 'test_error{}.csv'.format(portion)), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_error)

                if len(test_error_plus):
                    with open(os.path.join(save_loc, 'test_N+1_error{}.csv'.format(portion)), 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerow(test_error_plus)

                if len(test_error_minus):
                    with open(os.path.join(save_loc, 'test_N-1_error{}.csv'.format(portion)), 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerow(test_error_minus)
                if epoch % snap == snap - 1:
                    save_model(save_loc, diff_solver, opt, error_list, epoch)
                    
                et_time = time.time()
                print("Epoch time: {}".format(et_time - st_time))
                if ada == 1:
                    plt.close("all")
                    plt.figure(dpi=300)
                    ax = plt.gca()
                    plot_network_mask(ax, diff_solver, ylabel=True, sz=7)
                    plt.savefig(save_loc + "/inferred-epochs_"+str(epoch)+".png")
            plotter_c = myPlots(adaptive=0)
            plotter_c.plotDiff(save_loc, "diff_plots", self.device, error_list, test_error, test_loader, diff_solver, epoch=portion)
            print('TRAINING FINISHED')
            del diff_solver, opt

            if portion < 16000:
                k_indices = np.arange(len(pool_indices))
                r.shuffle(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=128,
                                                    shuffle=True, num_workers=NUM_WORKERS)
                print(len(train_loader.dataset))
            else:
                break
        return error_list, test_error, test_error_plus, test_error_minus, save_loc