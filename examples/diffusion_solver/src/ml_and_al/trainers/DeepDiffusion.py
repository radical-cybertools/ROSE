from copy import deepcopy
import csv
import os
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

def save_model(save_loc, model, opt, error_list, epoch, model_name="diffusion-model.pt"):
    file_name = os.path.join(save_loc, model_name)
    torch.save({'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_sate_dict': opt.state_dict(),
                'loss'               : error_list}, file_name)
    return

import collections

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

class Train:

    def __init__(self, device, std_tr, s, custom_loss=None):
        self.device = device
        self.std_tr = std_tr
        self.s = s

        if custom_loss is None:
            self.loss = self.my_loss
        else:
            self.loss = custom_loss
        print("alpha", 1)

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
    def my_loss_nomean(output, target, alph=1, w=1, w2=2000):
        loss = ((1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph))
#         dict["w"] = w
#         dict["w2"] = w2
#         dict["alph"] = alph
        return loss

    @staticmethod
    def custom_loss(output, target, alph=2, w=1, w2 = 2000):
        loss = (1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph)
        
        # loss = 100 * torch.abs((output - target)**alph)
        return loss

    @staticmethod
    def test_diff_error_by_dist(neural_net, loader, criterion, device, ada=0):
        neural_net.eval()

        with torch.no_grad():
            error = 0.0
            errors = torch.zeros((10, 1)).to(device)
            counts = torch.zeros((10, 1))
            for i, data in enumerate(loader):
                x = data[0].to(device)
                y = data[1].to(device)
                dist = data[-1].to(device)

                indices = np.array(assign_indices_based_on_intervals(dist))
                act_indices = np.arange(x.shape[0])
                
                yhat = neural_net(x)
                if ada == 1:
                    err = criterion(yhat.mean(0), y)
                else:
                    err = criterion(yhat, y)
                
                # print(act_indices[indices == 9])
                # print(err[act_indices[indices == 9]].mean())
                # print(err.shape, err[0].shape, err[0].mean().shape)
                # print(err[indices].shape)
                # if i == 0:
                #     print("continued")
                #     continue
                for index in np.arange(0, 10):
                    ind_matches = act_indices[indices == index]
                    if len(ind_matches) > 0:
                        counts[index] += 1
                        errors[index] += err[act_indices[indices == index]].mean()
                    else:
                        continue

        neural_net.train()
        return errors.detach().cpu().numpy() / (i + 1), errors.detach().cpu().numpy() / counts.numpy()

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

    @staticmethod
    def eval_mode_prediction(neural_net, loader, device, criterion, get_true=False, samples=1):
        lerr_arr = []
        out_arr = torch.tensor([])
        pred_arr = torch.tensor([])
        distances = torch.tensor([])
        lossess = torch.tensor([])
        # src_x, src_y, snk_x, snk_y = torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([])
        src_cds_arr = torch.tensor([])
        snk_cds_arr = torch.tensor([])

        error = 0.0
        neural_net.eval()
        for m in neural_net.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        for i, data in enumerate(loader):
            x = data[0].to(device)
            y = data[1].to(device)

            # Added for bounding box entropy
            src_cd = data[6].to(device)
            snk_cd = data[7].to(device)

            if samples > 1:
                preds_stack = torch.stack([neural_net(x) for _ in range(samples)])
                loss1 = criterion(preds_stack.mean(0), y)
            else:
                preds_stack = neural_net(x)
                loss1 = criterion(preds_stack, y)

            if i == 0:
                if get_true:
                    out_arr = y.detach().cpu()
                pred_arr = preds_stack.detach().cpu()
                lossess = loss1.detach().cpu()
                src_cds_arr = src_cd.detach().cpu().long()
                snk_cds_arr = snk_cd.detach().cpu().long()
            else:
                if get_true:
                    out_arr = torch.cat((out_arr, y.detach().cpu()))
                if samples > 1:
                    pred_arr = torch.cat((pred_arr, preds_stack.detach().cpu()), dim=1)
                else:
                    pred_arr = torch.cat((pred_arr, preds_stack.detach().cpu()))
                lossess = torch.cat((lossess, loss1.detach().cpu()))
                src_cds_arr = torch.cat((src_cds_arr, src_cd.detach().cpu().long()))
                snk_cds_arr = torch.cat((snk_cds_arr, snk_cd.detach().cpu().long()))

        lossess = torch.mean(lossess, (-1, -2))
        if get_true:
            return out_arr.detach().cpu().numpy(), pred_arr.detach().cpu().numpy(), lossess.detach().cpu().numpy(), src_cds_arr.detach().cpu().numpy(), snk_cds_arr.detach().cpu().numpy()
        else:
            return pred_arr.detach().cpu().numpy(), lossess.detach().cpu().numpy(), src_cds_arr.detach().cpu().numpy(), snk_cds_arr.detach().cpu().numpy()

    def train_diff_solver(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=500, snap=25,
                          dataset_name='TwoSourcesRdm', transformation='linear', args=None):
        from datetime import datetime
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)

        save_loc = os.path.join(save_loc, args.net)
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        # save_loc = os.path.join(save_loc, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        save_loc = os.path.join(save_loc, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        print('Save location:', save_loc)

        # diff_solver = util.NNets.SimpleClas().to(self.device)
        
        
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
        tester_arr = []
        for portion in range(1000, 17000, 1000):
            init_size = portion
            pool_size = len(train_data) - init_size
            init_indices = all_indices[:init_size]
            pool_indices = all_indices[init_size:]
            
            # disp_log_msg("Dataset Split for AL", "Init size: {}, Pool size: {}, Train size: {}, Repeated Indices: {}".format(len(init_indices), len(init_indices), len(init_indices), check_uniq_indices(init_indices, pool_indices)))
            labelled_idx = deepcopy(init_indices)
            init_dataset = Subset(train_data, labelled_idx)
            pool_dataset = Subset(train_data, pool_indices)
            # pool_dataset, valid_dataset, init_dataset = torch.utils.data.random_split(train_data, [pool_size, valid_size, init_size])
            
            train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                    shuffle=True, num_workers=NUM_WORKERS)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,
                                                    shuffle=False, num_workers=NUM_WORKERS)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=16,
                                                    shuffle=False, num_workers=NUM_WORKERS)
            pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32,
                                                    shuffle=False, num_workers=NUM_WORKERS)
            name_plus, name_minus = util.get_Nplus_Nminus(dataset_name)

            test_loader_plus, test_loader_minus = None, None

            ada = 0
            if ada == 1:
                diff_solver = util.NNets.AdaptiveConvNet(1, 1, 1, 3, args, self.device).to(self.device)
            else:
                if args.net == "simplecnn":
                    diff_solver = util.NNets.SimpleCNN100().to(self.device)
                elif args.net == "unetab":
                    diff_solver = util.NNets.UNetAB().to(self.device)
            print(diff_solver)
            criterion = self.loss

            opt = optim.Adam(diff_solver.parameters(), lr=lr)
            
            error_list = []
            test_error, test_error_plus, test_error_minus = [], [], []
            val_error = []
            patience = 25
            early_stop = 0
            best_loss = np.Inf
            # acc, accTrain = [], []

            # todo: add plots back
            temp_tester = 0
            for epoch in range(100):
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

                ve = self.test_diff_error(diff_solver, val_loader, criterion, self.device, ada=0)
                val_error.append(ve)

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

                with open(os.path.join(save_loc, 'valid_error{}.csv'.format(portion)), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(val_error)

                if len(test_error_plus):
                    with open(os.path.join(save_loc, 'test_N+1_error{}.csv'.format(portion)), 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerow(test_error_plus)

                if len(test_error_minus):
                    with open(os.path.join(save_loc, 'test_N-1_error{}.csv'.format(portion)), 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerow(test_error_minus)
                if args.es == 1:
                    if ve < best_loss:
                        print("Validation loss reduced {} --> {}. Saving model ...".format(best_loss, ve))
                        best_loss = ve
                        save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))
                        early_stop = 0
                        temp_tester = test_error[-1]
                    else:
                        early_stop += 1
                        if early_stop >= patience:
                            print("Patience broken. No improvement in training")
                            break
                else:
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

            tester_arr.append(temp_tester)
            print(tester_arr)
            plotter_c = myPlots(adaptive=0)
            plotter_c.plotDiff(save_loc, "diff_plots", self.device, error_list, test_error, test_loader, diff_solver, epoch=portion)
            print('TRAINING FINISHED')
            del diff_solver, opt


            # if portion == 16000:
            #     break
            # k_indices = np.arange(len(pool_indices))
            # r.shuffle(k_indices)
            # select_indices = pool_indices[k_indices[:1000]]
            # labelled_idx = np.append(labelled_idx, select_indices)
            # pool_indices = np.delete(pool_indices, k_indices[:1000])
            # init_dataset = Subset(train_data, labelled_idx)
            # train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
            #                                     shuffle=True, num_workers=NUM_WORKERS)
            print(len(train_loader.dataset))
        with open(os.path.join(save_loc, 'tester_arr.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(tester_arr)
        return error_list, test_error, test_error_plus, test_error_minus, save_loc

    def train_diff_solver512(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=200, snap=25,
                          dataset_name='All', transformation='linear', args=None):
        from datetime import datetime
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        # save_loc = os.path.join(save_loc, args.net, "debug")#datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        save_loc = os.path.join(save_loc, "1to20", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # save_loc = os.path.join(save_loc, args.net, "entropy")
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        print('Save location:', save_loc)

        # diff_solver = util.NNets.SimpleClas().to(self.device)
        
        
        train_data, test_data = util.loaders.generate1to20Datasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=16,
                                                                  num_workers=8,
                                                                  std_tr=self.std_tr,
                                                                  s=self.s,
                                                                  transformation=transformation).getDatasets()

        all_indices = np.arange(len(train_data))
        r = np.random.RandomState(12345)
        r.shuffle(all_indices)
        error_list = []
        test_error, test_error_plus, test_error_minus = [], [], []
        
        init_size = 1000
        pool_size = len(train_data) - init_size
        init_indices = all_indices[:init_size]
        valid_indices = all_indices[init_size:init_size+1000]
        pool_indices = all_indices[init_size+1000:]

        # print("Init size: {}, Pool size: {}, Train size: {}, Repeated Indices: {}".format(len(init_indices), len(pool_indices), len(init_indices), len(valid_indices)))
        # exit()
        labelled_idx = deepcopy(init_indices)
        init_dataset = Subset(train_data, labelled_idx)
        pool_dataset = Subset(train_data, pool_indices)
        valid_dataset = Subset(train_data, valid_indices)
        # pool_dataset, valid_dataset, init_dataset = torch.utils.data.random_split(train_data, [pool_size, valid_size, init_size])
        
        
        train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,
                                                shuffle=False, num_workers=8)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16,
                                                shuffle=False, num_workers=8)
        pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16,
                                                shuffle=False, num_workers=8)
        # name_plus, name_minus = util.get_Nplus_Nminus(dataset_name)

        # self.getParamEmbed(test_loader)
        # exit(0)

        print(len(train_loader.dataset), len(pool_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
        
        # for (i, data) in enumerate(pool_loader):
        #     x, y = data[0], data[1]
        #     print(x.shape, y.shape)
        
        # exit()


        test_loader_plus, test_loader_minus = None, None
        tester_arr = []
        tester_arr_by_count = []
        tester_arr_by_batch = []
        
        # raw_diff_solver = util.NNets.SimpleCNN100()
        for portion in range(1000, 17000, 1000):
            temp_tester = 0
            temp_tester_by_batch = []
            temp_tester_by_count = []
            ada = 0
            diff_solver = util.NNets.UNet().to(self.device)
            if args.func == "tod":
                cod_model = util.NNets.UNet().to(self.device)
                for param in cod_model.parameters():
                    param.detach_()

            print(diff_solver)

            if portion > 1000 and args.func == "tod":
                print("Loaded TOD model: {}".format(save_loc + "/diffusion-model-{}.pt".format(portion - 1000)))
                load_cod_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion - 1000))
                cod_model.load_state_dict(load_cod_model['model_state_dict'])
            else:
                print("No here", portion, "here")
            tod_st_time = time.time()
            # load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
            # diff_solver.load_state_dict(load_saved_model['model_state_dict'])
            # uncertainty = self.get_tod_uncertainty(diff_solver, cod_model, pool_loader)
            # exit(0)
            
            criterion = self.loss
            # selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion)
            # exit(0)
            
            opt = optim.Adam(diff_solver.parameters(), lr=lr)
            
            error_list = []
            test_error, test_error_plus, test_error_minus = [], [], []
            test_error_by_batch, test_error_by_count = [], []
            val_error = []
            # acc, accTrain = [], []

            # todo: add plots back
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

                # if args.by_dist == 1:
                #     test_err_by_batch , test_err_by_count = self.test_diff_error_by_dist(diff_solver, test_loader, self.my_loss_nomean, self.device, ada=ada)
                #     test_error_by_batch.append(test_err_by_batch) 
                #     test_error_by_count.append(test_err_by_count)
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

                ve = self.test_diff_error(diff_solver, val_loader, criterion, self.device, ada=0)
                val_error.append(ve)
                with open(os.path.join(save_loc, 'valid_error{}.csv'.format(portion)), 'w+') as f:
                    writer = csv.writer(f)
                    writer.writerow(val_error)
                # writer_e.add_scalar('Loss/valid', ve, epoch)
                

                if args.es == 1:
                    # pass
                    if ve < best_loss:
                        print("Validation loss reduced {} --> {}. Saving model ...".format(best_loss, ve))
                        best_loss = ve
                        # save_model(save_loc, diff_solver, opt, error_list, epoch)
                        save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))
                        early_stop = 0
                        temp_tester = test_error[-1]
                        # if args.by_dist == 1:
                        #     temp_tester_by_batch = test_error_by_batch[-1]
                        #     temp_tester_by_count = test_error_by_count[-1]
                    else:
                        # print(ve)
                        early_stop += 1
                        if early_stop >= patience:
                            print("Patience broken. No improvement in training")
                            break
                else:
                    print(ve)
                    temp_tester = test_error[-1]
                    save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))
                    # if epoch % snap == snap - 1:
                    #     save_model(save_loc, diff_solver, opt, error_list, epoch)

                    
                et_time = time.time()
                print("Epoch time: {}".format(et_time - st_time))
                
            tester_arr.append(temp_tester)
            # if args.by_dist == 1:
            #     tester_arr_by_batch.append(temp_tester_by_batch)
            #     tester_arr_by_count.append(temp_tester_by_count)
            #     print("Tester by batch", tester_arr_by_batch)
            #     print("Tester by count", tester_arr_by_count)

            print(tester_arr)
            
            # if portion == 3000:
                
                # exit()
            plotter_c = myPlots(adaptive=0)
            plotter_c.plotDiff(save_loc, "diff_plots", self.device, error_list, test_error, test_loader, diff_solver, epoch=portion)
            print('TRAINING FINISHED')

            if portion == 16000:
                break

            if args.func == "random":
                k_indices = np.arange(len(pool_indices))
                r.shuffle(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                    shuffle=True, num_workers=NUM_WORKERS)
            elif args.func == "entropy":
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                print("Getting losses and preds for {} acquisition".format(args.sub))
                _, pred_arr, lossess, _, _ = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 3)
                # print(lossess.shape, pred_arr.shape); exit(0)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                print(metric.shape, pred_arr.shape, mean_arr.shape)
                # x1, x2, y1, y2 = src_cds[:, 0] - 10, src_cds[:, 0] + 10, src_cds[:, 1] - 10, src_cds[:, 1] + 10
                # xx1, xx2, yy1, yy2 = snk_cds[:, 0] - 10, snk_cds[:, 0] + 10, snk_cds[:, 1] - 10, snk_cds[:, 1] + 10
               
                # x1[x1 < 0] = 0
                # x2[x2 > 100] = 100
                # y1[y1 < 0] = 0
                # y2[y2 > 100] = 100

                # xx1[xx1 < 0] = 0
                # xx2[xx2 > 100] = 100
                # yy1[yy1 < 0] = 0
                # yy2[yy2 > 100] = 100


                # metrics_arr = []
                # for m_counter in range(metric.shape[1]):
                #     metric_src = torch.mean(metric[:, m_counter, :, x1[m_counter]:x2[m_counter], y1[m_counter]:y2[m_counter]])
                #     metric_snk = torch.mean(metric[:, m_counter, :, xx1[m_counter]:xx2[m_counter], yy1[m_counter]:yy2[m_counter]])
                #     metrics_arr.append(0.5 * (metric_src + metric_snk))
                # print("Calculation of entropy done")
                # metric = torch.tensor(metrics_arr)

                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                # print(metric[0])
                # exit()
                if args.sub == "std":
                    k_indices = torch.argsort(metric, descending=True).numpy()
                elif args.sub == "loss":
                    k_indices = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

                del pred_arr, lossess
            elif args.func == "combine_entropy":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                true_arr, pred_arr, lossess = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                k_indices_std = torch.argsort(metric, descending=True).numpy()
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                c = np.empty((k_indices_std.size + k_indices_loss.size,), dtype=k_indices_std.dtype)
                c[0::2] = k_indices_std
                c[1::2] = k_indices_loss
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                plt.scatter(metric, lossess, color="blue", marker="o")
                plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "combine_all":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion)
                
                true_arr, pred_arr, lossess = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                k_indices_std = torch.argsort(metric, descending=True).numpy()
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                
                c = np.empty((k_indices_std[:1000].size + k_indices_loss[:1000].size + len(selected_indices),), dtype=k_indices_std.dtype)
                c[0::3] = selected_indices
                c[1::3] = k_indices_loss[:1000]
                c[2::3] = k_indices_std[:1000]
                # print(np.array(c).shape, c); exit(0)
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                plt.scatter(metric, lossess, color="blue", marker="o")
                plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "diversity":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion, args.param_size)
                select_indices = pool_indices[selected_indices]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, selected_indices)
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
            elif args.func == "combine_nodrop":
                print("Combining loss and p-diversity")
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion, args.param_size)

                true_arr, pred_arr, lossess, _, _ = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                c = np.empty((k_indices_loss[:1000].size + len(selected_indices),), dtype=k_indices_loss.dtype)
                c[0::2] = selected_indices
                c[1::2] = k_indices_loss[:1000]
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                # plt.scatter(metric, lossess, color="blue", marker="o")
                # plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "tod":
                tod_st_time = time.time()
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                uncertainty = self.get_tod_uncertainty(diff_solver, cod_model, pool_loader)

                k_indices = torch.argsort(uncertainty, descending=True).numpy()
                select_indices = pool_indices[k_indices[:1000]]
                
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                tod_et_time = time.time()
                print("TOD Time: {}".format(tod_et_time - tod_st_time))

            del diff_solver, opt
            if args.func == "tod":
                del cod_model
            print(len(train_loader.dataset))
        
        np.save(save_loc + "/tester_by_batch", tester_arr_by_batch)
        np.save(save_loc + "/tester_by_count", tester_arr_by_count)

        with open(os.path.join(save_loc, 'tester_arr.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(tester_arr)
        
        return error_list, test_error, test_error_plus, test_error_minus, save_loc


    def getParamEmbed(self, pool_queue, param_dim=7):
        embedding = torch.zeros([len(pool_queue.dataset), param_dim]).cuda()
        evaluated_instances = 0
        for batch_idx, unlabeled_data_batch in enumerate(pool_queue):
            _, _, c_src, c_snk, i_src, i_snk, _, _, dist = unlabeled_data_batch
            if param_dim == 7:
                d = torch.from_numpy(np.array([dist, c_src[:, 0], c_src[:, 1], c_snk[:, 0], c_snk[:, 1], i_src, i_snk])).transpose(1, 0)
            elif param_dim == 6:
                d = torch.from_numpy(np.array([dist, c_src[:, 0], c_src[:, 1], c_snk[:, 0], c_snk[:, 1], i_snk])).transpose(1, 0)
            start_slice = evaluated_instances
            end_slice = start_slice + c_src.shape[0]
            embedding[start_slice:end_slice] = d
            evaluated_instances = end_slice

        return embedding

    def getEmbed(self, pool_queue, model, device, criterion):
        model.eval()
        Lloss = F.cross_entropy
        embDim = 64
        embedding = torch.zeros([len(pool_queue.dataset), embDim]).cuda()
        evaluated_instances = 0
        for batch_idx, unlabeled_data_batch in enumerate(pool_queue):
            inputs = unlabeled_data_batch[0]
            targets = unlabeled_data_batch[1]
            start_slice = evaluated_instances
            end_slice = start_slice + inputs.shape[0]
            
            inputs = inputs.to(device, non_blocking=True)
            
            l1, out = model(inputs, last=True, freeze=True)
            l1 = torch.mean(l1, (2, 3))
            # l1, out = torch.stack([model(inputs) for _ in range(10)]).float()
            # print(l1.shape, out.shape)
            
            # l1 = out.mean(0, 2, 3, 4)
            start_slice = evaluated_instances
            end_slice = start_slice + inputs.shape[0]
            embedding[start_slice:end_slice] = l1
            evaluated_instances = end_slice
        return embedding
    
    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        m = unlabeled_embeddings.shape[0]
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.tile(float("inf"), m)
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]
                
        idxs = []
        
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
            min_dist = torch.min(min_dist, dist_new_ctr[:,0])
                
        return idxs
    
    def getcoreset(self, pool_queue, label_queue, model, device, criterion, param_dim=7):
        if param_dim == 7 or param_dim == 6:
            embed_unlabelled = self.getParamEmbed(pool_queue, param_dim)
            embed_labelled = self.getParamEmbed(label_queue, param_dim)
        else:
            embed_unlabelled = self.getEmbed(pool_queue, model, device, criterion)
            embed_labelled = self.getEmbed(label_queue, model, device, criterion)
        print(embed_unlabelled.shape, embed_labelled.shape)
        # print(embed_unlabelled)
        # print(embed_labelled)
        # print(embed_labelled.shape, embed_unlabelled.shape)
        print("Gradient Embedding obtained. Initiating Grouping...")
        chosen = self.furthest_first(embed_unlabelled, embed_labelled, 1000)
        # print(np.array(chosen).shape, chosen); exit()
        print("Grouping Complete.", len(chosen))
        return chosen

    def get_tod_uncertainty(self, diff_solver, cod_model, pool_loader):
        diff_solver.eval()
        cod_model.eval()
        uncertainty = torch.tensor([]).cuda()
        with torch.no_grad():
            for (i, data) in enumerate(pool_loader):
                x = data[0].to(self.device)
                print(x.shape)
                y = data[1].to(self.device)
                yhat = diff_solver(x)
                yhat_cod = cod_model(x)

                pred_loss = (yhat - yhat_cod).pow(2).sum(dim=(1, 2, 3)) / 2   
                uncertainty = torch.cat((uncertainty, pred_loss), dim=0)

        return uncertainty.cpu()
                

    def train_diff_solver_al(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=500, snap=25,
                          dataset_name='TwoSourcesRdm', transformation='linear', args=None):
        from datetime import datetime
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        # save_loc = os.path.join(save_loc, args.net, "debug")#datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # save_loc = os.path.join(save_loc, args.net, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        save_loc = os.path.join(save_loc, args.net, "entropy")
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        print('Save location:', save_loc)

        # diff_solver = util.NNets.SimpleClas().to(self.device)
        
        
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
        
        init_size = 1000
        pool_size = len(train_data) - init_size
        init_indices = all_indices[:init_size]
        pool_indices = all_indices[init_size:]

        # disp_log_msg("Dataset Split for AL", "Init size: {}, Pool size: {}, Train size: {}, Repeated Indices: {}".format(len(init_indices), len(init_indices), len(init_indices), check_uniq_indices(init_indices, pool_indices)))
        labelled_idx = deepcopy(init_indices)
        init_dataset = Subset(train_data, labelled_idx)
        pool_dataset = Subset(train_data, pool_indices)
        # pool_dataset, valid_dataset, init_dataset = torch.utils.data.random_split(train_data, [pool_size, valid_size, init_size])
        
        train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                shuffle=True, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,
                                                shuffle=False, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=16,
                                                shuffle=False, num_workers=NUM_WORKERS)
        pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=32,
                                                shuffle=False, num_workers=NUM_WORKERS)
        name_plus, name_minus = util.get_Nplus_Nminus(dataset_name)

        # self.getParamEmbed(test_loader)
        # exit(0)

        test_loader_plus, test_loader_minus = None, None
        tester_arr = []
        tester_arr_by_count = []
        tester_arr_by_batch = []
        
        # raw_diff_solver = util.NNets.SimpleCNN100()
        for portion in range(1000, 17000, 1000):
            temp_tester = 0
            temp_tester_by_batch = []
            temp_tester_by_count = []
            ada = 0
            if ada == 1:
                diff_solver = util.NNets.AdaptiveConvNet(1, 1, 1, 3, args, self.device).to(self.device)
            else:
                if args.net == "simplecnn":
                    diff_solver = util.NNets.SimpleCNN100().to(self.device)
                    if args.func == "tod":
                        cod_model = util.NNets.SimpleCNN100().to(self.device)
                        for param in cod_model.parameters():
                            param.detach_()
                    
                elif args.net == "unetab":
                    diff_solver = util.NNets.UNetAB().to(self.device)
                    if args.func == "tod":
                        cod_model = util.NNets.UNetAB().to(self.device)

                        for param in cod_model.parameters():
                            param.detach_()

            print(diff_solver)

            if portion > 1000 and args.func == "tod":
                print("Loaded TOD model: {}".format(save_loc + "/diffusion-model-{}.pt".format(portion - 1000)))
                load_cod_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion - 1000))
                cod_model.load_state_dict(load_cod_model['model_state_dict'])
            else:
                print("No here", portion, "here")

            criterion = self.loss
            # selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion)
            # exit(0)
            
            opt = optim.Adam(diff_solver.parameters(), lr=lr)
            
            error_list = []
            test_error, test_error_plus, test_error_minus = [], [], []
            test_error_by_batch, test_error_by_count = [], []
            val_error = []
            # acc, accTrain = [], []

            # todo: add plots back
            best_loss = np.Inf
            early_stop = 0
            patience = 50
            if portion > 12000:
                for epoch in range(epochs):
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

                    # if args.by_dist == 1:
                    #     test_err_by_batch , test_err_by_count = self.test_diff_error_by_dist(diff_solver, test_loader, self.my_loss_nomean, self.device, ada=ada)
                    #     test_error_by_batch.append(test_err_by_batch) 
                    #     test_error_by_count.append(test_err_by_count)
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

                    ve = self.test_diff_error(diff_solver, val_loader, criterion, self.device, ada=0)
                    val_error.append(ve)
                    with open(os.path.join(save_loc, 'valid_error{}.csv'.format(portion)), 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerow(val_error)
                    # writer_e.add_scalar('Loss/valid', ve, epoch)
                    

                    if args.es == 1:
                        if ve < best_loss:
                            print("Validation loss reduced {} --> {}. Saving model ...".format(best_loss, ve))
                            best_loss = ve
                            # save_model(save_loc, diff_solver, opt, error_list, epoch)
                            save_model(save_loc, diff_solver, opt, error_list, epoch, model_name="diffusion-model-{}.pt".format(portion))
                            # early_stop = 0
                            temp_tester = test_error[-1]
                            # if args.by_dist == 1:
                            #     temp_tester_by_batch = test_error_by_batch[-1]
                            #     temp_tester_by_count = test_error_by_count[-1]
                        # else:
                        #     early_stop += 1
                        #     if early_stop >= patience:
                        #         print("Patience broken. No improvement in training")
                        #         break
                    else:
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
                tester_arr.append(temp_tester)
                # if args.by_dist == 1:
                #     tester_arr_by_batch.append(temp_tester_by_batch)
                #     tester_arr_by_count.append(temp_tester_by_count)
                #     print("Tester by batch", tester_arr_by_batch)
                #     print("Tester by count", tester_arr_by_count)

                print(tester_arr)
                
                # if portion == 3000:
                    
                    # exit()
                plotter_c = myPlots(adaptive=0)
                plotter_c.plotDiff(save_loc, "diff_plots", self.device, error_list, test_error, test_loader, diff_solver, epoch=portion)
                print('TRAINING FINISHED')

            if portion == 16000:
                break

            if args.func == "random":
                k_indices = np.arange(len(pool_indices))
                r.shuffle(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16,
                                                    shuffle=True, num_workers=NUM_WORKERS)
            elif args.func == "entropy":
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                print("Getting losses and preds for {} acquisition".format(args.sub))
                _, pred_arr, lossess, _, _ = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 3)
                # print(lossess.shape, pred_arr.shape); exit(0)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                print(metric.shape, pred_arr.shape, mean_arr.shape)
                # x1, x2, y1, y2 = src_cds[:, 0] - 10, src_cds[:, 0] + 10, src_cds[:, 1] - 10, src_cds[:, 1] + 10
                # xx1, xx2, yy1, yy2 = snk_cds[:, 0] - 10, snk_cds[:, 0] + 10, snk_cds[:, 1] - 10, snk_cds[:, 1] + 10
               
                # x1[x1 < 0] = 0
                # x2[x2 > 100] = 100
                # y1[y1 < 0] = 0
                # y2[y2 > 100] = 100

                # xx1[xx1 < 0] = 0
                # xx2[xx2 > 100] = 100
                # yy1[yy1 < 0] = 0
                # yy2[yy2 > 100] = 100


                # metrics_arr = []
                # for m_counter in range(metric.shape[1]):
                #     metric_src = torch.mean(metric[:, m_counter, :, x1[m_counter]:x2[m_counter], y1[m_counter]:y2[m_counter]])
                #     metric_snk = torch.mean(metric[:, m_counter, :, xx1[m_counter]:xx2[m_counter], yy1[m_counter]:yy2[m_counter]])
                #     metrics_arr.append(0.5 * (metric_src + metric_snk))
                # print("Calculation of entropy done")
                # metric = torch.tensor(metrics_arr)

                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                # print(metric[0])
                # exit()
                if args.sub == "std":
                    k_indices = torch.argsort(metric, descending=True).numpy()
                elif args.sub == "loss":
                    k_indices = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

                del pred_arr, lossess
            elif args.func == "combine_entropy":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                true_arr, pred_arr, lossess = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                k_indices_std = torch.argsort(metric, descending=True).numpy()
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                c = np.empty((k_indices_std.size + k_indices_loss.size,), dtype=k_indices_std.dtype)
                c[0::2] = k_indices_std
                c[1::2] = k_indices_loss
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                plt.scatter(metric, lossess, color="blue", marker="o")
                plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "combine_all":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion)
                
                true_arr, pred_arr, lossess = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                metric = self.custom_loss(torch.from_numpy(pred_arr), torch.from_numpy(mean_arr))
                metric = torch.mean(metric, dim=(0, 2, 3, 4))
                k_indices_std = torch.argsort(metric, descending=True).numpy()
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                
                c = np.empty((k_indices_std[:1000].size + k_indices_loss[:1000].size + len(selected_indices),), dtype=k_indices_std.dtype)
                c[0::3] = selected_indices
                c[1::3] = k_indices_loss[:1000]
                c[2::3] = k_indices_std[:1000]
                # print(np.array(c).shape, c); exit(0)
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                plt.scatter(metric, lossess, color="blue", marker="o")
                plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "diversity":
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion, args.param_size)
                select_indices = pool_indices[selected_indices]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, selected_indices)
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
            elif args.func == "combine_nodrop":
                print("Combining loss and p-diversity")
                # load_saved_model = torch.load(save_loc + "/diffusion-model.pt")
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                selected_indices = self.getcoreset(pool_loader, train_loader, diff_solver, self.device, criterion, args.param_size)

                true_arr, pred_arr, lossess, _, _ = self.eval_mode_prediction(diff_solver, pool_loader, self.device, self.my_loss_nomean, True, 5)
                mean_arr = np.mean(pred_arr, axis=0)
                k_indices_loss = torch.argsort(torch.from_numpy(lossess).squeeze(1), descending=True).numpy()
                c = np.empty((k_indices_loss[:1000].size + len(selected_indices),), dtype=k_indices_loss.dtype)
                c[0::2] = selected_indices
                c[1::2] = k_indices_loss[:1000]
                k_indices = []
                for item in c:
                    if item not in k_indices:
                        k_indices.append(item)
                k_indices = np.array(k_indices)
                select_indices = pool_indices[k_indices[:1000]]
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
                plt.close("all"); plt.figure(figsize=(15, 15))
                # plt.scatter(metric, lossess, color="blue", marker="o")
                # plt.scatter(metric[k_indices[:1000]], lossess[k_indices[:1000]], color="magenta", marker="x")
                plt.savefig(save_loc + "/metric_vs_lossess_{}.png".format(portion))
            elif args.func == "tod":
                load_saved_model = torch.load(save_loc + "/diffusion-model-{}.pt".format(portion))
                diff_solver.load_state_dict(load_saved_model['model_state_dict'])
                uncertainty = self.get_tod_uncertainty(diff_solver, cod_model, pool_loader)

                k_indices = torch.argsort(uncertainty, descending=True).numpy()
                select_indices = pool_indices[k_indices[:1000]]
                
                labelled_idx = np.append(labelled_idx, select_indices)
                pool_indices = np.delete(pool_indices, k_indices[:1000])
                
                init_dataset = Subset(train_data, labelled_idx)
                pool_dataset = Subset(train_data, pool_indices)
                
                train_loader = torch.utils.data.DataLoader(init_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
                pool_loader = torch.utils.data.DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

            del diff_solver, opt
            if args.func == "tod":
                del cod_model
            print(len(train_loader.dataset))
        
        np.save(save_loc + "/tester_by_batch", tester_arr_by_batch)
        np.save(save_loc + "/tester_by_count", tester_arr_by_count)

        with open(os.path.join(save_loc, 'tester_arr.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(tester_arr)
        
        return error_list, test_error, test_error_plus, test_error_minus, save_loc


    def train_diff_solver_raw(self, load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS, epochs=100, snap=25,
                          dataset_name='TwoSourcesRdm', transformation='linear', args=None):
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        # save_loc = os.path.join(save_loc, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        save_loc = os.path.join(save_loc, args.mname)
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
            
        save_loc = os.path.join(save_loc, args.exp)
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
            
        print('Save location:', save_loc)

        # diff_solver = util.NNets.SimpleClas().to(self.device)
        diff_solver = util.NNets.UNet().to(self.device)
        criterion = self.loss

        opt = optim.Adam(diff_solver.parameters(), lr=lr)

        # Data loaders
        print(BATCH_SIZE)
        train_data, test_data = util.loaders.generate500Datasets(PATH=load_loc,
                                                                  datasetName=dataset_name,
                                                                  batch_size=BATCH_SIZE,
                                                                  num_workers=NUM_WORKERS,
                                                                  std_tr=self.std_tr,
                                                                  s=self.s,
                                                                  transformation=transformation).getDatasets()
        valid_size = 0#int(0.1 * len(train_data))
        train_size = 1
        # pool_size = len(train_data) - valid_size - init_size
        all_indices = np.arange(int(len(train_data) * 1))
        np.random.shuffle(all_indices)
        train_data = Subset(train_data, all_indices)
        
        # disp_log_msg("All Data size: {}", "Considered train size: {}".format(len(train_data), len(all_indices)))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,
                                                  shuffle=True, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16,
                                                  shuffle=False, num_workers=NUM_WORKERS)

        test_loader_plus, test_loader_minus = None, None
        print(len(train_loader.dataset), len(test_loader.dataset))
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
        plotter_c = myPlots(adaptive=0)
        # acc, accTrain = [], []

        # todo: add plots back

        for epoch in range(epochs):
            st_time = time.time()
            print("Epoch", epoch)
            error = 0.0
            for (i, data) in enumerate(train_loader):
                diff_solver.zero_grad()
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                yhat = diff_solver(x)
                err = criterion(yhat, y)
                err.backward()
                opt.step()
                error += err.item()
            error_list.append(error / (i + 1))

            test_error.append(self.test_diff_error(diff_solver, test_loader, criterion, self.device, ada=0))
            if test_loader_plus is not None:
                test_error_plus.append(self.test_diff_error(diff_solver, test_loader_plus, criterion, self.device))
            if test_loader_minus is not None:
                test_error_minus.append(self.test_diff_error(diff_solver, test_loader_minus, criterion, self.device))

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
            if epoch % snap == snap - 1:
                save_model(save_loc, diff_solver, opt, error_list, epoch)
                
            et_time = time.time()
            print("Epoch time: {}".format(et_time - st_time))
            plotter_c.plotDiff(save_loc, "diff_plots", self.device, error_list, test_error, test_loader, diff_solver, epoch=str(epoch))
        print('TRAINING FINISHED')
        return error_list, test_error, test_error_plus, test_error_minus, save_loc