#!/usr/bin/env python

print("Start at the beginning of training!")
import io, os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import copy
import h5py
import argparse
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import model as mdp
import util
import compute_kernel as ker

#The only difference between this one and train.py is that, it removes the processing of study set, so that study processing can start after training is done

#----------------------Parser settings---------------------------
print("Start setting parser!", flush=True)

parser = argparse.ArgumentParser(description='Exalearn_Training_v1')

parser.add_argument('--batch_size',     type=int,   default=2048,
                    help='input batch size for training (default: 2048)')
parser.add_argument('--epochs',         type=int,   default=1000,
                    help='number of epochs to train, save the best model instead of the model at last epoch (default: 1000)')
parser.add_argument('--lr',             type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random seed (default: 42)')
parser.add_argument('--log_interval',   type=int,   default=1,
                    help='how many batches to wait before logging training status')
parser.add_argument('--device',         default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--ntrain',         type=float, default=2000,
                    help='number of training samples (default: 2000)')
parser.add_argument('--ntest',          type=float, default=1000,
                    help='number of test samples (default: 500)')
parser.add_argument('--num_workers',    type=int,   default=1, 
                    help='set the number of op workers. only work for gpu')
parser.add_argument('--num_threads',    type=int,   default=8, 
                    help='set the number of threads per rank. should be 32 (1 rank), 8 (4 rank), 2 (4 rank with async)')
parser.add_argument('--phase_idx',      type=int, required=True,
                    help='which AL phase we are in. This is one-indexed! In other word, AL start with 1, no AL is 0')
parser.add_argument('--data_dir',       type=str,   default='./',
                    help='root directory of base/test/study/AL subdir')
parser.add_argument('--shared_file_dir',    type=str,   required=True,
                    help='a directory which saves sharedfile for DDP. It must be empty before running this script')


args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )

if args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if not args.cuda:
    torch.use_deterministic_algorithms(True)

def print_memory_usage(info):
    util.print_memory_usage_template("train_step2, phase_idx = {}".format(args.phase_idx), info)

print_memory_usage("All ranks should report Starting!")

#--------------------DDP initialization-------------------------

size = int(os.getenv("PMI_SIZE"))
rank = int(os.getenv("PMI_RANK"))
local_rank = int(os.getenv("PMI_LOCAL_RANK"))
print("DDP: I am worker size = {}, rank = {}, local_rank = {}".format(size, rank, local_rank))

# Pytorch will look for these:
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(size)

if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

def print_from_rank0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def print_memory_usage_from_rank0(info):
    if rank == 0:
        print_memory_usage(info)

print_from_rank0("args = ", args)
print_from_rank0("backend = ", backend)

torch.distributed.init_process_group(backend=backend, init_method='file://{}/sharedfile'.format(args.shared_file_dir), world_size=size, rank=rank)
print("Setting process group, is_initialized = {}, nccl_avail = {}, get_rank = {}, get_size = {}".format(torch.distributed.is_initialized(), torch.distributed.is_nccl_available(), torch.distributed.get_rank(), torch.distributed.get_world_size()))

if args.cuda:
    # DDP: pin GPU to local rank.
    print("rank = {}, local_rank = {}, num_of_gpus = {}".format(rank, local_rank, torch.cuda.device_count()))

    # Handles the case where we pinned GPU to local rank in run script
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
    else:
#        torch.cuda.set_device(int(local_rank))
        torch.cuda.set_device(torch.cuda.device_count() - 1 - int(local_rank)) # handles Polaris NUMA topology

if (not args.cuda) and (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)
print("Rank = {}".format(rank), " Torch Thread setup with number of threads: ", torch.get_num_threads(), " with number of inter_op threads: ", torch.get_num_interop_threads())

#-----------------------------Loading data--------------------------------
print_memory_usage_from_rank0("Before loading data!")

base_cubic_file       = os.path.join(args.data_dir, "base/data/cubic_1001460_cubic.hdf5")
base_trigonal_file    = os.path.join(args.data_dir, "base/data/trigonal_1522004_trigonal.hdf5")
base_tetragonal_file  = os.path.join(args.data_dir, "base/data/tetragonal_1531431_tetragonal.hdf5")
test_cubic_file       = os.path.join(args.data_dir, "test/data/cubic_1001460_cubic.hdf5")
test_trigonal_file    = os.path.join(args.data_dir, "test/data/trigonal_1522004_trigonal.hdf5")
test_tetragonal_file  = os.path.join(args.data_dir, "test/data/tetragonal_1531431_tetragonal.hdf5")

x_train,  y_train  = util.create_numpy_data(base_cubic_file, base_trigonal_file,  base_tetragonal_file)
x_test,  y_test  = util.create_numpy_data(test_cubic_file, test_trigonal_file,  test_tetragonal_file)

print_from_rank0("x_train.shape = ", x_train.shape)
print_from_rank0("y_train.shape = ", y_train.shape)
print_from_rank0("x_test.shape = ", x_test.shape)
print_from_rank0("y_test.shape = ", y_test.shape)

print_memory_usage_from_rank0("Finish loading data!")

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

x_train_torch = torch.from_numpy(x_train).float()
x_train_torch = x_train_torch.reshape((x_train_torch.shape[0], 1, x_train_torch.shape[1]))
y_train_torch = torch.from_numpy(y_train).float()
if args.phase_idx == 0:
    train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=size, rank=rank, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, **kwargs)
print_from_rank0("x_train_torch.shape = ", x_train_torch.shape)
print_from_rank0("y_train_torch.shape = ", y_train_torch.shape)

x_test_torch = torch.from_numpy(x_test).float()
x_test_torch = x_test_torch.reshape((x_test_torch.shape[0], 1, x_test_torch.shape[1]))
y_test_torch = torch.from_numpy(y_test).float()
test_dataset = torch.utils.data.TensorDataset(x_test_torch, y_test_torch)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=size, rank=rank, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, sampler=test_sampler, drop_last=True, **kwargs)
print_from_rank0("x_test_torch.shape = ", x_test_torch.shape)
print_from_rank0("y_test_torch.shape = ", y_test_torch.shape)

print_memory_usage_from_rank0("Finish creating torch dataset and loader!")

#----------------------------setup model---------------------------------
#Important! FIXME
#Here num_output should be 3+1 instead of 3, since each sample needs one value representing its uncertainty
model = mdp.FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model = model.cuda()
    model = DDP(model)

print_memory_usage_from_rank0("Finish creating model!")

#---------------------------setup optimizer------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * np.sqrt(size))

def criterion_reg(y_pred, y_true):
    y_pred_value = y_pred[:, 0:3].reshape(-1, 3)
    logsig2 = y_pred[:, 3].reshape(-1, 1)
    l2_diff = torch.sum((y_true - y_pred_value) ** 2, axis=1, keepdims=True) / torch.exp(logsig2) + logsig2
    return torch.mean(l2_diff)

#criterion_class = torch.nn.BCEWithLogitsLoss()
criterion_class = torch.nn.CrossEntropyLoss()

def lr_lambda(epoch):
    if epoch <= 5000:
        return 1.0
    elif 5001 <= epoch <= 10000:
        return 0.5
    else:
        return 0.2

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print_memory_usage_from_rank0("finish setting up optimizer!")


#------------------------Load possible previous model and extra AL dataset----------------------------

if args.phase_idx > 0:
    checkpoint = torch.load('ckpt.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print_memory_usage_from_rank0("finish loading ckpt from disk")

    x_AL_list = []
    y_AL_list = []

#In phase_k, we already have base data and AL_1 upto AL_k
    for i in range(1, args.phase_idx + 1):
        AL_cubic_file       = os.path.join(args.data_dir, "AL_phase_{}/data/cubic_1001460_cubic.hdf5".format(i))
        AL_trigonal_file    = os.path.join(args.data_dir, "AL_phase_{}/data/trigonal_1522004_trigonal.hdf5".format(i))
        AL_tetragonal_file  = os.path.join(args.data_dir, "AL_phase_{}/data/tetragonal_1531431_tetragonal.hdf5".format(i))
        x_AL_temp, y_AL_temp = util.create_numpy_data(AL_cubic_file, AL_trigonal_file, AL_tetragonal_file)
        x_AL_list.append(x_AL_temp)
        y_AL_list.append(y_AL_temp)

    for i in range(len(x_AL_list)):
        x_train_torch = torch.cat((x_train_torch, torch.from_numpy(x_AL_list[i]).float().reshape((x_AL_list[i].shape[0], 1, x_AL_list[i].shape[1]))), axis=0)
        y_train_torch = torch.cat((y_train_torch, torch.from_numpy(y_AL_list[i]).float()), axis=0)
    
    train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=size, rank=rank, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, **kwargs)

print_from_rank0("x_train_torch.shape = ", x_train_torch.shape)
print_from_rank0("y_train_torch.shape = ", y_train_torch.shape)

print_memory_usage_from_rank0("finish loading AL data")

for param_group in optimizer.param_groups:
    param_group['lr'] = args.lr
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#------------------------------start training----------------------------------

train_loss_list = []
test_loss_list = []

time_tot = time.time()
best_loss = 99999999.9
best_epoch = -1

for epoch in range(0, args.epochs):

    ker.train(epoch, rank, size,
          model = model,
          optimizer = optimizer,
          train_loader = train_loader,
          train_sampler = train_sampler,
          criterion_reg = criterion_reg, criterion_class = criterion_class,
          lr_scheduler = scheduler,
          on_gpu = args.cuda,
          log_interval = args.log_interval,
          loss_list = train_loss_list)

    test_loss = ker.test(epoch, rank, size,
                     model = model, 
                     test_loader = test_loader,
                     criterion_reg = criterion_reg, criterion_class = criterion_class,
                     on_gpu = args.cuda,
                     log_interval = args.log_interval,
                     loss_list = test_loss_list)

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        if rank == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
        print_from_rank0("Better model at epoch ", epoch)

if rank == 0:
    torch.save(checkpoint, 'ckpt.pth')
print_from_rank0("Best val loss = {} at epoch = {}".format(best_loss, best_epoch))
time_tot = time.time() - time_tot
print_from_rank0("Total training time = {}".format(time_tot))

print_memory_usage_from_rank0("finish first training part")

st = time.time()
model = mdp.FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
if args.cuda:
    model = model.cuda()
    model = DDP(model)
checkpoint = torch.load('ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
criterion_l2 = torch.nn.MSELoss()
print_memory_usage_from_rank0("finish loading from disk for validation")

l2_diff, sigma2, class_loss = ker.validation(rank, size, 
                                             model = model, 
                                             test_loader = test_loader,
                                             criterion_reg = criterion_l2, criterion_class = criterion_class,
                                             on_gpu = args.cuda)

print_from_rank0("Final validation time = {}".format(time.time() - st))

torch.distributed.destroy_process_group()
