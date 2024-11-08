import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import psutil

def print_memory_usage_template(caller, info):
    print(caller, " logging: ", info)
    print(torch.cuda.memory_summary())

    memory_info = psutil.virtual_memory()
    print(f"CPU Memory Usage: {memory_info.used / (1024 ** 3):.2f} GB / {memory_info.total / (1024 ** 3):.2f} GB")


def create_numpy_data(file_cubic, file_trigonal, file_tetragonal):
    with h5py.File(file_cubic, 'r') as f:
        dhisto = f['histograms']
        x_cubic = dhisto[:, 1, :]
        x_shape = x_cubic.shape
        dparams = f['parameters']
        y_cubic = dparams[:]
        y_shape = y_cubic.shape
        print(x_shape)
        print(y_shape)
    with h5py.File(file_trigonal, 'r') as f:
        dhisto = f['histograms']
        x_trigonal = dhisto[:, 1, :]
        x_shape = x_trigonal.shape
        dparams = f['parameters']
        y_trigonal = dparams[:]
        y_shape = y_trigonal.shape
        print(x_shape)
        print(y_shape)
    with h5py.File(file_tetragonal, 'r') as f:
        dhisto = f['histograms']
        x_tetragonal = dhisto[:, 1, :]
        x_shape = x_tetragonal.shape
        dparams = f['parameters']
        y_tetragonal = dparams[:]
        y_shape = y_tetragonal.shape
        print(x_shape)
        print(y_shape)
    
    x = np.concatenate([x_cubic, x_trigonal, x_tetragonal], axis=0)
    scaler_x = MinMaxScaler(copy=True)
    x = scaler_x.fit_transform(x.T).T

    y = np.concatenate([y_cubic, y_trigonal, y_tetragonal], axis=0)
    y[:,0] = (y[:,0] - 3.5 )  / 1.0
    y[:,1] = (y[:,1] - 3.5 )  / 1.0
    y[:,2] = (y[:,2] - 30.0 ) / 90.0

    return x, y

class EarlyStopping:
    def __init__(self, max_num, min_delta):
        self.max_num = max_num
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 99999999.9
        self.do_stop = False
        self.improve = False

    def __call__(self, loss):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.improve = True
        else:
            self.counter += 1
            if self.counter > self.max_num:
                self.do_stop = True
            self.improve = False
