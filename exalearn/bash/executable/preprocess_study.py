#!/usr/bin/env python

import io, os, sys
import argparse
import numpy as np
import torch

import util


parser = argparse.ArgumentParser(description='Exalearn_preprocess_study_set')
parser.add_argument('--data_dir',       type=str,   default='./',
                    help='root directory of base/test/study/AL subdir')

args = parser.parse_args()

study_cubic_file      = os.path.join(args.data_dir, "study/data/cubic_1001460_cubic.hdf5")
study_trigonal_file   = os.path.join(args.data_dir, "study/data/trigonal_1522004_trigonal.hdf5")
study_tetragonal_file = os.path.join(args.data_dir, "study/data/tetragonal_1531431_tetragonal.hdf5")

x_study, y_study = util.create_numpy_data(study_cubic_file, study_trigonal_file, study_tetragonal_file)
print("x_study.shape = ", x_study.shape)
print("y_study.shape = ", y_study.shape)

x_study_torch = torch.from_numpy(x_study).float()
x_study_torch = x_study_torch.reshape((x_study_torch.shape[0], 1, x_study_torch.shape[1]))
y_study_torch = torch.from_numpy(y_study).float()
print("x_study_torch.shape = ", x_study_torch.shape)
print("y_study_torch.shape = ", y_study_torch.shape)
torch.save(x_study_torch, "x_study_torch.pt")

