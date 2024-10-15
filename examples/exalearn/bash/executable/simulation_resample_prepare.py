#!/usr/bin/env python

import os,sys
sys.path.insert(0,os.path.expanduser("~/g2full_polaris/GSASII/"))
import GSASIIscriptable as G2sc

from mpi4py import MPI
import numpy as np
import time
import h5py
from contextlib import redirect_stdout      #for debug


def _generate_random_uniform(sz, low, high):
    random_numbers = np.random.uniform(low, high, sz)
    return random_numbers.reshape(-1, 1)

def generate_cubic_random_sample(sz, a_min, a_max):
    a_arr = _generate_random_uniform(sz, a_min, a_max)
    return a_arr

def generate_trigonal_random_sample(sz, a_min, a_max, alpha_min, alpha_max):
    a_arr = _generate_random_uniform(sz, a_min, a_max)
    alpha_arr = _generate_random_uniform(sz, alpha_min, alpha_max)
    return np.column_stack((a_arr, alpha_arr))

def generate_tetragonal_random_sample(sz, a_min, a_max, c_min, c_max):
    a_arr = _generate_random_uniform(sz, a_min, a_max)
    c_arr = _generate_random_uniform(sz, c_min, c_max)
    return np.column_stack((a_arr, c_arr))

#Here the input y_study is mixed which has [a, c, alpha, class]
#Here freq is also mixed based on y_study
#Here samples we generate will be separated for better comm with _remove_out_of_box 
def generate_gaussian_sample_separate(y_study, freq, std_dev_cubic, std_dev_trigonal, std_dev_tetragonal, do_print = False):
    samples_cubic = []
    samples_trigonal = []
    samples_tetragonal = []
    for i in range(len(freq)):
        if freq[i] > 0:
            if do_print:
                print("i = ", i, " freq[i] = ", freq[i], " y_study[i] = ", y_study[i], '\n')
            if y_study[i, 3] == 0:  #trigonal
                for _ in range(freq[i]):
                    sample = np.random.normal(loc=[y_study[i, 0], y_study[i, 2]], scale=std_dev_trigonal)
                    if do_print:
                        print("trigonal sample = ", sample, '\n')
                    samples_trigonal.append(sample)
            elif y_study[i, 3] == 1:    #tetragonal
                for _ in range(freq[i]):
                    sample = np.random.normal(loc=[y_study[i, 0], y_study[i, 1]], scale=std_dev_tetragonal)
                    if do_print:
                        print("tetragonal sample = ", sample, '\n')
                    samples_tetragonal.append(sample)
            elif y_study[i, 3] == 2:    #cubic
                for _ in range(freq[i]):
                    sample = np.random.normal(loc=[y_study[i, 0]], scale=std_dev_cubic)
                    if do_print:
                        print("cubic sample = ", sample, '\n')
                    samples_cubic.append(sample)

    return np.vstack(samples_cubic), np.vstack(samples_trigonal), np.vstack(samples_tetragonal)

def _remove_out_of_box(sym, sample_in, bounding_box, min_diff):
    assert bounding_box.shape[1] == 2       #[[begin1, end1], [begin2, end2], [begin3, end3], ...]
    if sym == "cubic":
        assert sample_in.shape[1] == 1
        assert bounding_box.shape[0] == 1   #on dim a
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1])
        sample_in = sample_in[idx]
    elif sym == "trigonal":
        assert sample_in.shape[1] == 2
        assert bounding_box.shape[0] == 2   #on dim a, angle
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1]) & \
              (sample_in[:,1] >= bounding_box[1,0]) & (sample_in[:,1] <= bounding_box[1,1]) & \
              (abs(sample_in[:,1] - 90) > min_diff)   #angle different from 90 degree
        sample_in = sample_in[idx]
    elif sym == "tetragonal":
        assert sample_in.shape[1] == 2
        assert bounding_box.shape[0] == 2   #on dim a, c
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1]) & \
              (sample_in[:,1] >= bounding_box[1,0]) & (sample_in[:,1] <= bounding_box[1,1]) & \
              (abs(sample_in[:,0] - sample_in[:,1]) > min_diff)  #two edge should be different
        sample_in = sample_in[idx]
    else:
        exit("Error! Unrecognized sym argument = {} in _remove_out_of_box function call".format(sym))
    return sample_in

#All rank will execute this function
#Different from other simulation_xxx, it will first generate exactly the same sample list
#Then different rank will simulate different part of it
#Because of that all rank use global seed!
def create_all_samples(rank, size, seed_g,
                       y_study, freq, 
                       cubic_bounding_box, 
                       trigonal_bounding_box, alpha_diff_trigonal,
                       tetragonal_bounding_box, ac_diff_tetragonal):
    np.random.seed(seed_g)
    do_print = (rank == 0)
    print("IMPORTANT: In resample, rank = {}, seed for create_all_samples = {}!!!".format(rank, seed_g))

    sample_cubic, sample_trigonal, sample_tetragonal = generate_gaussian_sample_separate(y_study, freq, 
                                         std_dev_cubic = [0.0001],
                                         std_dev_trigonal = [0.002, 0.3], 
                                         std_dev_tetragonal = [0.002, 0.002],
                                         do_print = do_print)
    sample_cubic      = _remove_out_of_box("cubic", sample_cubic, cubic_bounding_box, None)
    sample_trigonal   = _remove_out_of_box("trigonal", sample_trigonal, trigonal_bounding_box, alpha_diff_trigonal)
    sample_tetragonal = _remove_out_of_box("tetragonal", sample_tetragonal, tetragonal_bounding_box, ac_diff_tetragonal)

    return sample_cubic[rank::size][:], sample_trigonal[rank::size][:], sample_tetragonal[rank::size][:]

def main():

    start = time.time()
    if ( len ( sys.argv ) != 5 ) :
        print(sys.argv)
        sys.stderr.write("Usage: python simulation_resample.py "
                         "global_seed cubic_studyset "
                         "trigonal_studyset tetragonal_studyset\n")
        sys.exit(0)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Rank = {} out of size = {}".format(rank, size))

    global_seed = int(sys.argv[1])
    cubic_studyset = str(sys.argv[2])
    trigonal_studyset = str(sys.argv[3])
    tetragonal_studyset = str(sys.argv[4])

    a_min1 = 2.5
    a_max1 = 5.5
    a_min2 = 3.5
    a_max2 = 4.5
    alpha_min = 30.0
    alpha_max = 119.5
    c_min = 3.5005
    c_max = 4.4995

    cubic_bounding_box = np.array([[a_min1, a_max1]])
    trigonal_bounding_box = np.array([[a_min2, a_max2], [alpha_min, alpha_max]])
    tetragonal_bounding_box = np.array([[a_min2, a_max2], [c_min, c_max]])
    alpha_diff_trigonal = 0.2
    ac_diff_tetragonal = 0.001

    with h5py.File(cubic_studyset, 'r') as f:
        dparams = f['parameters']
        y_study_cubic = dparams[:]
        y_shape_cubic = y_study_cubic.shape
    print("y_shape_cubic = ", y_shape_cubic)

    with h5py.File(trigonal_studyset, 'r') as f:
        dparams = f['parameters']
        y_study_trigonal = dparams[:]
        y_shape_trigonal = y_study_trigonal.shape
    print("y_shape_trigonal = ", y_shape_trigonal)

    with h5py.File(tetragonal_studyset, 'r') as f:
        dparams = f['parameters']
        y_study_tetragonal = dparams[:]
        y_shape_tetragonal = y_study_tetragonal.shape
    print("y_shape_tetragonal = ", y_shape_tetragonal)

    y_study = np.concatenate([y_study_cubic, y_study_trigonal, y_study_tetragonal], axis=0)

    freq = np.load('AL-freq.npy')
    
    sample_cubic, sample_trigonal, sample_tetragonal = create_all_samples(rank, size, global_seed,
                       y_study, freq,  
                       cubic_bounding_box, 
                       trigonal_bounding_box, alpha_diff_trigonal,
                       tetragonal_bounding_box, ac_diff_tetragonal)

    np.save('sample_cubic_rank_{}.npy'.format(rank), sample_cubic)
    np.save('sample_trigonal_rank_{}.npy'.format(rank), sample_trigonal)
    np.save('sample_tetragonal_rank_{}.npy'.format(rank), sample_tetragonal)

if __name__ == '__main__':
    main()

