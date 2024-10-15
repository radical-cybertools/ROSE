#!/usr/bin/env python

import os,sys
sys.path.insert(0,os.path.expanduser("~/g2full_polaris/GSASII/"))
import GSASIIscriptable as G2sc

from mpi4py import MPI
import numpy as np
import time
import h5py
from contextlib import redirect_stdout      #for debug

import sweep_utils as su

gpx = []
phase = []


def cubic_lattice(prm):
    """ This function uses a 1D grid.

        Parameters for cubic lattice:
        a = b = c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def trigonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for trigonal lattice:
        a = b = c
        alpha = beta = gamma != 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = prm[1] # alpha
    phase['General']['Cell'][5] = prm[1] # beta
    phase['General']['Cell'][6] = prm[1] # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def tetragonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for tetragonal lattice:
        a = b != c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    # c should be differe
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[1] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y


def _sim_impl(rank, size, sym, conffile_name, sample):
    start = time.time()

    gParameters = su.read_config_file(conffile_name)
    # Create project
    name = gParameters['name'] +'_rank' + str(rank)
    path_in = gParameters['path_in']
    path_out = gParameters['path_out']
    name_out = gParameters['name_out']
    global gpx
    gpx = G2sc.G2Project(newgpx=path_out+name+'.gpx')
    
    # Add phase: Requires CIF file
    cif = path_in + gParameters['cif']
    global phase
    phase = gpx.add_phase(cif,phasename=name,fmthint='CIF')
    
    # Get instrument file specification
    instprm = path_in + gParameters['instprm']
    # Histogram range
    Tmin = gParameters['tmin']
    Tmax = gParameters['tmax']
    Tstep = gParameters['tstep']
    hist = gpx.add_simulated_powder_histogram(name+'TOFsimulation',instprm,Tmin,Tmax,Tstep,phases=gpx.phases())
    hist.SampleParameters['Scale'][0] = 1000.
    # Set to no-background
    hist['Background'][0][3]=0.0
    
    symmetry = gParameters['symmetry']
    assert symmetry == sym, "symmetry = {} and sym = {}".format(symmetry, sym)
    
    # Configure sweep according to symmetry
    if symmetry == 'cubic':
        sweepf_ = cubic_lattice
    elif symmetry == 'trigonal':
        sweepf_ = trigonal_lattice
    elif symmetry == 'tetragonal':
        sweepf_ = tetragonal_lattice
    else:
        exit("Do not recognize symmetry of {}".format(symmetry))
    
    # Distribute computation
    nsim, histosz = su.grid_sample(rank, size, sweepf_, sample, path_out, name_out + '_' + symmetry)
    end = time.time()
    print('----------------------------------------------------------')
    print("Rank = {}, Number of simulations ({}): {}, size of histogram: {}, cost {} seconds".format(rank, symmetry, nsim, histosz, end-start))


def main():

    start = time.time()
    if ( len ( sys.argv ) != 6 ) :
        print(sys.argv)
        sys.stderr.write("Usage: python simulation_resample_real_work.py "
                         "conf_name_cubic conf_name_trigonal conf_name_tetragonal "
                         "which_half first_half_percent\n")
        sys.exit(0)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Rank = {} out of size = {}", rank, size)

    conf_name_cubic = sys.argv[1]
    conf_name_trigonal = sys.argv[2]
    conf_name_tetragonal = sys.argv[3]
    which_half = sys.argv[4]
    first_half_percent = float(sys.argv[5])

    sample_cubic      = np.load('sample_cubic_rank_{}.npy'.format(rank))
    sample_trigonal   = np.load('sample_trigonal_rank_{}.npy'.format(rank))
    sample_tetragonal = np.load('sample_tetragonal_rank_{}.npy'.format(rank))

    split_index_cubic      = int(sample_cubic.shape[0] * first_half_percent)
    split_index_trigonal   = int(sample_trigonal.shape[0] * first_half_percent)
    split_index_tetragonal = int(sample_tetragonal.shape[0] * first_half_percent)

    if which_half == "first":
        sample_cubic      = sample_cubic[:split_index_cubic,:]
        sample_trigonal   = sample_trigonal[:split_index_trigonal,:]
        sample_tetragonal = sample_tetragonal[:split_index_tetragonal,:]
    elif which_half == "second":
        sample_cubic      = sample_cubic[split_index_cubic:,:]
        sample_trigonal   = sample_trigonal[split_index_trigonal:,:]
        sample_tetragonal = sample_tetragonal[split_index_tetragonal:,:]

    _sim_impl(rank, size, "cubic", conf_name_cubic, sample_cubic)
    _sim_impl(rank, size, "trigonal", conf_name_trigonal, sample_trigonal)
    _sim_impl(rank, size, "tetragonal", conf_name_tetragonal, sample_tetragonal)


if __name__ == '__main__':
    main()

