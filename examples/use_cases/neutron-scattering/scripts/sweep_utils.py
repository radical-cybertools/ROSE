from __future__ import absolute_import

from pprint import pprint

import numpy as np
from mpi4py import MPI
import itertools
import h5py


try:
    import configparser
except ImportError:
    import ConfigParser as configparser


def read_config_file(file):
    """Functionality to read the configue file.

    Parameters
    ----------
    file : string
      Name of configuration file.
      
    Returns
    -------
    fileParams : python dictionary
      The parameters read from the file are stored in a python dictionary as
      (key, value) tuples.
    """

    config=configparser.ConfigParser()
    config.read(file)
    section=config.sections()
    fileParams={}
        
    # parse specified arguments (minimal validation: if arguments
    # are written several times in the file, just the first time
    # will be used)
    for sec in section:
        for k,v in config.items(sec):
            if not k in fileParams:
                fileParams[k] = eval(v)
    
    pprint(fileParams)

    return fileParams


def read_sweep_ranges(pdict):
    """Functionality to extract range values from python dictionary.
    The operation is applied over dictionary elements with key starting
    by the string 'sweep'. The ranges are returned in a dictionary.

    Parameters
    ----------
    pdict : python dictionary
      The dictionary include all the configuration parameters read from
      the configuration file.
      
    Returns
    -------
    ranges : python dictionary
      The parameters that correspond to range specifications are returned as
      (key, tuple_values) tuples. The tuple values for the range are assumed
      to be configured as: (initial, final, step).
    """

    ranges={}
    for key in pdict.keys():
        key_list = key.split('_')
        if key_list[0] == 'sweep':
            key_out = key_list[1] + '_' + key_list[2]
            ranges[key_out] = pdict[key]

    return ranges


######################
# MPI functionality
######################

def _get_rank_limits(comm, arrlen):
    """Determine the chunk of the grid that has to be computed per
    process. The grid has been 'flattened' and has arrlen length. The
    chunk assigned to each process depends on its rank in the MPI
    communicator.

    Parameters
    ----------
    comm : MPI communicator object
      Describes topology of network: number of processes, rank
    arrlen : int
      Number of points in grid search.

    Returns
    -------
    begin : int
      Index, with respect to 'flattened' grid, where the chunk
      for this process starts.
    end : int
      Index, with respect to 'flattened' grid, where the chunk
      for this process ends.
    """

    rank = comm.Get_rank()  # Id of this process
    size = comm.Get_size()  # Total number of processes in communicator
    end = 0
    # The scan should be done with ints, not floats
    ranklen = int(arrlen / size)
    if rank < arrlen % size:
        ranklen += 1
    # Compute upper limit based on the sizes covered by the processes
    # with less rank
    end = comm.scan(sendobj=ranklen, op=MPI.SUM)
    begin = end - ranklen

    return (begin, end)


def grid_sweep(fn, grid, prefix, fnout, comm=None):
    """Perform a grid sweep launching simulation processes for each
    combination of parameters specified. It is assumed that each
    simulation returns an np array. The computation of the simulation
    at the grid points is executed in parallel using MPI. The simulation
    results (together with the launching parameters) are stored in hdf5 files
    independently by each process in the communicator.

    The `mpi4py <https://mpi4py.readthedocs.io>`__ package is
    required for use of the ``mpiutil.grid_search`` function. It is also
    necessary to run Python with the ``mpiexec`` command; for example,
    if ``mpiscript.py`` calls this function, use::

      mpiexec -n 8 python mpiscript.py

    to distribute the grid search over 8 processors.


    Parameters
    ----------
    fn : function
      Function to be simulated. It should take a tuple of parameter
      values as an argument, and return an array of float values.
    grid : tuple of array_like
      A tuple providing an array of sample points for each axis of the
      grid on which the search is to be performed.
    prefix : string
      A string with the path to store results.
    fnout : string
      A string with the filename to store results. Results are
      stored per launched process so the rank and hdf5 extension are
      added to this string to generate the actual output filename.
    comm : MPI communicator object, optional (default None)
      Topology of network (number of processes and rank). If None,
      ``MPI.COMM_WORLD`` is used.

    Returns
    -------
    sprm : ndarray
      Optimal parameter values on each axis. If `fn` is multi-valued,
      `sprm` is a matrix with rows corresponding to parameter values
      and columns corresponding to function values.
    sfvl : float or ndarray
      Optimum function value or values
    fvmx : ndarray
      Function value(s) on search grid
    sidx : tuple of int or tuple of ndarray
      Indices of optimal values on parameter grid
    """

    if comm is None:
        comm = MPI.COMM_WORLD

    fprm = itertools.product(*grid)
    rank = comm.Get_rank()
    print("TW: rank = {}, print fprm!\n".format(rank), fprm)

    # Distribute computation among processes in MPI communicator
    afprm = np.asarray(list(fprm))  # Faster to communicate array data
    print("TW: rank = {}, print afprm!\n".format(rank), afprm)

    print("TW: rank = ", rank, " shape of afprm = ", afprm.shape)
    iterlen = afprm.shape[0]
    begin, end = _get_rank_limits(comm, iterlen)
    rankgrid = (afprm[begin:end, :])
    if rank == 0:
        print(type(rankgrid))
        print("rankgrid = ", rankgrid)
    rankfval = np.asarray(list(map(fn, rankgrid)))
    print("TW: rank = ", rank, " iterlen = ", iterlen, " begin = ", begin, " end = ", end)
    print("TW: shape of rankfval = ", rankfval.shape, " shape of rankgrid = ", rankgrid.shape)

    fname = prefix + fnout + '_part' + str(rank) + '.hdf5'
    with h5py.File(fname, 'w') as f:
        f.create_dataset('histograms', data=rankfval)
        f.create_dataset('parameters', data=rankgrid)
#        print('Data generated stored in file: ', fname)

    
#    print('rankfval.shape: ', rankfval.shape)

    return iterlen, rankfval.shape[2]


def grid_sample(rank, size, fn, sample, prefix, fnout):

    print("TW: rank = {}, sample.shape = {}, sample = \n{}".format(rank, sample.shape, sample))
    rankfval = np.asarray(list(map(fn, sample)))
    print("TW: rank = {}, shape of rankfval = ".format(rank), rankfval.shape)

    fname = prefix + fnout + '_part' + str(rank) + '.hdf5'
    with h5py.File(fname, 'w') as f:
        f.create_dataset('histograms', data=rankfval)
        f.create_dataset('parameters', data=sample)

    return sample.shape[0], rankfval.shape[2]

