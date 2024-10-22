import h5py
import numpy as np
import sys, os

#We currently handle cubic, trigonal and tetragonal sym, and we handle them separately here, and only solve the y-axis (parameter)
#We create Y-data with size n*4 and save to file
#For trigonal    [a, a, alpha, 0]
#For tetragonal  [a, c, 90, 1]
#For cubic       [a, a, 90, 2]
#It does not do data normalization!!!
#Be careful! Here cubic has label 2 instead of 0!!!
def main():
    if(len(sys.argv) != 4):
        print(sys.argv)
        sys.stderr.write("Usage: python merge_preprocess_hdf5.py dir_path sym_class num_rank")
        sys.exit(0)
    dir_path  = sys.argv[1]
    sym_class = str(sys.argv[2])
    num_rank  = int(sys.argv[3])
    print("dir_path = {}\nsym_class = {}\nnum_rank = {}".format(dir_path, sym_class, num_rank))

    histograms = []
    parameters = []

    if sym_class == "trigonal":
        filename_prefix = "trigonal_1522004_trigonal"
    elif sym_class == "tetragonal":
        filename_prefix = "tetragonal_1531431_tetragonal" 
    elif sym_class == "cubic":
        filename_prefix = "cubic_1001460_cubic"
    else:
        sys.stderr.write("sym has to be cubic, trigonal or tetragonal!\n")
        sys.exit(0)

    for rank in range(num_rank):
        filename = "".join([dir_path, "/", filename_prefix, "_part", str(rank), ".hdf5"])
        with h5py.File(filename, 'r') as f:
            histograms.append(f["histograms"][:])
            parameters.append(f["parameters"][:])

    all_histograms = np.concatenate(histograms, axis=0)
    all_parameters = np.concatenate(parameters, axis=0)

    if sym_class == "trigonal":
        parameters_a = all_parameters[:,0].reshape(-1, 1)
        class_label = np.array([0])
        class_label = np.tile(class_label, (all_parameters.shape[0], 1))
        all_parameters = np.concatenate([all_parameters, parameters_a, class_label], axis=1)
        all_parameters[:,[1,2]] = all_parameters[:,[2,1]]
    elif sym_class == "tetragonal":
        parameters_alpha = np.ones((all_parameters.shape[0], 1)) * 90.0
        class_label = np.array([1])
        class_label = np.tile(class_label, (all_parameters.shape[0], 1))
        all_parameters = np.concatenate([all_parameters, parameters_alpha, class_label], axis=1)
    elif sym_class == "cubic":
        parameters_a = all_parameters[:,0].reshape(-1, 1)
        parameters_alpha = np.ones((all_parameters.shape[0], 1)) * 90.0
        class_label = np.array([2])
        class_label = np.tile(class_label, (all_parameters.shape[0], 1))
        all_parameters = np.concatenate([parameters_a, parameters_a, parameters_alpha, class_label], axis=1)

    newfile = "".join([dir_path, "/", filename_prefix, ".hdf5"])
    with h5py.File(newfile, 'w') as f:
        f.create_dataset("histograms", data=all_histograms)
        f.create_dataset("parameters", data=all_parameters)

if __name__ == '__main__':
    main()
