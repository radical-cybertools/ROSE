import numpy as np

# This function samples k points uniformly in the n-dim bounding box 
# var_ranges has n 2-tuples, where the i-th 2-tuple describes the min and max of i-th dim
def sample_points(k, var_ranges):
    m = len(var_ranges)
    samples = np.zeros((k, m))

    for i in range(m):
        var_min, var_max = var_ranges[i]
        samples[:, i] = np.random.uniform(var_min, var_max, k)

    return samples

#k = 5  # Number of data points to sample
#var_ranges = [(0, 1), (10, 20), (100, 200)]  # 3d, min, max pair
#sampled_points = sample_points(k, var_ranges)
#print("Sampled Points:\n", sampled_points)
