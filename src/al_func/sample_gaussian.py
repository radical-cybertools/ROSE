import numpy as np

#This function samples n points following mix-gaussian distribution
#Here y_study is m points in the parameter space, which serves as the center of each mixed gaussian distribution
#std_dev is the std dev of each dimension in the parameter space
#freq is an array of size m, where freq[i] represents how many points will be sample around y_study[i]
#The sum of freq will be n
def generate_gaussian_sample(y_study, freq, std_dev):
    all_samples = []

    for i in range(len(freq)):
        if freq[i] > 0:
            samples = np.random.normal(loc=y_study[i], scale=std_dev, size=(freq[i], len(y_study[i])))
            all_samples.extend(samples)

    return np.array(all_samples)

## Example usage
#y_study = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
#freq = [2, 3]
#std_dev = [0.1, 2.0]
#samples = generate_gaussian_sample(y_study, freq, std_dev)
#print("Generated Samples:\n", samples)
