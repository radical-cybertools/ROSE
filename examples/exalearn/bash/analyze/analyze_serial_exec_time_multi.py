import os
import re
import numpy as np

directory = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/archive/exp_2_1/'
exp_start_with = "exp2_1_submit.sh.o"

def extract_values_from_file(filename, pattern):
    values = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                if pattern in line:
                    match = re.search(r".* (\d+) milliseconds", line)
                    if match:
                        values.append(int(match.group(1)))
    except IOError as e:
        print(f"Error opening file {filename}: {e}")
        return None
    if len(values) != 1:
        print(f"Error! More than one occurence of {pattern} in {filename}")
        exit(0)
    return values[0]

def myadd(*args):
    result = []
    num = len(args)
    size = len(args[0])
    for i in range(size):
        temp = 0
        for j in range(num):
            temp = temp + args[j][i]
        result.append(temp)
    return result

pattern_prefix = "Logging: End "
pattern_body = [
                "base simulation and merge",
                "test simulation and merge",
                "study simulation and merge",
                "training phase 0",
                "AL phase 0",
                "resample simulation and merge, phase 1",
                "training, phase 1",
                "AL phase 1",
                "resample simulation and merge, phase 2",
                "training, phase 2",
                "AL phase 2",
                "resample simulation and merge, phase 3",
                "training, phase 3",
                "entire script"
                ]

values = [[] for _ in pattern_body]
print(len(values))

for filename in os.listdir(directory):
    if filename.startswith(exp_start_with):
        for idx, body in enumerate(pattern_body):
            pattern = pattern_prefix + body
            full_path = os.path.join(directory, filename)
            value = extract_values_from_file(full_path, pattern)
            values[idx].append(value)

results = []
results.append(myadd(values[0], values[1], values[2]))
results.append(values[3])
results.append(values[4])
results.append(values[5])
results.append(values[6])
results.append(values[7])
results.append(values[8])
results.append(values[9])
results.append(values[10])
results.append(values[11])
results.append(values[12])
results.append(values[13])

for result in results:
    average = int(np.mean(result))
    std_dev = int(np.std(result))
    print(f"   ${average} \pm {std_dev}$")
