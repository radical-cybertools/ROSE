import os
import re
import numpy as np

filename = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/archive/exp_3_4/exp3_4_submit.sh.33210'

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
                "preprocessing study set",
                "training phase 0",
                "parallel group 1",
                "AL phase 0",
                "resample prepare, phase 1",
                "resample simulation and merge, first (AL), phase 1",
                "resample simulation and merge, second (stream), phase 1",
                "training phase 1",
                "parallel group 2",
                "AL phase 1",
                "resample prepare, phase 2",
                "resample simulation and merge, first (AL), phase 2",
                "resample simulation and merge, second (stream), phase 2",
                "training phase 2",
                "parallel group 3",
                "AL phase 2",
                "resample prepare, phase 3",
                "resample simulation and merge, phase 3",
                "training, phase 3",
                "entire script"
                ]

values = [[] for _ in pattern_body]
print(len(values))

for idx, body in enumerate(pattern_body):
    pattern = pattern_prefix + body
    value = extract_values_from_file(filename, pattern)
    values[idx].append(value)

results = []
results.append(myadd(values[0], values[1]))
results.append(values[4])
results.append(myadd(values[2], values[3]))
results.append(values[5])
results.append(myadd(values[6], values[7]))
results.append(values[8])
results.append(values[10])
results.append(values[9])
results.append(values[11])
results.append(myadd(values[12], values[13]))
results.append(values[14])
results.append(values[16])
results.append(values[15])
results.append(values[17])
results.append(myadd(values[18], values[19]))
results.append(values[20])
results.append(values[21])
results.append(values[22])

for result in results:
    print(f"${result[0]/1000.0:.1f}$")
