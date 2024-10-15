import os
import re
import numpy as np

#metric = "class"
metric = "l2"
#metric = "uncertainty"

# Specify the directory where the files are located
directory = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/archive/exp_1_5/fac_1.2_0.5/'

filename_start_with = "exp1_5_submit.sh.o"

def extract_values_from_file(filename, metric):
    """ Extract the second occurrence of 'Avg class loss on test set = <value>' from the file. """
    values = []
    try:
        with open(filename, 'r') as file:
            # Collect all matches in the file
            for line in file:
                if metric == "class":
                    if "Avg class loss" in line:
                        match = re.search(r"Avg class loss on test set \D*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                        if match:
                            values.append(float(match.group(1)))

                elif metric == "l2":
                    if "Avg diff" in line:
                        match = re.search(r"Avg diff on test set \D*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                        if match:
                            values.append(float(match.group(1)))

                elif metric == "uncertainty":
                    if "Avg sigma^2 on test set" in line:
                        match = re.search(r"Avg sigma\^2 on test set \D*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                        if match:
                            values.append(float(match.group(1)))

                else:
                    print("Error! metric not found")        
                    exit(0)
    except IOError as e:
        print(f"Error opening file {filename}: {e}")
        return None
    # Check if we have at least two occurrences
    if len(values) == 4:
        return values  # Return the second occurrence
    else:
        print(f"size is not 4 in file {filename}")
        return None

# List to store all the extracted values
values = [[], [], [], []]

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.startswith(filename_start_with):
        full_path = os.path.join(directory, filename)
        value = extract_values_from_file(full_path, metric)
        if value is not None:
            for i in range(4):
                values[i].append(value[i])

# Calculate the average and standard deviation
if values:
    for i in range(4):
        average = np.mean(values[i])
        std_dev = np.std(values[i])
        print(f"{metric} Average {i}: {average}")
        print(f"{metric} Error (std. dev) {i}: {std_dev}")
else:
    print("No valid data found to calculate.")

