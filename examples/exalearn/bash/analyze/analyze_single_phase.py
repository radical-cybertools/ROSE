import os
import re
import numpy as np

#metric = "class"
#metric = "l2"
metric = "uncertainty"

directory = '/lus/eagle/projects/RECUP/twang/exalearn_stage2/archive/exp_1_3/72000/'
exp_start_with = "exp1_3_submit.sh.o"

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
    if len(values) == 1:
        return values  # Return the second occurrence
    else:
        print(f"size is not 1 in file {filename}")
        return None

# List to store all the extracted values
values = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.startswith(exp_start_with):
        full_path = os.path.join(directory, filename)
        value = extract_values_from_file(full_path, metric)
        if value is not None:
            values.append(value[0])

# Calculate the average and standard deviation
if values:
    average = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric} Average: {average}")
    print(f"{metric} Error (std. dev): {std_dev}")
else:
    print("No valid data found to calculate.")

