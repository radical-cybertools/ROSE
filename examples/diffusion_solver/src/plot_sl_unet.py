import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_0drop = {"random": "", "loss": "", "diverse": ""}

# WORKING LOCATIONS
path_0drop["random"] = "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/"
path_0drop["loss"] = "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/"
path_0drop["diverse"] = "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/"

accs_nodrop = {"random": [], "loss": [], "diverse": []}

for k in path_0drop.keys():
    temp = pd.read_csv(path_0drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_nodrop[k] = temp

keys = ["random", "loss", "diverse"]
all_to_print = []
for k in accs_nodrop.keys():
    to_print = []
    print(k, np.mean(accs_nodrop[k]))
    for p in [3, 7, 11, 15]:
        to_print.append(accs_nodrop[k][p])
        
    print("{}\t{}".format(k, to_print))
    all_to_print.append(to_print)
for a in all_to_print:
    print(a)
# df = pd.DataFrame(all_to_print)
# df.to_csv("./test_loss_csv_nodrop.csv", index=False)
exit(0)