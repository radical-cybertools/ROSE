import numpy as np
import matplotlib.pyplot as plt

random_nodrop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-07-47-43/"
ent_std_nodrop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-15-24/"
ent_lss_nodrop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-15-40/"
diverse_nodrop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-15-43/"


random_drop40 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-16-32/"
ent_std_drop40 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-16-34/"
ent_lss_drop40 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-16-36/"
diverse_drop40 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-16-13-16-38/"


# random_drop20 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-13-18-42/"
# ent_std_drop20 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-07-08-00/"
# ent_lss_drop20 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-07-08-08/"
# diverse_drop20 = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-07-08-16/"

accs_nodrop = {"random": [], "std": [], "loss": [], "diverse": []}
accs_drop40 = {"random": [], "std": [], "loss": [], "diverse": []}

import pandas as pd

for i in range(1000, 17000, 1000):
    temp = pd.read_csv(random_nodrop + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_nodrop["random"].append(temp[-1][-1])

    temp = pd.read_csv(ent_std_nodrop + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_nodrop["std"].append(temp[-1][-1])

    temp = pd.read_csv(ent_lss_nodrop + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_nodrop["loss"].append(temp[-1][-1])

    temp = pd.read_csv(diverse_nodrop + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_nodrop["diverse"].append(temp[-1][-1])

    temp = pd.read_csv(random_drop40 + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_drop40["random"].append(temp[-1][-1])

    temp = pd.read_csv(ent_std_drop40 + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_drop40["std"].append(temp[-1][-1])

    temp = pd.read_csv(ent_lss_drop40 + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_drop40["loss"].append(temp[-1][-1])

    temp = pd.read_csv(diverse_drop40 + "test_error{}.csv".format(i), sep=',', header=None).to_numpy()
    accs_drop40["diverse"].append(temp[-1][-1])


keys = ["random", "std", "loss", "diverse"]
colors = ["k", "orange", "r", "g"]
plt.figure(figsize=(20, 10))
for i, k in enumerate(keys):
    # print(k, len(accs_nodrop[k]), accs_nodrop[k]); exit()
    plt.subplot(121); plt.plot(accs_nodrop[k], color=colors[i], linestyle="solid", label=k)
    plt.legend()
    plt.subplot(122); plt.plot(accs_drop40[k], color=colors[i], linestyle="solid", label=k)
    plt.legend()
    plt.savefig("plot.png")