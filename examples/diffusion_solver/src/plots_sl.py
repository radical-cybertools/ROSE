import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_0drop = {"random": "", "std": "", "loss": "", "diverse": ""}
path_0drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-16-36/"
path_0drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-16-51/"
path_0drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-17-11/"
path_0drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-17-26/"

path_20drop = {"random": "", "std": "", "loss": "", "diverse": ""}
path_20drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-12-31/"
path_20drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-12-39/"
path_20drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-12-47/"
path_20drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-12-56/"

path_40drop = {"random": "", "std": "", "loss": "", "diverse": ""}
path_40drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-14-37/"
path_40drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-10-22-17/"
path_40drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-10-22-55/"
path_40drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-17-15-15-08/"
path_40drop["combine_en"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-11-27-13/"
path_40drop["combine_all"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-11-27-14/"

fixed_0drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-04-45/"
fixed_20drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-05-31/"
fixed_40drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-05-51/"

acc_f_0drop = pd.read_csv(fixed_0drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
acc_f_20drop = pd.read_csv(fixed_20drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
acc_f_40drop = pd.read_csv(fixed_40drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]

accs_nodrop = {"random": [], "std": [], "loss": [], "diverse": []}
accs_drop20 = {"random": [], "std": [], "loss": [], "diverse": []}
accs_drop40 = {"random": [], "std": [], "loss": [], "diverse": [], "combine_en": [], "combine_all": []}

for k in path_0drop.keys():
    temp = pd.read_csv(path_0drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_nodrop[k] = temp

    temp = pd.read_csv(path_20drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_drop20[k] = temp

for k in path_40drop.keys():
    temp = pd.read_csv(path_40drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_drop40[k] = temp

keys = ["random", "std", "loss", "diverse"]
colors = ["k", "orange", "r", "g"]
plt.figure(figsize=(20, 10))
accs_drop40["std"][0] = accs_drop40["random"][0] 
accs_drop40["loss"][0] = accs_drop40["random"][0] 



for i, k in enumerate(keys):
    # print(k, len(accs_nodrop[k]), accs_nodrop[k]); exit()
    plt.subplot(131)
    plt.plot(accs_nodrop[k], color=colors[i], linestyle="solid", label=k); plt.title("No dropout")
    plt.ylim([0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.legend()
    plt.subplot(132); plt.plot(accs_drop20[k], color=colors[i], linestyle="solid", label=k); plt.title("20 percent dropout")
    plt.ylim([0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.legend()
    plt.subplot(133); plt.plot(accs_drop40[k], color=colors[i], linestyle="solid", label=k); plt.title("40 percent dropout")
    plt.ylim([0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.legend()
    plt.savefig("plot_sl_.png")

# print(k, len(accs_nodrop[k]), accs_nodrop[k]); exit()
plt.close("all")
plt.figure(figsize=(20,10))
plt.subplot(141); 
plt.plot(accs_nodrop["random"], color="black", linestyle="solid", label="No Drop")
plt.plot(accs_drop20["random"], color="black", linestyle="dotted", label="Drop 20")
plt.plot(accs_drop40["random"], color="black", linestyle="dashed", label="Drop 40"); plt.title("Random")
plt.title("Random")
plt.legend()

plt.subplot(142); 
plt.plot(accs_nodrop["std"], color="orange", linestyle="solid", label="No Drop")
plt.plot(accs_drop20["std"], color="orange", linestyle="dotted", label="Drop 20")
plt.plot(accs_drop40["std"], color="orange", linestyle="dashed", label="Drop 40"); plt.title("Std")
plt.title("Std")
plt.legend()

plt.subplot(143); 
plt.plot(accs_nodrop["loss"], color="red", linestyle="solid", label="No Drop")
plt.plot(accs_drop20["loss"], color="red", linestyle="dotted", label="Drop 20")
plt.plot(accs_drop40["loss"], color="red", linestyle="dashed", label="Drop 40"); plt.title("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(144); 
plt.plot(accs_nodrop["diverse"], color="green", linestyle="solid", label="No Drop")
plt.plot(accs_drop20["diverse"], color="green", linestyle="dotted", label="Drop 20")
plt.plot(accs_drop40["diverse"], color="green", linestyle="dashed", label="Drop 40"); plt.title("Diverse")
plt.title("Diverse")
plt.legend()
plt.savefig("plot_sl_3.png")

keys = ["random", "std", "loss", "diverse", "combine_en", "combine_all"]
colors = ["k", "orange", "r", "g", "magenta", "brown"]

plt.figure(figsize=(10, 10))
for i, k in enumerate(keys):
    plt.plot(accs_drop40[k], color=colors[i], linestyle="solid", label=k)
    plt.ylim([0, 0.7])
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.legend()
    plt.savefig("plot_sl_combine_.png")