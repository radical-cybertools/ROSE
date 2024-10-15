import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_0drop = {"random": "", "loss": "", "diverse": "", "diverse_param6": "", "diverse_param7": ""}

# WORKING LOCATIONS
path_0drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-30/"
# path_0drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-58/"
path_0drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/"
path_0drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-40-58/"
path_0drop["diverse_param6"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/"
path_0drop["diverse_param7"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-02-59/"

# path_0drop = {"random": "", "loss": "", "diverse": "", "diverse_param7": ""}
# path_0drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-23-21/"
# path_0drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-05-12-04-29/"
# path_0drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-05-12-04-39/"
# path_0drop["diverse_param7"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-05-12-04-47/"

path_20drop = {"random": "", "std": "", "loss": "", "diverse": ""}
path_20drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-14-51-54/"
path_20drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-14-52-15/"
path_20drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-14-52-32/"
path_20drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-14-52-48/"

path_40drop = {"random": "", "std": "", "loss": "", "diverse": ""}
path_40drop["random"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-12-06-45/"
path_40drop["std"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-12-07-11/"
path_40drop["std_bb"] = "/home/pb8294/Projects/DeepDiffusionSolver//diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-07-11-54-25/"
path_40drop["std_bb2"] = "/home/pb8294/Projects/DeepDiffusionSolver//diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-14-13-43-04/"
path_40drop["loss"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-12-07-39/"
path_40drop["diverse"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-14-10-02/"
path_40drop["combine_en"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-15-20-33/"
path_40drop["combine_all"] = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-22-15-20-52/"

fixed_0drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-04-45/"
fixed_20drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-05-31/"
fixed_40drop = "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-18-08-05-51/"

acc_f_0drop = pd.read_csv(fixed_0drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
acc_f_20drop = pd.read_csv(fixed_20drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
acc_f_40drop = pd.read_csv(fixed_40drop + "tester_arr.csv", sep=',', header=None).to_numpy()[0]

accs_nodrop = {"random": [], "loss": [], "diverse": [], "diverse_param6": [], "diverse_param7": []}
accs_drop20 = {"random": [], "std": [], "loss": [], "diverse": []}
accs_drop40 = {"random": [], "std": [], "std_bb": [], "std_bb2": [], "loss": [], "diverse": [], "combine_en": [], "combine_all": []}

for k in path_0drop.keys():
    temp = pd.read_csv(path_0drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_nodrop[k] = temp

# for k in accs_drop20.keys():
#     temp = pd.read_csv(path_20drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
#     accs_drop20[k] = temp

for k in path_40drop.keys():
    temp = pd.read_csv(path_40drop[k] + "tester_arr.csv", sep=',', header=None).to_numpy()[0]
    accs_drop40[k] = temp

# keys = ["random", "std", "loss", "diverse"]
# colors = ["k", "orange", "r", "g"]
# plt.figure(figsize=(20, 10))
# accs_drop40["std"][0] = accs_drop40["random"][0] 
# accs_drop40["loss"][0] = accs_drop40["random"][0] 



# for i, k in enumerate(keys):
#     # print(k, len(accs_nodrop[k]), accs_nodrop[k]); exit()
#     plt.subplot(131)
#     plt.plot(accs_nodrop[k], color=colors[i], linestyle="solid", label=k); plt.title("No dropout")
#     plt.ylim([0, 0.7])
#     plt.yticks(np.arange(0, 0.7, 0.1))
#     plt.legend()
#     plt.subplot(132); plt.plot(accs_drop20[k], color=colors[i], linestyle="solid", label=k); plt.title("20 percent dropout")
#     # plt.ylim([0, 0.7])
#     # plt.yticks(np.arange(0, 0.7, 0.1))
#     plt.legend()
#     plt.subplot(133); plt.plot(accs_drop40[k], color=colors[i], linestyle="solid", label=k); plt.title("40 percent dropout")
#     # plt.ylim([0, 0.7])
#     # plt.yticks(np.arange(0, 0.7, 0.1))
#     plt.legend()
#     plt.savefig("plot_sl_linear.png")

# # print(k, len(accs_nodrop[k]), accs_nodrop[k]); exit()
# plt.close("all")
# plt.figure(figsize=(20,10))
# plt.subplot(141); 
# plt.plot(accs_nodrop["random"], color="black", linestyle="solid", label="No Drop")
# plt.plot(accs_drop20["random"], color="black", linestyle="dotted", label="Drop 20")
# plt.plot(accs_drop40["random"], color="black", linestyle="dashed", label="Drop 40"); plt.title("Random")
# plt.title("Random")
# plt.legend()

# plt.subplot(142); 
# plt.plot(accs_nodrop["std"], color="orange", linestyle="solid", label="No Drop")
# plt.plot(accs_drop20["std"], color="orange", linestyle="dotted", label="Drop 20")
# plt.plot(accs_drop40["std"], color="orange", linestyle="dashed", label="Drop 40"); plt.title("Std")
# plt.title("Std")
# plt.legend()

# plt.subplot(143); 
# plt.plot(accs_nodrop["loss"], color="red", linestyle="solid", label="No Drop")
# plt.plot(accs_drop20["loss"], color="red", linestyle="dotted", label="Drop 20")
# plt.plot(accs_drop40["loss"], color="red", linestyle="dashed", label="Drop 40"); plt.title("Loss")
# plt.title("Loss")
# plt.legend()

# plt.subplot(144); 
# plt.plot(accs_nodrop["diverse"], color="green", linestyle="solid", label="No Drop")
# plt.plot(accs_drop20["diverse"], color="green", linestyle="dotted", label="Drop 20")
# plt.plot(accs_drop40["diverse"], color="green", linestyle="dashed", label="Drop 40"); plt.title("Diverse")
# plt.title("Diverse")
# plt.legend()
# plt.savefig("plot_sl_3_linear.png")



# keys = ["random", "std", "combine_en", "combine_all"]
# colors = ["k", "orange", "magenta", "brown"]

# keys = ["random", "loss", "combine_en", "combine_all"]
# colors = ["k", "r", "magenta", "brown"]

# keys = ["random", "diverse", "combine_en", "combine_all"]
# colors = ["k", "g", "magenta", "brown"]

# keys = ["random", "combine_all"]
# colors = ["k", "brown"]


# keys = ["random", "std", "loss", "diverse"]
# colors = ["k", "orange", "r", "g"]

# keys = ["loss", "diverse", "diverse_param7", "diverse_param6"]
# colors = ["r", "g", "brown", "orange"]
# print(accs_nodrop)
# for i, k in enumerate(keys):
#     plt.figure(figsize=(12, 12))
#     plt.plot(accs_nodrop["random"], color="black", linestyle="solid", label="random")
#     if k == "std":
#         continue
#     plt.plot(accs_nodrop[k], color=colors[i], linestyle="solid", label=k)
#     plt.legend()
#     plt.grid()
#     plt.xticks(np.arange(0, 17, step=1))
#     plt.yticks(np.arange(min(accs_nodrop[k]), max(accs_nodrop[k]), step=0.05))
#     plt.savefig("plot_sl_nodrop_{}_re.png".format(k))

# UNCOMMENT MAIN CODE
# keys = ["std", "std_bb", "std_bb2", "loss", "diverse"]
# colors = ["r", "g", "blue", "brown", "magenta"]
# print(accs_drop40)
# for i, k in enumerate(keys):
#     plt.figure(figsize=(10, 10))
#     plt.plot(accs_drop40["random"], color="black", linestyle="solid", label="random")
#     plt.plot(accs_drop40[k], color=colors[i], linestyle="solid", label=k)
#     plt.xlabel("acquisition rounds")
#     plt.ylabel("test loss")
#     plt.legend()
#     plt.savefig("plot_sl_40drop_{}_bounding_notanh.png".format(k))

keys = ["random", "loss", "diverse_param6", "diverse_param7"]
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