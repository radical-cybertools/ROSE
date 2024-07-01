import numpy as np
import matplotlib.pyplot as plt
import matplotlib

arrs = {"random": [], "loss": [], "diverse": [], "diverse_param6": [], "diverse_param7": []}
arrs["random"] = np.load("./distwise_random.npy")
arrs["loss"] = np.load("./distwise_loss.npy")
arrs["diverse"] = np.load("./distwise_diverse.npy")
arrs["diverse_param6"] = np.load("./distwise_diverse_param6.npy")
arrs["diverse_param7"] = np.load("./distwise_diverse_param7.npy")

a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
x = [245, 466, 638, 706, 638, 582, 417, 219, 71, 18]
# matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
for k in arrs.keys():
    if k == "random":
        continue
    plt.figure(figsize=(50, 30))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.plot(arrs["random"][:, i], label="random", linewidth=4)
        label = k
        if k == 'diverse':
            label = "feat diversity"
        elif k == "diverse_param7":
            label = "param diversty"
        
        plt.plot(arrs[k][:, i], label=label, linewidth=4)
        plt.ylabel("Test Loss")
        plt.xlabel("Acquisition rounds")
        # plt.plot(diverse[:, i], label="feat diverse")
        # plt.plot(diverse_param6[:, i], label="param diverse 6")
        # plt.plot(diverse_param7[:, i], label="param diverse 7")
        plt.title("{} Data \n Distance between {}-{}".format(x[i], a[i], a[i + 1]))
        plt.legend()
        # plt.tight_layout()
        plt.savefig("zzzz_{}.png".format(k))
        
