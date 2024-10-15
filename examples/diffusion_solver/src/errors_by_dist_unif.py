import json
import numpy as np
import matplotlib.pyplot as plt
for param in ["all", "src", "field", "ring1", "ring2", "ring3"]:
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    for i in range(1000, 17000, 1000):
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-40-58/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-28.json".format(i),))
        # errors_diverse.append(jfile["100DGX-2023-08-28"]["mean"]["field"][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-25-27/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-28.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-08-28"]["mean"]["field"][0])
        temp_random = []
        temp_loss = []
        temp_p6 = []
        temp_p7 = []
        for j in range(10):
            # /home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/1000/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16_MSE-0.json
            jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_loss-{}.json".format(i, j),))
            temp_random.append(jfile["100DGX-2023-09-28"]["mean"][param][0])
            
            jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_loss-{}.json".format(i, j),))
            temp_loss.append(jfile["100DGX-2023-09-28"]["mean"][param][0])

            jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_loss-{}.json".format(i, j),))
            temp_p6.append(jfile["100DGX-2023-09-28"]["mean"][param][0])

            jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_loss-{}.json".format(i, j),))
            temp_p7.append(jfile["100DGX-2023-09-28"]["mean"][param][0])
            
            # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-02-59/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16-{}.json".format(i, j),))
            # temp_p7.append(jfile["100DGX-2023-09-16"]["mean"]["all"][0])
        errors_random.append(temp_random)
        errors_loss.append(temp_loss)
        errors_combine_std.append(temp_p6)
        errors_combine_all.append(temp_p7)

    errors_random = np.array(errors_random)
    errors_loss = np.array(errors_loss)
    errors_combine_std = np.array(errors_combine_std)
    errors_combine_all = np.array(errors_combine_all)
    import pandas as pd
    to_save_random = errors_random[[3, 7, 11, 15], :]
    to_save_loss = errors_loss[[3, 7, 11, 15], :]
    to_save_combine_std = errors_combine_std[[3, 7, 11, 15], :]
    to_save_combine_lp = errors_combine_all[[3, 7, 11, 15], :]

    # print(to_save_combine_lp)
    # df = pd.DataFrame(to_save_combine_lp)
    # df.to_csv("./tests_results/9-29-unif_bydist/_{}_unif_bydist.csv".format(param), index=False)
# exit(0)
    print(to_save_random.shape, to_save_loss.shape, to_save_combine_std.shape, to_save_combine_lp.shape)
    to_save = np.zeros((16, 10))
    to_save[0::4,:] = to_save_random
    to_save[1::4,:] = to_save_loss
    to_save[2::4,:] = to_save_combine_std
    to_save[3::4,:] = to_save_combine_lp

# to_save = np.vstack((to_save_random, to_save_loss, to_save_combine_std))
    df = pd.DataFrame(to_save)
    df.to_csv("./tests_results/9-29-unif_bydist/loss_{}.csv".format(param), index=False)
    print(to_save.shape)
    print(to_save_random.shape, to_save_loss.shape, to_save_combine_std.shape)
    exit(0)

# print(np.mean(errors_random), np.mean(errors_loss), np.mean(errors_diverse), np.mean(errors_combine_all), np.mean(errors_combine_std))
# exit(0)

import pandas as pd
all_print = []
print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

# print(errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15])
# all_print.append([errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15]])

# print(errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15], np.mean(errors_combine_all))
# all_print.append([errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15], np.mean(errors_combine_all)])

print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])
df = pd.DataFrame(all_print)
df.to_csv("./MSE_error_nodrop_field.csv", index=False)

exit(0)
    
plt.figure(figsize=(12, 12))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_diverse, label="diverse")
plt.xticks(np.arange(0, 17, step=1))
plt.yticks(np.arange(min(errors_diverse), max(errors_diverse), step=0.0001))
plt.grid()
plt.legend()
plt.savefig("errors_nodrop_field_r_d.png")

# plt.figure(figsize=(12, 12))
# # plt.plot(errors_percent, label="percent")
# plt.plot(errors_random, label="random")
# plt.plot(errors_std, label="std")
# plt.xticks(np.arange(0, 17, step=1))
# plt.yticks(np.arange(min(errors_std), max(errors_std), step=0.0001))
# plt.legend()
# plt.savefig("errors_nodrop_field_r_s.png")

plt.figure(figsize=(12, 12))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_loss, label="loss")
plt.xticks(np.arange(0, 17, step=1))
plt.yticks(np.arange(min(errors_loss), max(errors_loss), step=0.0001))
plt.grid()
plt.legend()
plt.savefig("errors_nodrop_field_r_l.png")

plt.figure(figsize=(12, 12))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_combine_std, label="diverse_param6")
plt.xticks(np.arange(0, 17, step=1))
plt.yticks(np.arange(min(errors_combine_std), max(errors_combine_std), step=0.0001))
plt.grid()
plt.legend()
plt.savefig("errors_nodrop_field_r_p6.png")

plt.figure(figsize=(12, 12))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_combine_all, label="diverse_param7")
plt.xticks(np.arange(0, 17, step=1))
plt.yticks(np.arange(min(errors_combine_all), max(errors_combine_all), step=0.0001))
plt.grid()
plt.legend()
plt.savefig("errors_nodrop_field_r_p7.png")

# # plt.savefig("errors.png")
# plt.figure(figsize=(12, 12))
# plt.plot(errors_random, label="random")
# plt.plot(errors_diverse, label="diverse")
# plt.plot(errors_std, label="std")
# plt.plot(errors_loss, label="loss")
# plt.plot(errors_combine_std, label="diverse_param6")
# plt.plot(errors_combine_all, label="diverse_param7")
# plt.legend()
# plt.savefig("errors_nodrop_field.png")
# print(errors_percent)
# print(errors_random)
# print(errors_diverse)
# print(errors_std)
# print(errors_loss)