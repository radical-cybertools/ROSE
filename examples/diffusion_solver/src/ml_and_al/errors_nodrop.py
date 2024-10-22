import json
import numpy as np
import matplotlib.pyplot as plt

errors_percent = []
errors_random = []
errors_diverse = []
errors_std = []
errors_loss = []
errors_combine_all = []
errors_combine_std = []
errors_combine_lp = []
for param in ["all", "src", "field", "ring1", "ring2", "ring3"]:
    for i in range(1000, 17000, 1000):
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-40-58/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-28.json".format(i),))
        # errors_diverse.append(jfile["100DGX-2023-08-28"]["mean"]["field"][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-28-10-25-27/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-28.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-08-28"]["mean"]["field"][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-30/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16.json".format(i),))
        errors_random.append(jfile["100DGX-2023-09-16"]["mean"]["src"][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-09-16"]["mean"]["src"][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-09-16"]["mean"]["src"][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-02-59/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-16.json".format(i),))
        errors_combine_all.append(jfile["100DGX-2023-09-16"]["mean"]["src"][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-17.json".format(i),))
        errors_combine_lp.append(jfile["100DGX-2023-09-17"]["mean"]["src"][0])


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

    print(errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15], np.mean(errors_combine_all))
    all_print.append([errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15], np.mean(errors_combine_all)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./MAE_error_nodrop_src_lp.csv", index=False)

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