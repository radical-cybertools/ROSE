import json
import numpy as np
import matplotlib.pyplot as plt

errors_percent = []
errors_random = []
errors_diverse = []
errors_std = []
errors_std_bb = []
errors_loss = []
errors_combine_all = []
errors_combine_std = []
for i in range(1000, 17000, 1000):
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-16-00/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-23.json".format(i),))
    
    errors_percent.append(jfile["100DGX-2023-08-23"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-12-30/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-23.json".format(i),))
    errors_random.append(jfile["100DGX-2023-08-23"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-13-49/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-23.json".format(i),))
    errors_diverse.append(jfile["100DGX-2023-08-23"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-13-19/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-23.json".format(i),))
    errors_std.append(jfile["100DGX-2023-08-23"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-13-33/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-23.json".format(i),))
    errors_loss.append(jfile["100DGX-2023-08-23"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-16-43-29/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-24.json".format(i),))
    errors_combine_std.append(jfile["100DGX-2023-08-24"]["mean"]["all"][0])
    
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-16-43-12/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-08-24.json".format(i),))
    errors_combine_all.append(jfile["100DGX-2023-08-24"]["mean"]["all"][0])
    
    
    # Extra for bounding box
    jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-07-11-54-25//{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-14.json".format(i),))
    errors_std_bb.append(jfile["100DGX-2023-09-14"]["mean"]["all"][0])
    
    
    
# print(np.mean(errors_random), np.mean(errors_loss), np.mean(errors_diverse), np.mean(errors_combine_all), np.mean(errors_combine_std), np.mean(errors_std_bb))
# exit(0)

import pandas as pd
all_print = []
print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], )
all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

print(errors_std[3], errors_std[7], errors_std[11], errors_std[15]) 
all_print.append([errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std)])

print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15]) 
all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

print(errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15])
all_print.append([errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse)])

print(errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15])
all_print.append([errors_combine_all[3], errors_combine_all[7], errors_combine_all[11], errors_combine_all[15], np.mean(errors_combine_all)])

print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15])
all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

print(errors_std_bb[3], errors_std_bb[7], errors_std_bb[11], errors_std_bb[15]) 
all_print.append([errors_std_bb[3], errors_std_bb[7], errors_std_bb[11], errors_std_bb[15], np.mean(errors_std_bb)])

df = pd.DataFrame(all_print)
df.to_csv("./errors_csv_40drop_all.csv", index=False)    
    
    
    
plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_diverse, label="diverse")
plt.legend()
plt.savefig("errors40_all_r_d.png")

plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_std, label="std")
plt.legend()
plt.savefig("errors40_all_r_s.png")

plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_loss, label="loss")
plt.legend()
plt.savefig("errors40_all_r_l.png")

plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_combine_std, label="combine_ent")
plt.legend()
plt.savefig("errors40_all_r_ce.png")

plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_combine_all, label="combine_all")
plt.legend()
plt.savefig("errors40_all_r_ca.png")

plt.figure(figsize=(10, 10))
# plt.plot(errors_percent, label="percent")
plt.plot(errors_random, label="random")
plt.plot(errors_std_bb, label="Bounding box")
plt.legend()
plt.savefig("errors40_all_r_bb.png")

# plt.savefig("errors.png")
# plt.figure(figsize=(10, 10))
# plt.plot(errors_random, label="random")
# plt.plot(errors_diverse, label="diverse")
# plt.plot(errors_std, label="std")
# plt.plot(errors_loss, label="loss")
# plt.plot(errors_combine_std, label="combine_ent")
# plt.plot(errors_combine_all, label="combine_all")
# plt.legend()
# plt.savefig("errors_field.png")
# print(errors_percent)
# print(errors_random)
# print(errors_diverse)
# print(errors_std)
# print(errors_loss)