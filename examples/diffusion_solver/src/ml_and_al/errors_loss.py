import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

errors_random = {"all": [], "src": [], "field": [], "ring1": [], "ring2": [], "ring3": []}
errors_loss = {"all": [], "src": [], "field": [], "ring1": [], "ring2": [], "ring3": []}
errors_diverse = {"all": [], "src": [], "field": [], "ring1": [], "ring2": [], "ring3": []}
errors_lp = {"all": [], "src": [], "field": [], "ring1": [], "ring2": [], "ring3": []}
keys = ["all", "src", "field", "ring1", "ring2", "ring3"]
for key in keys:
    to_print = []
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-19_loss.json".format(i),))
        errors_random[key].append(jfile["100DGX-2023-09-19"]["mean"][key][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-19_loss.json".format(i),))
        errors_loss[key].append(jfile["100DGX-2023-09-19"]["mean"][key][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-19_loss.json".format(i),))
        errors_diverse[key].append(jfile["100DGX-2023-09-19"]["mean"][key][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-19_loss.json".format(i),))
        errors_lp[key].append(jfile["100DGX-2023-09-19"]["mean"][key][0])
        
        
        
    to_print.append(errors_random[key])
    to_print.append(errors_loss[key])
    to_print.append(errors_diverse[key])
    to_print.append(errors_lp[key])
    df = pd.DataFrame(to_print)
    df.to_csv("/home/pb8294/Projects/DeepDiffusionSolver/tests_results/errors_loss_{}.csv".format(key), index=False) 
    
# print(np.mean(errors_random), np.mean(errors_loss), np.mean(errors_diverse), np.mean(errors_combine_all), np.mean(errors_combine_std), np.mean(errors_std_bb))
# exit(0)
