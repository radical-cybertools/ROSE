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
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []

    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16s.json".format(i),))
        errors_random.append(jfile["100DGX-2023-10-16"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16s.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16s.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

    

    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_random)
    print(errors_loss)
    print(errors_combine_std)
    # exit(0)
    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/uniform/MAE_error_nodrop_{}_lp.csv".format(param), index=False)
    
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_MSEs.json".format(i),))
        errors_random.append(jfile["100DGX-2023-10-16"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_MSEs.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_MSEs.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

    # print(errors_random)
    # print(errors_loss)
    # print(errors_combine_std)
    # exit(0)
    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/uniform/MSE_error_nodrop_{}_lp.csv".format(param), index=False)

    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_losss.json".format(i),))
        errors_random.append(jfile["100DGX-2023-10-16"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_losss.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-10-16_losss.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-16"]["mean"][param][0])

    all_print = []
    print(errors_random)
    print(errors_loss)
    print(errors_combine_std)
    # exit(0)

    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_random)
    print(errors_loss)
    print(errors_combine_std)
    exit(0)

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/uniform/loss_error_nodrop_{}_lp.csv".format(param), index=False)

exit(0)
    