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
    errors_lp = []

    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-22-10-15-43/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13.json".format(i),))
        errors_random.append(jfile["100DGX-2023-11-13"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-12-00-46/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-10-50-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13.json".format(i),))
        errors_diverse.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13.json".format(i),))
        errors_lp.append(jfile["100DGX-2023-11-13"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-16-10-24-43{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-16.json".format(i),))
        errors_std.append(jfile["100DGX-2023-11-16"]["mean"][param][0])

    print(errors_random)
    print(errors_loss)
    print(errors_diverse)
    print(errors_lp)
    print(errors_std)
    # exit(0)
    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse))
    all_print.append([errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse)])

    print(errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp))
    all_print.append([errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp)])
    
    print(errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std))
    all_print.append([errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/simplecnn/mae_loss/MAE_error_nodrop_{}_lp.csv".format(param), index=False)
    
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    errors_lp = []
    # UNET
    for i in range(1000, 17000, 1000):
        # UNET PATHS (RANDOM, DIVERSITY, LOSS, TOD) --> MSE 500 epochs
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-03-09-14-44/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-01-10-13-01/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-01-10-10-53/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-01-10-08-13/"]



        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-22-10-15-43/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_MSE.json".format(i),))
        errors_random.append(jfile["100DGX-2023-11-13"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-12-00-46/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_MSE.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-10-50-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_MSE.json".format(i),))
        errors_diverse.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_MSE.json".format(i),))
        errors_lp.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-16-10-24-43{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-16_MSE.json".format(i),))
        errors_std.append(jfile["100DGX-2023-11-16"]["mean"][param][0])

    print("MSES")
    print(errors_random)
    print(errors_loss)
    print(errors_diverse)
    print(errors_lp)
    print(errors_std)
    # exit(0)
    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse))
    all_print.append([errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse)])

    print(errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp))
    all_print.append([errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp)])
    
    print(errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std))
    all_print.append([errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/simplecnn/mae_loss/MSE_error_nodrop_{}_lp.csv".format(param), index=False)

    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    errors_lp = []
    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-22-10-15-43/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_loss.json".format(i),))
        errors_random.append(jfile["100DGX-2023-11-13"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-12-00-46/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_loss.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-10-50-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_loss.json".format(i),))
        errors_diverse.append(jfile["100DGX-2023-11-13"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-13_loss.json".format(i),))
        errors_lp.append(jfile["100DGX-2023-11-13"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-16-10-24-43{}/AfterPlots/errors/errorsPerDS-100DGX-2023-11-16_loss.json".format(i),))
        errors_std.append(jfile["100DGX-2023-11-16"]["mean"][param][0])

    print("LOSS")
    print(errors_random)
    print(errors_loss)
    print(errors_diverse)
    print(errors_lp)
    print(errors_std)
    # exit(0)
    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse))
    all_print.append([errors_diverse[3], errors_diverse[7], errors_diverse[11], errors_diverse[15], np.mean(errors_diverse)])

    print(errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp))
    all_print.append([errors_lp[3], errors_lp[7], errors_lp[11], errors_lp[15], np.mean(errors_lp)])
    
    print(errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std))
    all_print.append([errors_std[3], errors_std[7], errors_std[11], errors_std[15], np.mean(errors_std)])


    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/simplecnn/mae_loss/loss_error_nodrop_{}_lp.csv".format(param), index=False)

exit(0)
    