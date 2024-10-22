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
    # for i in range(1000, 17000, 1000):
        # UNIFORMLY SAMPLED BASED ON DISTANCE BETWEEN TWO SOURCES
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-19s.json".format(i),))
        # errors_random.append(jfile["100DGX-2023-09-19"]["mean"][param][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-19s.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-09-19"]["mean"][param][0])

        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-19s.json".format(i),))
        # errors_combine_std.append(jfile["100DGX-2023-09-19"]["mean"][param][0])

        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-19s.json".format(i),))
        # errors_combine_lp.append(jfile["100DGX-2023-09-19"]["mean"][param][0])

    # OVER ALL TEST DATA
    # for i in range(1000, 17000, 1000):
    #     jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25s.json".format(i),))
    #     errors_random.append(jfile["100DGX-2023-09-25"]["mean"][param][0])
        
    #     jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25s.json".format(i),))
    #     errors_loss.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

    #     jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25s.json".format(i),))
    #     errors_combine_std.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

    #     jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25s.json".format(i),))
    #     errors_combine_lp.append(jfile["100DGX-2023-09-25"]["mean"][param][0])


    # UNET
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-05.json".format(i),))
        errors_random.append(jfile["100DGX-2023-10-05"]["mean"][param][0])
        
        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-05.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-10-05"]["mean"][param][0])

        jfile = json.load(open("/diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-05.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-05"]["mean"][param][0])

    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/MAE_error_nodrop_{}_lp.csv".format(param), index=False)
    
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_MSEs.json".format(i),))
        errors_random.append(jfile["100DGX-2023-09-25"]["mean"][param][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_MSEs.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_MSEs.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_MSEs.json".format(i),))
        errors_combine_lp.append(jfile["100DGX-2023-09-25"]["mean"][param][0])


    import pandas as pd
    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/MSE_error_nodrop_{}_lp.csv".format(param), index=False)

    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    for i in range(1000, 17000, 1000):
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_losss.json".format(i),))
        errors_random.append(jfile["100DGX-2023-09-25"]["mean"][param][0])
        
        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_losss.json".format(i),))
        errors_loss.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_losss.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

        jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/{}/AfterPlots/errors/uniform/errorsPerDS-100DGX-2023-09-25_losss.json".format(i),))
        errors_combine_lp.append(jfile["100DGX-2023-09-25"]["mean"][param][0])

    all_print = []
    print(errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random))
    all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[15], np.mean(errors_random)])

    print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)) 
    all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[15], np.mean(errors_loss)])

    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[15], np.mean(errors_combine_std)])

    print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp))
    all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[15], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/loss_error_nodrop_{}_lp.csv".format(param), index=False)

exit(0)
    