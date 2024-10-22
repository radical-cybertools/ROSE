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
    
    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-19-10-33-08/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/"]

    # OVER ALL TEST DATA
    for i in range(1000, 16000, 1000):
        # jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-19-10-33-08/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28s.json".format(i),))
        # errors_random.append(jfile["100DGX-2023-09-28"]["mean"][param][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28ss.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-09-28"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-10-25-11-56-02/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-27.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-27"]["mean"][param][0])

        


    import pandas as pd
    all_print = []
    # print(errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random))
    # all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random)])

    # print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)) 
    # all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)])
    if param == "all":
        print("MAE", errors_combine_std)
    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/tod/MAE_error_nodrop_{}_lp.csv".format(param), index=False)
    
    plt.close("all")
    plt.figure()
    plt.plot(errors_random, color="blue", label="random")
    plt.plot(errors_combine_std, color="red", label="diversity")
    plt.xlabel("acquisition round")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig("uniform_unet_mae_{}.png".format(param))
        
    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    for i in range(1000, 16000, 1000):
        # jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-19-10-33-08/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_MSEs.json".format(i),))
        # errors_random.append(jfile["100DGX-2023-09-28"]["mean"][param][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28ss.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-09-28"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-10-25-11-56-02/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-27_MSE.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-27"]["mean"][param][0])


    import pandas as pd
    all_print = []
    # print(errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random))
    # all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random)])

    # print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)) 
    # all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)])
    if param == "all":
        print("MSE", errors_combine_std)
    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/tod/MSE_error_nodrop_{}_lp.csv".format(param), index=False)

    plt.close("all")
    plt.figure()
    plt.plot(errors_random, color="blue", label="random")
    plt.plot(errors_combine_std, color="red", label="diversity")
    plt.xlabel("acquisition round")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("uniform_unet_mse_{}.png".format(param))

    errors_percent = []
    errors_random = []
    errors_diverse = []
    errors_std = []
    errors_loss = []
    errors_combine_all = []
    errors_combine_std = []
    errors_combine_lp = []
    for i in range(1000, 16000, 1000):
        # jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-19-10-33-08/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28_losss.json".format(i),))
        # errors_random.append(jfile["100DGX-2023-09-28"]["mean"][param][0])
        
        # jfile = json.load(open("/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-09-28ss.json".format(i),))
        # errors_loss.append(jfile["100DGX-2023-09-28"]["mean"][param][0])

        jfile = json.load(open("./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-10-25-11-56-02/{}/AfterPlots/errors/errorsPerDS-100DGX-2023-10-27_loss.json".format(i),))
        errors_combine_std.append(jfile["100DGX-2023-10-27"]["mean"][param][0])

    all_print = []
    # print(errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random))
    # all_print.append([errors_random[3], errors_random[7], errors_random[11], errors_random[14], np.mean(errors_random)])

    # print(errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)) 
    # all_print.append([errors_loss[3], errors_loss[7], errors_loss[11], errors_loss[14], np.mean(errors_loss)])
    if param == "all":
        print("loss", errors_combine_std)
    print(errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std))
    all_print.append([errors_combine_std[3], errors_combine_std[7], errors_combine_std[11], errors_combine_std[14], np.mean(errors_combine_std)])

    # print(errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp))
    # all_print.append([errors_combine_lp[3], errors_combine_lp[7], errors_combine_lp[11], errors_combine_lp[14], np.mean(errors_combine_lp)])

    df = pd.DataFrame(all_print)
    df.to_csv("./tests_results/unet/tod/loss_error_nodrop_{}_lp.csv".format(param), index=False)

exit(0)
    