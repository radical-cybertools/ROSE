from lib2to3.pgen2 import driver
import sys
sys.path.insert(1, '../util')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import json
import PIL
import logging
import sys
import argparse


# sys.path.insert(1, '/home/javier/Projects/DiffSolver/DeepDiffusionSolver/util')  #Change
sys.path.insert(1, '/home/pb8294/Projects/DeepDiffusionSolver/util')

from util.loaders import generateDatasets, inOut, saveJSON, loadJSON#, MyData
from util.NNets import SimpleCNN
from util.tools import accuracy, tools, per_image_error, predVsTarget, errInDS
from util.plotter import myPlots, plotSamp, plotSampRelative

from trainModel_100 import select_nn

def my_loss_nomean(output, target, alph=1, w=1, w2=2000):
    loss = ((1 + torch.tanh(w*target) * w2) * torch.abs((output - target)**alph))
    return loss

class thelogger(object):
    def logFunc(self, PATH, dict, dir="0"):
        self.initTime = datetime.now()
        os.path.isdir(PATH + "Logs/") or os.mkdir(PATH + "Logs/")
        os.path.isdir(PATH + "Logs/" + dir) or os.mkdir(PATH + "Logs/" + dir)
        path = PATH + "Logs/" + dir + "/"

        self.logging = logging
        self.logging = logging.getLogger()
        self.logging.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(os.path.join(path, 'tests.log'))
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.handler.setFormatter(formatter)
        self.logging.addHandler(self.handler)

        self.logging.info(f'{str(self.initTime).split(".")[0]} - Log')

def selectNN(dict):
#     if dict["NN"] != "SimpleAdapCNN100":
#         diffSolv = select_nn(dict["NN"])
#         diffSolv = diffSolv().to(device)
#     else:
#         diffSolv = SimpleAdapCNN100(dict).to(device)
    diffSolv = select_nn(dict["NN"], d=dict)
    diffSolv = diffSolv.to(device)
    return diffSolv

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Deep Diffusion Solver")

    # parser.add_argument('--path', dest="path", type=str, default="/home/pb8294/Documents/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/EDNet_Simple100/",help="Specify path to dataset")   #Change
    # parser.add_argument('--path', dest="path", type=str, default=args.path + "", help="Specify path to dataset")   #Change
    parser.add_argument('--path', dest="path", type=str, default="/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/", help="Specify path to dataset")

    parser.add_argument("--alvsall", dest="alvsall", type=str, default="all", help="Specify Active learning acquisition round (eg 1, 2, 3) or 'all' for non-al methods")
    parser.add_argument('--dataset', dest="dataset", type=str, default="All",
                        help="Specify dataset")
    parser.add_argument('--dir', dest="dir", type=str, default="100DGX-" + str(datetime.now()).split(" ")[0],
                        help="Specify directory name associated to model")
    parser.add_argument('--bashtest', dest="bashtest", type=bool, default=False,
                        help="Leave default unless testing flow")

    parser.add_argument('--bs', dest="bs", type=int, default=50,
                        help="Specify Batch Size")
    parser.add_argument('--nw', dest="nw", type=int, default=4,
                        help="Specify number of workers")
    parser.add_argument('--ngpu', dest="ngpu", type=int, default=1,
                        help="Specify ngpu. (Never have tested >1)")
    parser.add_argument('--lr', dest="lr", type=float, default=0.0001,
                        help="Specify learning rate")
    parser.add_argument('--maxep', dest="maxep", type=int, default=100,
                        help="Specify max epochs")

    parser.add_argument('--newdir', dest="newdir", type=bool, default=False,
                        help="Is this a new model?")
    parser.add_argument('--newtrain', dest="newtrain", type=bool, default=False,
                        help="Are you starting training")


    parser.add_argument('--transformation', dest="transformation", type=str, default="linear",
                        help="Select transformation: linear, sqrt or log?")
    parser.add_argument('--loss', dest="loss", type=str, default="exp",
                        help="Select loss: exp, step, toggle or rand?")
    parser.add_argument('--wtan', dest="wtan", type=float, default=10.0,
                        help="Specify hparam wtan")
    parser.add_argument('--w', dest="w", type=float, default=1.0,
                        help="Specify hparam w")
    parser.add_argument('--alpha', dest="alpha", type=int, default=1,
                        help="Specify hparam alpha")
    parser.add_argument('--w2', dest="w2", type=float, default=4000.0,
                        help="Specify hparam w2")
    parser.add_argument('--delta', dest="delta", type=float, default=0.02,
                        help="Specify hparam delta")
    parser.add_argument('--toggle', dest="toggle", type=int, default=1,
                        help="Specify hparam toggle")
        
    args = parser.parse_args()

    # UNET PATHS
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-18-14-15-48/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-08-28-10-41-45/"]


    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/"]
    
    # paths = ["/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/2023-08-23-09-21-30/", "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/", "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/", "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-02-59/"]

    paths = ["/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/", "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/","/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/"]


    # UNET PATHS (RANDOM, DIVERSITY)
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-19-10-33-08/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/"]
    # UNET PATHS (RANDOM, LOSS, DIVERSITY, TOD) --> MAE 500 epochs
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-28-11-52-57/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-10-25-11-56-02/"]

    # UNET PATHS (RANDOM, DIVERSITY, LOSS, TOD) --> MSE 500 epochs
    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-03-09-14-44/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-01-10-10-53/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-01-10-08-13/"]

    # # SimpleCNN PATHS (RANDOM, DIVERSITY, LOSS, TOD) --> MAE 500 epochs
    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-22-10-15-43/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-10-50-31/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-12-00-46/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/"]

    # # SimpleCNN PATHS (RANDOM, DIVERSITY, LOSS, TOD) --> MSE 500 epochs
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-06-08-27-41/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-09-09-51-55/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-06-08-28-48/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-07-13-18-33/"]
    
    # UNET MSE DIVERSITY (WORKING)
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-09-09-53-45/"]
    
    # ENTROPY (WITH DROPOUTS)
    # SIMPLE CNN MAE
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-16-10-24-43/"]
    
    # SIMPLE CNN MSE
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-16-10-23-56/"]
    
    # UNET MSE AND MAE
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-21-10-05-44/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-11-22-09-54-29/"]
    
    # UNET Entropy
    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/entropy/"]

    for pt in paths:
        for alvsall in range(1000, 17000, 1000):
            args.alvsall = str(alvsall)
            print("I am here", args.alvsall)
            ###Start Here
            args.path = pt
            PATH = args.path# "/raid/javier/Datasets/DiffSolver/"
            # PATH = PATH≥÷÷ß
            os.path.isdir(PATH + "/") or os.mkdir(PATH + "/")
            PATH = args.path + args.alvsall
            os.path.isdir(PATH + "/") or os.mkdir(PATH + "/")
            os.path.isdir(PATH + "/AfterPlots/") or os.mkdir(PATH + "/AfterPlots/")
            os.path.isdir(PATH + "/AfterPlots/errors/") or os.mkdir(PATH + "/AfterPlots/errors/")
            os.path.isdir(PATH + "/AfterPlots/Samples/") or os.mkdir(PATH + "/AfterPlots/Samples/") 
            os.path.isdir(PATH + "/Dict/") or os.mkdir(PATH + "/Dict/")
            os.path.isdir(PATH + "/Dict/" + args.dir + "/") or os.mkdir(PATH + "/Dict/" + args.dir + "/")
        # 	DATASETNAME = args.dataset # "All"
            # exit(0)
            dir = args.dir #'1DGX' #'Test'#str(21)
            BATCH_SIZE=args.bs #50
            NUM_WORKERS=args.nw #8
            ngpu = args.ngpu #1
            lr = args.lr #0.0001
            device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
            # diffSolv = select_nn("SimpleCNN100")
            diffSolv = select_nn("UNETAB")

            diffSolv = diffSolv.to(device)
            
            # os.listdir(PATH + "Dict/")

            selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}

            

            datasetNameList = ['TwoSourcesRdm'] # [f'{i}SourcesRdm' for i in range(1,21)]
            error, errorField, errorSrc = [], [], []
            dict = {
                "Diff": [args.path + "diffusion-model-"+str(args.alvsall)+".pt"], 
                "transformation": "linear"}
            print("Path model", dict["Diff"][-1])
            # spath = "/home/pb8294/Documents/Projects/DiffSolverAdaptive/diffusion-ai-results/log-transform/TwoSourcesRdm/EDNet_Simple100/exp25/diffusion-model.pt"
            spath = args.path + "diffusion-model-"+str(args.alvsall)+".pt"
            # print(spath); exit(0)
            for selectedDir in selectedDirs.keys():
                dir = selectedDir
                # os.listdir(os.path.join(PATH, "Dict", dir))[0]
                # dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
                # dict = inOut().loadDict(spath)
        #		print(dict, '\n')
                ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
                theModel.eval();
                myLog = thelogger()
                myLog.logFunc(PATH, dict, dir)
                myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
                for (j, ds) in enumerate(datasetNameList):
                    myLog.logging.info(f'Dataset: {ds}')
                    trainloader, testloader, valloader = generateDatasets("/home/pb8294/data/", datasetName=ds, batch_size=BATCH_SIZE, num_workers=4, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
                    print("Test size: {}".format(len(testloader.dataset)))
                    
                # for i, data in enumerate(testloader):
                #     x = data[0]
                #     y = data[1]
                    
                #     srcs = x > 0
                #     fields = x <=0
                #     rings1 = (y - y * torch.sign(x)) >= 0.2
                #     rings2 = ( (y - y * torch.sign(x)) >= 0.1 ) * ((y - y * torch.sign(x)) < 0.2)
                #     rings3 = ( (y - y * torch.sign(x)) >= 0.05 ) * ((y - y * torch.sign(x)) < 0.1)

                #     plt.close("all")
                #     plt.figure(figsize=(20, 20))
                #     plt.imshow(srcs[43].reshape(100, 100), cmap="gray")
                #     plt.xticks([]),plt.yticks([])
                #     plt.savefig("v-src.png", dpi=300, transparent=True)
                    
                #     plt.close("all")
                #     plt.figure(figsize=(20, 20))
                #     plt.imshow(fields[43].reshape(100, 100), cmap="gray")
                #     plt.xticks([]),plt.yticks([])
                #     plt.savefig("v-fld.png", dpi=300, transparent=True)
                    
                #     plt.close("all")
                #     plt.figure(figsize=(20, 20))
                #     plt.imshow(rings1[43].reshape(100, 100), cmap="gray")
                #     plt.xticks([]),plt.yticks([])
                #     plt.savefig("v-ring1.png", dpi=300, transparent=True)
                    
                #     plt.close("all")
                #     plt.figure(figsize=(20, 20))
                #     plt.imshow(rings2[43].reshape(100, 100), cmap="gray")
                #     plt.xticks([]),plt.yticks([])
                #     plt.savefig("v-ring2.png", dpi=300, transparent=True)
                    
                #     plt.close("all")
                #     plt.figure(figsize=(20, 20))
                #     plt.imshow(rings3[43].reshape(100, 100), cmap="gray")
                #     plt.xticks([]),plt.yticks([])
                #     plt.savefig("v-ring3.png", dpi=300, transparent=True)
                #     exit()
                    # print(len(testloader.dataset)); exit(0)
            #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=100)
                    arr = errInDS(theModel, testloader, device, transformation="linear", error_fnc=nn.MSELoss(reduction='none'))
                    selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
                    selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
                    selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
                    selectedDirs[selectedDir]["max"]["all"].append(arr[3])
                    selectedDirs[selectedDir]["max"]["field"].append(arr[4])
                    selectedDirs[selectedDir]["max"]["src"].append(arr[5])
                    selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
                    selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
                    selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
                    selectedDirs[selectedDir]["min"]["all"].append(arr[9])
                    selectedDirs[selectedDir]["min"]["field"].append(arr[10])
                    selectedDirs[selectedDir]["min"]["src"].append(arr[11])
                    selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
                    selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
                    selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

                    selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
                    selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
                    selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
                    selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
                    selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
                    selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
                    selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
                    selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
                    selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
                    selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
                    selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
                    selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
                    selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
                    selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
                    selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
                myLog.logging.info(f'Finished tests over datasets')
            print("done 1")
            # exit(0)
            # print(selectedDirs); exit(0)
            modelName = next(iter(selectedDirs.keys()))
            saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_MSE.json')
            myLog.logging.info(f'JSON object saved')
            
        # 	selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}

        # 	# datasetNameList = [f'{i}SourcesRdm' for i in [2]]
        # 	datasetNameList = ['TwoSourcesRdm']
            
        # 	error, errorField, errorSrc = [], [], []

        # 	for selectedDir in selectedDirs.keys():
        # 		dir = selectedDir
        # 		print(dir)
        # 		# os.listdir(os.path.join(PATH, "Dict", dir))[0]
        # 		# dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
        # #		print(dict, '\n')
        # 		ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        # 		theModel.eval();
        # # 		myLog = thelogger()
        # # 		myLog.logFunc(PATH, dict, dir)
        # 		myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
        # 		for (j, ds) in enumerate(datasetNameList):
        # 			myLog.logging.info(f'Dataset: {ds}')
        # 			trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
        # 	#     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=100)
        # 			arr = errInDS(theModel, trainloader, device, transformation="linear", error_fnc=nn.MSELoss(reduction='none'))
        # 			selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
        # 			selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
        # 			selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
        # 			selectedDirs[selectedDir]["max"]["all"].append(arr[3])
        # 			selectedDirs[selectedDir]["max"]["field"].append(arr[4])
        # 			selectedDirs[selectedDir]["max"]["src"].append(arr[5])
        # 			selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
        # 			selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
        # 			selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
        # 			selectedDirs[selectedDir]["min"]["all"].append(arr[9])
        # 			selectedDirs[selectedDir]["min"]["field"].append(arr[10])
        # 			selectedDirs[selectedDir]["min"]["src"].append(arr[11])
        # 			selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
        # 			selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
        # 			selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

        # 			selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
        # 			selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
        # 			selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
        # 			selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
        # 			selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
        # 			selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
        # 			selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
        # 			selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
        # 			selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
        # 			selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
        # 			selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
        # 			selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
        # 			selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
        # 			selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
        # 			selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
        # 		myLog.logging.info(f'Finished tests over datasets')

        # 	modelName = next(iter(selectedDirs.keys()))
        # 	saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}_MSE.json')
        # 	myLog.logging.info(f'JSON object saved')

        ###################    
            selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}
            # datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
            datasetNameList = ['TwoSourcesRdm']
            error, errorField, errorSrc = [], [], []

            for selectedDir in selectedDirs.keys():
                dir = selectedDir
                # os.listdir(os.path.join(PATH, "Dict", dir))[0]
                # dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
                
        #		print(dict, '\n')
                ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
                theModel.eval();
                myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
                for (j, ds) in enumerate(datasetNameList):
                    myLog.logging.info(f'Dataset: {ds}')
                    trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
                    print("Test size: {}".format(len(testloader.dataset)))
                    arr = errInDS(theModel, testloader, device, transformation="linear")
                    selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
                    selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
                    selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
                    selectedDirs[selectedDir]["max"]["all"].append(arr[3])
                    selectedDirs[selectedDir]["max"]["field"].append(arr[4])
                    selectedDirs[selectedDir]["max"]["src"].append(arr[5])
                    selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
                    selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
                    selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
                    selectedDirs[selectedDir]["min"]["all"].append(arr[9])
                    selectedDirs[selectedDir]["min"]["field"].append(arr[10])
                    selectedDirs[selectedDir]["min"]["src"].append(arr[11])
                    selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
                    selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
                    selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

                    selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
                    selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
                    selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
                    selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
                    selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
                    selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
                    selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
                    selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
                    selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
                    selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
                    selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
                    selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
                    selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
                    selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
                    selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])            
                myLog.logging.info(f'Finished tests over datasets')
            print("done 2")
            modelName = next(iter(selectedDirs.keys()))
            saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}.json')
            myLog.logging.info(f'JSON object saved')
            

        ###################    
            selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}
            # datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
            datasetNameList = ['TwoSourcesRdm']
            error, errorField, errorSrc = [], [], []

            for selectedDir in selectedDirs.keys():
                dir = selectedDir
                # os.listdir(os.path.join(PATH, "Dict", dir))[0]
                # dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
                
        #		print(dict, '\n')
                ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
                theModel.eval();
                myLog.logging.info(f'Generating tests using Loss... for model {selectedDir}')
                # for (j, ds) in enumerate(datasetNameList):
                myLog.logging.info(f'Dataset: TwoSourcesRdm')
                trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
                print("Test size: {}".format(len(testloader.dataset)))
                arr = errInDS(theModel, testloader, device, transformation="linear", error_fnc=my_loss_nomean)
                selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
                selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
                selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
                selectedDirs[selectedDir]["max"]["all"].append(arr[3])
                selectedDirs[selectedDir]["max"]["field"].append(arr[4])
                selectedDirs[selectedDir]["max"]["src"].append(arr[5])
                selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
                selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
                selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
                selectedDirs[selectedDir]["min"]["all"].append(arr[9])
                selectedDirs[selectedDir]["min"]["field"].append(arr[10])
                selectedDirs[selectedDir]["min"]["src"].append(arr[11])
                selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
                selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
                selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

                selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
                selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
                selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
                selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
                selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
                selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
                selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
                selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
                selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
                selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
                selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
                selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
                selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
                selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
                selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])            
                myLog.logging.info(f'Finished tests over datasets')
            print("done 3")
            modelName = next(iter(selectedDirs.keys()))
            saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_loss.json')
            myLog.logging.info(f'JSON object saved')

        # 	selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : []}}}
        # 	# datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
        # 	datasetNameList = ['TwoSourcesRdm']
        # 	error, errorField, errorSrc = [], [], []

        # 	for selectedDir in selectedDirs.keys():
        # 		dir = selectedDir
        # 		# os.listdir(os.path.join(PATH, "Dict", dir))[0]
        # 		# dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))

        # #		print(dict, '\n')
        # 		ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        # 		theModel.eval();
        # 		myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
        # 		for (j, ds) in enumerate(datasetNameList):
        # 			myLog.logging.info(f'Dataset: {ds}')
        # 			trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
        # 			arr = errInDS(theModel, trainloader, device, transformation="linear")
        # 			selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
        # 			selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
        # 			selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
        # 			selectedDirs[selectedDir]["max"]["all"].append(arr[3])
        # 			selectedDirs[selectedDir]["max"]["field"].append(arr[4])
        # 			selectedDirs[selectedDir]["max"]["src"].append(arr[5])
        # 			selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
        # 			selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
        # 			selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
        # 			selectedDirs[selectedDir]["min"]["all"].append(arr[9])
        # 			selectedDirs[selectedDir]["min"]["field"].append(arr[10])
        # 			selectedDirs[selectedDir]["min"]["src"].append(arr[11])
        # 			selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
        # 			selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
        # 			selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

        # 			selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
        # 			selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
        # 			selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
        # 			selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
        # 			selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
        # 			selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
        # 			selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
        # 			selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
        # 			selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
        # 			selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
        # 			selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
        # 			selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
        # 			selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
        # 			selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
        # 			selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])            
        # 		myLog.logging.info(f'Finished tests over datasets')

        # 	modelName = next(iter(selectedDirs.keys()))
        # 	saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}.json')
        # 	myLog.logging.info(f'JSON object saved')
            

        #     myLog.logging.info(f'Generating Sample')
        #     dsName = "TwoSourcesRdm"    #args.dataset #"19SourcesRdm"
        #     trainloader, testloader, valloader = generateDatasets(PATH, datasetName=dsName ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.0, s=100, transformation="linear").getDataLoaders()
        #     plotName = f'Model-{dir}_DS-{dsName}_sample.png'
        #     # os.listdir(os.path.join(PATH, "Dict", dir))[0]
        #     # dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
        #     diffSolv = select_nn("SimpleCNN100")
        #     diffSolv = diffSolv.to(device)
        #     ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        #     theModel.eval();
        #     plotSampRelative(theModel, testloader, dict, device, PATH, plotName, maxvalue=0.5, power=2.0)
        #     myLog.logging.info(f'Sample generated')
        # #################  
            
        #     dsName = "TwoSourcesRdm"
        #         # os.listdir(os.path.join(PATH, "Dict", dir))[0]
        #         # dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
        #     diffSolv = select_nn("SimpleCNN100")
        #     diffSolv = diffSolv.to(device)
        #     ep,err, theModel = inOut().load_model(diffSolv, "Diff", dict)
        #     theModel.eval();
        #     try:
        #     # 		print("linear")
        #         trainloader, testloader, valloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
        #     except:
        #         trainloader, testloader, valloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100).getDataLoaders()

        #     xi, yi, zi = predVsTarget(testloader, theModel, device, transformation = "linear", threshold = 0.0, nbins = 100, BATCH_SIZE = BATCH_SIZE, size = 100, lim = 10)
        #     dataname = f'Model-{dir}_DS-{dsName}.txt'
        #     np.savetxt(os.path.join(PATH, "AfterPlots", dataname), zi.reshape(100,100).transpose())

        #     power = 1/8
        #     plotName = f'Model-{dir}_DS-{dsName}_pow-{power}.png'
        #     plt.pcolormesh(xi, yi, np.power(zi.reshape(xi.shape) / zi.reshape(xi.shape).max(),1/8), shading='auto')
        #     plt.plot([0,1],[0,1], c='r', lw=0.2)
        #     plt.xlabel("Target")
        #     plt.ylabel("Prediction")
        #     plt.title(f'Model {dir},\nDataset {dsName}')
        #     plt.colorbar()
        #     plt.savefig(os.path.join(PATH, "AfterPlots", plotName), transparent=False)
        #     # plt.show()
        #     print("done 3")

        #How to run
        # python -W ignore tests.py --dir 1DGX