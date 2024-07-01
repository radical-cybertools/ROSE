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
from util.tools import accuracy, plotFigureDS, tools, per_image_error, predVsTarget, errInDS
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
    
    
    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-29-06-54-16/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-09-21-10-51-02/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/unetab/2023-10-25-11-56-02/"]
    
    prefixes = ["ur", "ud", "ut"]
    
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-22-10-15-43/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-21-10-50-31/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/"]
    
    # prefixes = ["wr", "wd", "wt"]

    paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-11-11-10-10-31/"]
    prefixes = "f8c"

    for iter, pt in enumerate(paths):
        for alvsall in [16000]:
        # for alvsall in range(1000, 17000, 1000):
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
            diffSolv = select_nn("SimpleCNN100")
            # diffSolv = select_nn("UNETAB")

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
                    
                    
                    plotFigureDS(theModel, testloader, device, transformation="linear", error_fnc=nn.MSELoss(reduction='none'), it=alvsall, pref=prefixes[iter])
                
                myLog.logging.info(f'Finished tests over datasets')
            print("done 1")
            # exit(0)
            # print(selectedDirs); exit(0)
            # modelName = next(iter(selectedDirs.keys()))
            # saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_MSE.json')
            myLog.logging.info(f'JSON object saved')
        # continue
        # exit(0)