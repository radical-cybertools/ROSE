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
    parser.add_argument('--path', dest="path", type=str, default="/home/pb8294/Projects/DeepDiffusionSolver//diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-07-11-54-25/", help="Specify path to dataset")

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
    # RANDOM
    # paths = ["/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/"]

    # LOSS + P DIVERSITY
    # paths = ["./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/"]

    paths = ["/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-08-00-58/", "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/","/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/", "./diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-09-16-15-29-33/"]

    # LOSS, PDIVERSITY
    # paths = ["/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-29-11-01-14/",
    # "/home/pb8294/Projects/DeepDiffusionSolver/diffusion-ai-results/log-transform/TwoSourcesRdm/simplecnn/2023-08-27-17-03-24/"]
    # trainloader, testloader, valloader = generateDatasets("/home/pb8294/data/", datasetName="TwoSourcesRdm", batch_size=args.bs, num_workers=4, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
    train_data, test_data, val_data = generateDatasets(PATH="/home/pb8294/data/",
                                                                  datasetName="TwoSourcesRdm",
                                                                  batch_size=args.bs,
                                                                  num_workers=3,
                                                                  std_tr=0.01,
                                                                  s=100,
                                                                  transformation="linear").getDatasets()

    testloader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                                shuffle=False, num_workers=4)

    distances = []
    for i, data in enumerate(testloader):
        dist = np.array(data[-1]).reshape(-1, 1)
        if i == 0:
            distances = dist
        else:
            distances = np.vstack((distances, dist))
    distances = distances.reshape(-1, 1)
    actual_indices = np.arange(4000)
    counts, bins = np.histogram(distances, 100)
    
    empty_diction = {}
    for i in range(100):
        empty_diction[i] = []
        
    for ind, dist in enumerate(distances):
        for i, bn in enumerate(bins):
            if bins[i] < dist and dist <= bins[i+1] and len(empty_diction[i]) < 10:
                empty_diction[i].append(ind)

    all_values = []
    for val in empty_diction.values():
        all_values += val
    all_values = np.array(all_values) 
    
    selected_indices = actual_indices[all_values]
    selected_distances = distances[all_values]
    
    # plt.hist(distances, 100)
    # plt.hist(selected_distances, 100, color="red")
    # plt.savefig("z_.png")
    # exit(0)
    
    

    test_subset = torch.utils.data.Subset(test_data, selected_indices)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=128,
                                            shuffle=True, num_workers=4)


    def assign_indices_based_on_intervals(distances):
        # Define the intervals. In this example, we split [0, 1] into 10 equal intervals.
        num_intervals = 10
        interval_width = 1 / num_intervals
        intervals = [(i * interval_width, (i + 1) * interval_width) for i in range(num_intervals)]
        
        # Initialize a list to store the assigned indices
        assigned_indices = []
        
        # Assign an index to each distance based on the interval it falls into
        for distance in distances:
            assigned_index = None
            for i, (start, end) in enumerate(intervals):
                if start <= distance <= end:
                    assigned_index = i
                    break
            assigned_indices.append(assigned_index)
        
        return assigned_indices


    indices = []
    actual_indices = np.arange(len(testloader.dataset))
    for i, data in enumerate(testloader):
        dist = data[-1]
        index = np.array(assign_indices_based_on_intervals(dist)).reshape(-1, 1)
        if i == 0:
            indices = index
        else:
            indices = np.vstack((indices, index))

    indices = indices.reshape(1, -1)

    indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9:[]}
    for i in range(10):
        indexes[i] = actual_indices[indices.flatten() == i]
        print(len(indexes[i]))

    # print(indexes)
    # exit(0)

    for pt in paths:
        for alvsall in range(1000, 17000, 1000):
            for index_j, i in enumerate(range(10)):
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
                device = torch.device("cuda:3" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
                diffSolv = select_nn("SimpleCNN100")
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
                    myLog.logging.info(f'Dataset: TwoSourcesRdm')
                    
                    # trainloader, testloader, valloader = generateDatasets("/home/pb8294/data/", datasetName=ds, batch_size=BATCH_SIZE, num_workers=4, std_tr=0.1, s=100, transformation="linear").getDataLoaders()
                    # print(len(testloader.dataset)); exit()
            #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=100)
                    test_subset = torch.utils.data.Subset(test_data, indexes[i])
                    testloader = torch.utils.data.DataLoader(test_subset, batch_size=128,
                                                            shuffle=True, num_workers=NUM_WORKERS)

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
                modelName = next(iter(selectedDirs.keys()))
                saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_MSE-{i}.json')
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
                    myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
                    # for (j, ds) in enumerate(datasetNameList):
                    myLog.logging.info(f'Dataset: TwoSourcesRdm')
                    # trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()

                    test_subset = torch.utils.data.Subset(test_data, indexes[i])
                    testloader = torch.utils.data.DataLoader(test_subset, batch_size=128,
                                                            shuffle=True, num_workers=NUM_WORKERS)


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
                saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}-{i}.json')
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
                    myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
                    # for (j, ds) in enumerate(datasetNameList):
                    myLog.logging.info(f'Dataset: TwoSourcesRdm')
                    # trainloader, testloader, valloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=100, transformation="linear").getDataLoaders()

                    test_subset = torch.utils.data.Subset(test_data, indexes[i])
                    testloader = torch.utils.data.DataLoader(test_subset, batch_size=128,
                                                            shuffle=True, num_workers=NUM_WORKERS)


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
                saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_loss-{i}.json')
                myLog.logging.info(f'JSON object saved')
   