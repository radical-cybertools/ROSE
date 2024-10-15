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


# sys.path.insert(1, '/home/javier/Projects/DiffSolver/DeepDiffusionSolver/util')
# sys.path.insert(1, '/home/javier/Projects/DiffSolverAdaptive/util')
sys.path.insert(1, '../util')


from util.loaders import generateDatasets, inOut, saveJSON, loadJSON#, MyData
from util.NNets import SimpleCNN100, JuliaCNN100, JuliaCNN100_2, UNetAB
from util.tools import accuracy, tools, per_image_error, predVsTarget
from util.plotter import myPlots, plotSamp, plotSampRelative

# class DiffSur(SimpleCNN):
#     pass
def select_nn(arg, d=None, num_samples=1):
    if arg == "SimpleCNN100":
        class DiffSur(SimpleCNN100):
            pass
        return DiffSur()
    elif arg == "UNETAB":
        class DiffSur(UNetAB):
            pass
        return DiffSur()
    # elif arg == "JuliaCNN100":
    #     class DiffSur(JuliaCNN100):
    #         pass
    #     return DiffSur()
    # elif arg == "JuliaCNN100_2":
    #     class DiffSur(JuliaCNN100_2):
    #         pass
    #     return DiffSur()
    # elif arg == "SimpleAdapCNN100":
    #     class DiffSur(SimpleAdapCNN100):
    #         def __init__(self, d):
    #             super(DiffSur, self).__init__(d)
    #     return DiffSur(d)
    # elif arg == "SimpleAdap2CNN100":
    #     class DiffSur(SimpleAdap2CNN100):
    #         def __init__(self, d, num_samples=2):
    #             super(DiffSur, self).__init__(d, num_samples)
    #     return DiffSur(d,num_samples)
#     return DiffSur()

def inverse_huber_loss(target,output, C=0.5):
    absdiff = torch.abs(output-target)
#     C = 0.5#*torch.max(absdiff).item() ---->> 0.05
#     return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
    return torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) )


class train(object):
    def __init__(self, latentSpace=100, std_tr=0.0, s=512, transformation="linear", wtan=10, w=1, w2=6000, select="step", alph=1, delta=0.02, toggleIdx=1, p=0.5):
        self.real_label = 1.0
        self.fake_label = 0.0
        self.latentSpace = latentSpace
        self.std_tr = std_tr
        self.s = s
        self.trans = transformation
        dict["transformation"] = self.trans
        dict["w"] = w
        dict["wtan"] = wtan
        dict["w2"] = w2
        dict["alph"] = alph
        dict["lossSelection"] = select
        dict["delta"] = delta
        dict["togIdx"] = toggleIdx
        dict["p"] = p
        
    def my_loss(self, output, target, ep, dict):
        if dict["lossSelection"] == "step":
            loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) *dict["w2"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "exp":
            loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "huber":
            loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.nn.HuberLoss(reduction='none', delta=dict["delta"])(output, target))
        elif dict["lossSelection"] == "toggle":
            if np.mod(np.divmod(ep, dict["togIdx"])[0], 2) == 0:
                loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.abs((output - target)**dict["alph"]))
            else:
                loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "rand":
#             r = np.random.rand()            
            if dict["r"]<dict["p"]:
                loss = torch.mean((1 + torch.tanh(dict["wtan"]*target) * dict["w2"]) * torch.abs((output - target)**dict["alph"]))
            else:
                loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * torch.abs((output - target)**dict["alph"]))
        elif dict["lossSelection"] == "invhuber":
            loss = torch.mean(torch.exp(-torch.abs(torch.ones_like(output) - output)/dict["w"]) * inverse_huber_loss(target,output, C=dict["delta"]))
        return loss

    def trainClass(self, epochs=100, snap=25):
        clas = MLP().to(device)
        criterion = nn.NLLLoss()
        # criterion = my_loss
        opt = optim.Adam(clas.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list = []
        acc, accTrain = [], []
        for epoch in range(epochs):
            error = 0.0
            for (i, data) in enumerate(trainloader):
                clas.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = clas(x)
                err = criterion(yhat,y)
                err.backward()
                opt.step()
                error += err.item()
                # if i > 2:
                #     break
            error_list.append(error/(i+1))
            acc.append(accuracy().validation(testloader, clas).item())
            accTrain.append(accuracy().validation(trainloader, clas).item())
            # acc.append(i)
            # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            # ax1.plot(error_list, 'b*-', lw=3, ms=12)
            # ax1.set(ylabel='loss', title='Epoch {}'.format(epoch+1))
            # ax2.plot(acc, 'r*-', lw=3, ms=12)
            # ax2.plot(accTrain, 'g*-', lw=3, ms=12)
            # ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
            # plt.show()
            myPlots().clasPlots(error_list, acc, accTrain, epoch)

            if epoch % snap == snap-1 :
                inOut().save_model(clas, 'Class', opt, error_list, epoch, dir)
        print("Done!!")
        return error_list, acc, accTrain

    def trainDiffSolver(self, epochs=100, snap=25, bashTest=False):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Training Diff Solver')
#         diffSolv = DiffSur().to(device)
        diffSolv = select_nn(dict["NN"])
        diffSolv = diffSolv.to(device)
        # criterion = nn.NLLLoss()
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = [], []
        acc, accTrain = [], []
#         myPlots().plotDiff(PATH, dir, device, [1,2,3], [2,3,4], testloader, diffSolv, 1, transformation=self.trans, bash=True)
        for epoch in range(epochs):
            dict["r"] = np.random.rand()
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
#                 print(f'Epoch: {epoch}/{epochs}. Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if dict["NN"] != "SimpleCNN100":
                self.updatePs(trainloader, diffSolv, epoch, dict)            
            diffSolv.eval()
            myPlots().plotDiff_JQTM(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) #<----------------------------------------------
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain

    def continuetrainDiffSolver(self, epochs=100, snap=25):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Continue Training Diff Solver')
#         diffSolv = DiffSur().to(device)
        diffSolv = select_nn(dict["NN"])
        diffSolv = diffSolv.to(device)
        # criterion = nn.NLLLoss()
        # load_model(self, module, dict)
        lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=dict["lr"])
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        myLog.logging.info(f'Last Epoch = {lastEpoch}')
        error_list, error_list_test = dict["Loss"], dict["LossTest"]
        acc, accTrain = [], []
        for epoch in range(lastEpoch+1, epochs):
            dict["r"] = np.random.rand()
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)
                err = criterion(yhat,y, epoch, dict)
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
#                 print(f'Epoch: {epoch}/{epochs}. Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if dict["NN"] != "SimpleCNN100":
                self.updatePs(trainloader, diffSolv, epoch, dict) 
            
            diffSolv.eval()
            myPlots().plotDiff_JQTM(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True)

            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain
    
    def trainDiffSolverAdapt(self, epochs=100, snap=25, bashTest=False):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Training Diff Solver')
        diffSolv = select_nn(dict["NN"], dict, 1)
        diffSolv = diffSolv.to(device)
#         diffSolv = SimpleAdapCNN100(dict).to(device) #select_nn(dict["NN"])
#         diffSolv = diffSolv().to(device)
        # criterion = nn.NLLLoss()
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=lr)
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        error_list, error_list_test = [], []
        acc, accTrain = [], []
        self.initInferenceDict(diffSolv)
#         myPlots().plotDiff(PATH, dir, device, [1,2,3], [2,3,4], testloader, diffSolv, 1, transformation=self.trans, bash=True)
        for epoch in range(epochs):
            diffSolv.train()
            dict["r"] = np.random.rand()
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
#                 try:
#                     y = diffSolv.replicaTarget(y)
#                 except:
#                     pass
                yhat = diffSolv(x)
                err_do = diffSolv.estimate_ELBO()
                err = criterion(yhat,y, epoch, dict) + err_do
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
#                 print(f'Epoch: {epoch}/{epochs}. Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                if bashTest:
                    break
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            dict["Inference"] = self.updateInferenceDict(diffSolv)
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir) 
            diffSolv.eval()
            myPlots().plotDiff_JQTM(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True) #<----------------------------------------------
            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
            if bashTest:
                break
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain

    def continuetrainDiffSolverAdapt(self, epochs=100, snap=25):
        myLog = inOut()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info('Continue Training Diff Solver')
        diffSolv = select_nn(dict["NN"], dict, 2)
        diffSolv = diffSolv.to(device)
#         diffSolv = SimpleAdapCNN100(dict).to(device)  #select_nn(dict["NN"])
#         diffSolv = diffSolv().to(device)
        # criterion = nn.NLLLoss()
        # load_model(self, module, dict)
        lastEpoch, _, diffSolv = inOut().load_model(diffSolv, "Diff", dict)
        criterion = self.my_loss
        opt = optim.Adam(diffSolv.parameters(), lr=dict["lr"])
        trainloader, testloader = generateDatasets(PATH, datasetName=DATASETNAME, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s, transformation=self.trans).getDataLoaders()
        myLog.logging.info(f'Last Epoch = {lastEpoch}')
        error_list, error_list_test = dict["Loss"], dict["LossTest"]
        acc, accTrain = [], []
        self.initInferenceDict(diffSolv)
        for epoch in range(lastEpoch+1, epochs):
            diffSolv.train()
            dict["r"] = np.random.rand()
            error = 0.0
            for (i, data) in enumerate(trainloader):
                diffSolv.zero_grad()
                x = data[0].to(device)
                y = data[1].to(device)
                yhat = diffSolv(x)
                err_do = diffSolv.estimate_ELBO()
                err = criterion(yhat,y, epoch, dict) + err_do
                error += err.item()
                err.backward()
                opt.step()
                # if i > 2:
                #     break
#                 print(f'Epoch: {epoch}/{epochs}. Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
                myLog.logging.info(f'Epoch: {epoch}/{epochs}, Minibatch: {i}/{len(trainloader)}, error: {error/(i+1)}')
            error_list.append(error/(i+1))
            error_list_test.append(self.computeError(testloader, diffSolv, epoch, dict))
            dict["Inference"] = self.updateInferenceDict(diffSolv)
            self.saveBestModel(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)  
            diffSolv.eval()
            myPlots().plotDiff_JQTM(PATH, dir, device, error_list, error_list_test, testloader, diffSolv, epoch, transformation=self.trans, bash=True)

            if epoch % snap == snap-1 :
                inOut().save_model(PATH, dict, diffSolv, 'Diff', opt, error_list, error_list_test, epoch, dir)
        myLog.logging.info("Done!!")
        return error_list#, acc, accTrain

    def computeError(self, testloader, theModel, ep, dict):  
        criterion = self.my_loss
        with torch.no_grad():
            erSum = 0
            for (i, data) in enumerate(testloader):
                x = data[0].to(device)
                y = data[1].to(device)
#                 erSum += torch.mean(torch.abs(theModel(x) - y)).item()
                erSum += criterion(theModel(x), y, ep, dict).item()

            return erSum / len(testloader)
    
    def saveBestModel(self, PATH, dict, theModel, module, opt, error_list, error_list_test, epoch, dir, tag='Best'):
        if error_list_test.index(np.min(error_list_test)) == len(error_list_test)-1:
            inOut().save_model(PATH, dict, theModel, module, opt, error_list, error_list_test, epoch, dir, tag=tag)
            #save model
    
    def updatePs(self, trainloader, theModel, ep, dict, lr_a = 0.001):  
        criterion = self.my_loss
        with torch.no_grad():
            erSum1 = 0
            erSum2 = 0
            p1 = theModel.p1
            for (i, data) in enumerate(trainloader):
                x = data[0].to(device)
                y = data[1].to(device)
#                 erSum += torch.mean(torch.abs(theModel(x) - y)).item()
                erSum1 += criterion(theModel.nn1(x), y, ep, dict).item()
                erSum2 += criterion(theModel.nn2(x), y, ep, dict).item()
            p1 = p1 - dict["lr_a"] * (erSum1 - erSum2)
            p1 = np.max([0.0,p1])
            p1 = np.min([1.0,p1])
            theModel.p1 = p1
            theModel.p2 = 1.0 - p1
            if p1 == 1 or p1 == 0:
                dict["lr_a"] = dict["lr_a"]/10
            if np.sum([list(dict.keys())[i] == "p1" for i in range(len(dict))]) == 0:
                dict["p1"] = [theModel.p1]
            else:
                dict["p1"].append(theModel.p1)
                
    def initInferenceDict(self, model):
        self.infDict ={"Enc" : {"beta_a": {}, "beta_b": {}, "kl": {} }, "Dec" : {"beta_a": {}, "beta_b": {}, "kl": {} } }
        for i in range(len(model.seqIn)):
            self.infDict["Enc"]['beta_a'][i] = []
            self.infDict["Enc"]['beta_b'][i] = []
            self.infDict["Enc"]['kl'][i] = []
        for i in range(len(model.seqOut)):
            self.infDict["Dec"]['beta_a'][i] = []
            self.infDict["Dec"]['beta_b'][i] = []
            self.infDict["Dec"]['kl'][i] = []
            
    def updateInferenceDict(self, model):
        for i in range(len(model.seqIn)):
            beta_a, beta_b = model.seqIn[i][-1].get_variational_params()
            kl_loss = model.seqIn[i][-1].get_kl().item()
            self.infDict["Enc"]['beta_a'][i].append(f'{beta_a.cpu().detach().numpy().mean()}')
            self.infDict["Enc"]['beta_b'][i].append(f'{beta_b.cpu().detach().numpy().mean()}')
            self.infDict["Enc"]['kl'][i].append(kl_loss)
        for i in range(len(model.seqOut)):
            beta_a, beta_b = model.seqOut[i][-1].get_variational_params()
            kl_loss = model.seqOut[i][-1].get_kl().item()
            self.infDict["Dec"]['beta_a'][i].append(f'{beta_a.cpu().detach().numpy().mean()}')
            self.infDict["Dec"]['beta_b'][i].append(f'{beta_b.cpu().detach().numpy().mean()}')
            self.infDict["Dec"]['kl'][i].append(kl_loss)
            
        return self.infDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Deep Diffusion Solver")
    parser.add_argument('--path', dest="path", type=str, default="/raid/javier/Datasets/DiffSolverAdaptive/",
                        help="Specify path to dataset")
    parser.add_argument('--dataset', dest="dataset", type=str, default="All",
                        help="Specify dataset")
    parser.add_argument('--dir', dest="dir", type=str, default="100DGX-" + str(datetime.now()).split(" ")[0],
                        help="Specify directory name associated to model")
    parser.add_argument('--bashtest', dest="bashtest", type=bool, default=False,
                        help="Leave default unless testing flow")
    
    parser.add_argument('--nn', dest="nn", type=str, default="SimpleCNN100",
                        help="Select between SimpleCNN100, JuliaCNN100, SimpleAdapCNN100, SimpleAdap2CNN100")
    
    parser.add_argument('--bs', dest="bs", type=int, default=50,
                        help="Specify Batch Size")
    parser.add_argument('--nw', dest="nw", type=int, default=8,
                        help="Specify number of workers")
    parser.add_argument('--ngpu', dest="ngpu", type=int, default=1,
                        help="Specify ngpu. (Never have tested >1)")
    parser.add_argument('--lr', dest="lr", type=float, default=0.0001,
                        help="Specify learning rate")
    parser.add_argument('--lra', dest="lr_a", type=float, default=0.0001,
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
    parser.add_argument('--p', dest="p", type=int, default=1,
                        help="Specify hparam p")
    
    parser.add_argument("--prior_temp", type=float, default=1.,
                        help="Temperature for prior Concrete Bernoulli ")
    parser.add_argument("--temp", type=float, default=.1,
                        help="Temperature for posterior Concrete Bernoulli")
    parser.add_argument("--a_prior", type=float, default=4.5,
                        help="a parameter for Beta distribution")
    parser.add_argument("--b_prior", type=float, default=0.5,
                        help="b parameter for Beta distribution")
    parser.add_argument("--num_samples", type=float, default=5,
                        help="Number of samples of Z matrix. Currently not used.")
      
    args = parser.parse_args()
    
    
    ###Start Here
#     if args.nn == "SimpleCNN100":
#         class DiffSur(SimpleCNN100):
#             pass       
#     elif args.nn == "JuliaCNN100":
#         class DiffSur(JuliaCNN100):
#             pass
        
    PATH = args.path # "/raid/javier/Datasets/DiffSolver/"
    DATASETNAME = args.dataset # "All"

    dir = args.dir #'1DGX' #'Test'#str(21)
    BATCH_SIZE=args.bs #50
    NUM_WORKERS=args.nw #8
    ngpu = args.ngpu #1
    lr = args.lr #0.0001
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if args.newdir:
        dict = inOut().newDict(PATH, dir)
    else:
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))


#     dict
    if args.newtrain:
        dict["NN"] = args.nn
        dict["lr"]=lr
        dict["lr_a"] = args.lr_a
        dict["temp"] = args.temp
        dict["prior_a"] = args.a_prior
        dict["prior_b"] = args.b_prior

        if dict["NN"] in ["SimpleAdapCNN100", "SimpleAdap2CNN100"]:
            error_list = train(std_tr=0.0, s=100, transformation=args.transformation, wtan=args.wtan, w=args.w, alph=args.alpha, w2=args.w2, select=args.loss, delta=args.delta, toggleIdx=args.toggle, p=args.p).trainDiffSolverAdapt(args.maxep,1, bashTest=args.bashtest)
        else:
            error_list = train(std_tr=0.0, s=100, transformation=args.transformation, wtan=args.wtan, w=args.w, alph=args.alpha, w2=args.w2, select=args.loss, delta=args.delta, toggleIdx=args.toggle, p=args.p).trainDiffSolver(args.maxep,1, bashTest=args.bashtest)
        
    else:
        if dict["NN"] in ["SimpleAdapCNN100", "SimpleAdap2CNN100"]:
            error_list = train(std_tr=0.0, s=100, transformation=args.transformation, wtan=args.wtan, w=args.w, alph=args.alpha, w2=args.w2, select=args.loss, delta=args.delta, toggleIdx=args.toggle, p=args.p).continuetrainDiffSolverAdapt(args.maxep,1)
        else:
            error_list = train(std_tr=0.0, s=100, transformation=dict["transformation"], wtan=dict["wtan"], w=dict["w"], alph=dict["alph"], w2=dict["w2"], select=dict["lossSelection"], delta=dict["delta"], toggleIdx=dict["togIdx"], p=dict["p"]).continuetrainDiffSolver(args.maxep,1)
  

   
# python trainModel-100.py --dataset TwoSourcesRdm --dir AL1 --newdir True --newtrain True --transformation linear --loss exp
# python trainModel-100.py --dataset TwoSourcesRdm --dir AL2 --newdir True --newtrain True --transformation linear --loss step
# python trainModel-100.py --dataset TwoSourcesRdm --dir AL3 --newdir True --newtrain True --transformation linear --loss exp --nn JuliaCNN100
# python trainModel-100.py --dataset TwoSourcesRdm --dir AL4 --newdir True --newtrain True --transformation linear --loss step --nn JuliaCNN100

# python trainModel-100.py --dataset TwoSourcesRdm --dir AL8 --newdir True --newtrain True --transformation linear --loss exp --nn SimpleAdapCNN100

