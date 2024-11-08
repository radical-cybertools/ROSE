import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from .loaders import generateDatasets, transformation_inverse
from .tools import errInSample
import torch


class myPlots():
    def __init__(self, adaptive=0):
        self.adaptive = adaptive
        
    def clasPlots(self, error_list, acc, accTrain, epoch):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(error_list, 'b*-', lw=3, ms=12)
        ax1.set(ylabel='loss', title='Epoch {}'.format(epoch + 1))
        ax2.plot(acc, 'r*-', lw=3, ms=12)
        ax2.plot(accTrain, 'g*-', lw=3, ms=12)
        ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
        plt.show()

    def plotDiff(self, PATH, dir, device, error_list, error_list_test, testloader, diff, epoch,
                 transformation="linear", bash=True):
        # (x, y, _, _, _, _, _, _, _) = next(iter(testloader))
        (x, y) = next(iter(testloader))
        # y = next(iter(testloader))[1]
        yhat = diff(x.to(device))
        if self.adaptive == 1:
            yhat = yhat.mean(0)
        yhat, y = transformation_inverse(yhat, y, transformation)
        #if transformation == "sqrt":
        #    yhat = yhat.pow(2)
        #    y = y.pow(2)
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].plot(error_list, 'b*-', lw=3, ms=12)
        # axs[0,0].set(ylabel='Loss', title='Epoch {}'.format(epoch+1))
        # axs[0, 1].imshow(vutils.make_grid(yhat.to(device)[:25], padding=2, normalize=False, value_range=(-1,1),  nrow=5).cpu()[1,:,:],cmap='cividis', interpolation='nearest')
        # frame1 = fig.gca()
        # frame1.axes.get_xaxis().set_visible(False)
        # frame1.axes.get_yaxis().set_visible(False)
        #
        # axs[1, 0].plot(error_list, 'r*-', lw=3, ms=12)
        # axs[1,0].set(ylabel='Loss', title='')
        #
        # axs[1, 1].imshow(vutils.make_grid(y.to(device), padding=2, normalize=False, value_range=(-1,1),  nrow=5).detach().cpu()[1,:,:],cmap='cividis', interpolation='nearest')
        # plt.show()
        # plt.imshow(vutils.make_grid((theModel(r[0].to(device))[1:2])/1.85 - r[1].to(device)[1:2], padding=2, normalize=False, value_range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],cmap='cividis', interpolation='nearest')
        # plt.colorbar()
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 8.5))
        axs[0, 0].plot(error_list, 'b*-', lw=3, ms=12)
        axs[0, 0].plot(error_list_test, 'r*-', lw=3, ms=12)
        axs[0, 0].set(ylabel='Loss', title='Epoch {}'.format(epoch))

        im = axs[0, 1].imshow(vutils.make_grid(
            yhat.to(device)[:25], padding=2, normalize=False, value_range=(-1, 1), nrow=5).detach().cpu().numpy()[1, :, :],
                              cmap='cividis', interpolation='nearest')
        axs[0, 1].set_title('Prediction')
        fig.colorbar(im, ax=axs[0, 1])
        # plt.colorbar()

        frame1 = fig.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        # axs[1, 0].plot(error_list, 'r*-', lw=3, ms=12)
        # # axs[1,0].colorbar()
        # axs[1,0].set(ylabel='Loss', title='')

        im = axs[1, 1].imshow(vutils.make_grid(y.to(device)[:25], padding=2, normalize=False, value_range=(-1, 1),
                                               nrow=5).detach().cpu().numpy()[1, :, :], cmap='cividis',
                              interpolation='nearest')
        fig.colorbar(im, ax=axs[1, 1])

        im = axs[1, 0].imshow(vutils.make_grid(x.to(device)[:25], padding=2, normalize=False, value_range=(-1, 1),
                                               nrow=5).detach().cpu().numpy()[1, :, :], cmap='cividis',
                              interpolation='nearest')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[2, 0].imshow(
            vutils.make_grid(y.to(device)[:25] - yhat.to(device)[:25], padding=2, normalize=False, value_range=(-1, 1),
                             nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
        fig.colorbar(im, ax=axs[2, 0])

        im = axs[2, 1].imshow(
            vutils.make_grid(y.to(device)[:25] - (yhat.to(device) / yhat.to(device).max())[:25], padding=2,
                             normalize=False, value_range=(-1, 1), nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic',
            interpolation='nearest')
        fig.colorbar(im, ax=axs[2, 1])
        if bash == False:
            plt.show()
        # Adding this here for the time being
        os.path.isdir(PATH + "Plots/") or os.mkdir(PATH + "Plots/")
        path = PATH + "Plots/" + dir + "/"
        os.path.isdir(path) or os.mkdir(path)
        fig.savefig(path + f'Epoch_{epoch}.png', dpi=fig.dpi)

    def plotGANs(self, error_list_D, error_list_G, testloader, gen, epoch):
        real_batch = next(iter(testloader))
        fake_batch = gen(torch.randn(25, 100).to(device))
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(error_list_D, 'b*-', lw=3, ms=12)
        axs[0, 0].set(ylabel='Disc Loss', title='Epoch {}'.format(epoch + 1))
        axs[0, 1].imshow(
            vutils.make_grid(real_batch[0].to(device)[:25], padding=2, normalize=False, value_range=(-1, 1), nrow=5).cpu()[1,
            :, :], cmap='cividis', interpolation='nearest')
        frame1 = fig.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        axs[1, 0].plot(error_list_G, 'r*-', lw=3, ms=12)
        axs[1, 0].set(ylabel='Gen Loss', title='')

        axs[1, 1].imshow(
            vutils.make_grid(fake_batch.to(device), padding=2, normalize=False, value_range=(-1, 1), nrow=5).detach().cpu()[1,
            :, :], cmap='cividis', interpolation='nearest')
        plt.show()
        
        
# @torch.no_grad()
def plotSamp(theModel, testloader, dict, device, PATH, plotName, n=1, adaptive=0):
    data = next(iter(testloader))
    (x, y, _, _, _, _, _) = data

    yhat = theModel(x.to(device))
    if adaptive == 1:
        yhat = yhat.mean(0)
    yhat, y = transformation_inverse(yhat, y, dict['transformation'])
    
    er, erF, erS, erList, erFList, erSList = errInSample(data, device, theModel)
    

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(80.5, 80.5))

    im = axs.imshow(
        vutils.make_grid(y.to(device)[:n] - yhat.to(device)[:n], padding=2, normalize=False, value_range=(-1, 1),
                         nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
    axs.xaxis.set_tick_params(labelsize=50)
    axs.yaxis.set_tick_params(labelsize=50)
    axs.tick_params(axis='both', which='major', labelsize=50)
    axs.tick_params(axis='both', which='minor', labelsize=50)
    axs.set_title(f'all: {erList[0]}, Src: {erSList[0]}, Field: {erFList[0]}',fontsize=70)
    cb = fig.colorbar(im, ax=axs)
    cb.ax.tick_params(labelsize=50)

    plt.savefig(os.path.join(PATH, "AfterPlots", "Samples", plotName), transparent=False)
    # plt.show()
    
def plotSampRelative(theModel, testloader, dict, device, PATH, plotName, n=1, TOL = 10**(-15), maxvalue=1.0, power = 1.0, adaptive=0):
    data = next(iter(testloader))
    (x, y, _, _, _, _, _) = data

    yhat = theModel(x.to(device))
    if adaptive == 1:
        yhat = yhat.mean(0)
    yhat, y = transformation_inverse(yhat, y, dict['transformation'])
    
    er, erF, erS, erList, erFList, erSList = errInSample(data, device, theModel)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(80.5, 80.5)) #+ torch.ones_like(y[:n]).to(device) * TOL
    
    t = (y.to(device)[:n] - yhat.to(device)[:n])/(y.to(device)[:n] + torch.ones_like(y[:n]).to(device) * TOL)
    im = axs[0,0].imshow(
        vutils.make_grid(torch.sign(t) * torch.min(torch.Tensor([maxvalue]).to(device),torch.abs(t)), padding=2, normalize=False, value_range=(-1, 1),
                         nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
    axs[0,0].xaxis.set_tick_params(labelsize=50)
    axs[0,0].yaxis.set_tick_params(labelsize=50)
    axs[0,0].tick_params(axis='both', which='major', labelsize=50)
    axs[0,0].tick_params(axis='both', which='minor', labelsize=50)
    axs[0,0].set_title(f'Relative error. Cutoff set to {maxvalue}',fontsize=70)
    cb = fig.colorbar(im, ax=axs[0,0])
    cb.ax.tick_params(labelsize=50)

    im = axs[0,1].imshow(
        vutils.make_grid((y.to(device)[:n] - yhat.to(device)[:n]), padding=2, normalize=False, value_range=(-1, 1),
                         nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
    axs[0,1].xaxis.set_tick_params(labelsize=50)
    axs[0,1].yaxis.set_tick_params(labelsize=50)
    axs[0,1].tick_params(axis='both', which='major', labelsize=50)
    axs[0,1].tick_params(axis='both', which='minor', labelsize=50)
    axs[0,1].set_title(f'Error \n all: {np.round(erList[0],5)}, \n Src: {np.round(erSList[0],5)}, \n Field: {np.round(erFList[0],5)}',fontsize=70)
    cb = fig.colorbar(im, ax=axs[0,1])
    cb.ax.tick_params(labelsize=50)
    
    im = axs[1,0].imshow(
        vutils.make_grid((y.to(device)[:n] - yhat.to(device)[:n]) * ( 2.0 * torch.ones_like(y[:n]).to(device) - y[:n].to(device) ), padding=2, normalize=False, value_range=(-1, 1),
                         nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
    axs[1,0].xaxis.set_tick_params(labelsize=50)
    axs[1,0].yaxis.set_tick_params(labelsize=50)
    axs[1,0].tick_params(axis='both', which='major', labelsize=50)
    axs[1,0].tick_params(axis='both', which='minor', labelsize=50)
    axs[1,0].set_title(f'Interpolation b/ relative and absolute error. \n Power set to 1',fontsize=70)
    cb = fig.colorbar(im, ax=axs[1,0])
    cb.ax.tick_params(labelsize=50)
    
    im = axs[1,1].imshow(
        vutils.make_grid((y.to(device)[:n] - yhat.to(device)[:n]) * ( torch.pow(y[:n].to(device), power-1.0) + torch.pow(torch.ones_like(y[:n]).to(device) - y[:n].to(device), power) ), padding=2, normalize=False, value_range=(-1, 1),
                         nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
    axs[1,1].xaxis.set_tick_params(labelsize=50)
    axs[1,1].yaxis.set_tick_params(labelsize=50)
    axs[1,1].tick_params(axis='both', which='major', labelsize=50)
    axs[1,1].tick_params(axis='both', which='minor', labelsize=50)
    axs[1,1].set_title(f'Interpolation b/ relative and absolute error. \n Power set to {power}',fontsize=70)
    cb = fig.colorbar(im, ax=axs[1,1])
    cb.ax.tick_params(labelsize=50)

    plt.savefig(os.path.join(PATH, "AfterPlots", "Samples", plotName), transparent=False)
    # plt.show()
