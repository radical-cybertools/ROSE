import torch
import torch.nn as nn
from .loaders import generateDatasets, transformation_inverse
import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib


@torch.no_grad()
def per_image_error(neural_net, loader, device, transformation="linear",
                    error_fnc=nn.L1Loss(reduction='none')):
    neural_net.eval()
    error1 = 0.0
    
    error1_field = 0.0
    
    error1_src = 0.0
    
    error1_per_im = []
    
    error1_per_im_field = []
    
    error1_per_im_src = []
    
    for i, data in enumerate(loader):
        x = data[0].to(device)
        
        srcs = x > 0
        
        nan_srcs = torch.where(srcs, float('nan'), 1.0)
        nan_rest = torch.where(srcs, 1.0, float('nan'))
        
        y = data[1].to(device)
        
        yhat = neural_net(x)
        
        yhat, y = transformation_inverse(yhat, y, transformation)
        
        e1 = error_fnc(yhat,y)
        
        e1_srcs = nan_rest * e1
        e1_field = nan_srcs * e1
        
        
        error1 += np.mean(e1.cpu().numpy())
        
        error1_field += np.nanmean(e1_field.cpu().numpy())
        error1_src += np.nanmean(e1_srcs.cpu().numpy())
                
        e1_list =[]
        
        e1_list_field =[]
        
        e1_list_src =[]
        
        
        for j in range(e1.shape[0]):
            e1_list.append(np.mean(e1[j].cpu().numpy()))
            
            e1_list_field.append(np.nanmean(e1_field[j].cpu().numpy()))
            
            e1_list_src.append(np.nanmean(e1_srcs[j].cpu().numpy()))
            
            
        error1_per_im.extend(e1_list)
        
        error1_per_im_field.extend(e1_list_field)
        
        error1_per_im_src.extend(e1_list_src)
        
    return error1/(i+1), error1_field/(i+1),  error1_src/(i+1),  \
           error1_per_im, error1_per_im_field, error1_per_im_src


@torch.no_grad()
def predVsTarget(loader, neural_net, device, transformation = "linear", threshold = 0.0, nbins = 100, BATCH_SIZE = 30, size = 512, lim = 10):
    l_real, l_pred = np.array([]), np.array([])
    if lim == 0 or lim > len(loader):
        lim = len(loader)
    with torch.no_grad():
        for (i, data) in enumerate(loader):
            if i > lim:
                break
            x = data[0].to(device)
            y = data[1].to(device)
            pred = neural_net(x)
            pred, y = transformation_inverse(pred, y, transformation)
#             try:
#                 if transformation == "sqrt":
#                     pred = pred.pow(2)
#                     y = y.pow(2)
#             except:
#                 pass

            l_pred = np.append(l_pred,pred.reshape(BATCH_SIZE*size*size).cpu().numpy())
            l_real = np.append(l_real,y.reshape(BATCH_SIZE*size*size).cpu().numpy())
            
    # create data 
    x = l_real[l_real >= threshold]
    y = l_pred[l_real >= threshold]

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#     plt.pcolormesh(xi, yi, np.power(zi.reshape(xi.shape) / zi.reshape(xi.shape).max(),1/8), shading='auto')
    return xi, yi, zi

@torch.no_grad()    
def errInSample(data, device, theModel):
    error1 = 0.0
    
    error1_field = 0.0
    
    error1_src = 0.0
    
    error1_per_im = []
    
    error1_per_im_field = []
    
    error1_per_im_src = []
    
#     for i, data in enumerate(loader):
#     data = next(iter(testloader))
    x = data[0].to(device)

    srcs = x > 0

    nan_srcs = torch.where(srcs, float('nan'), 1.0)
    nan_rest = torch.where(srcs, 1.0, float('nan'))

    y = data[1].to(device)

    yhat = theModel(x)

    # yhat, y = transformation_inverse(yhat, y, dict['transformation'])

    e1 = nn.L1Loss(reduction='none')(yhat,y.to(device))

    e1_srcs = nan_rest * e1
    e1_field = nan_srcs * e1


    error1 += np.mean(e1.cpu().numpy())

    error1_field += np.nanmean(e1_field.cpu().numpy())
    error1_src += np.nanmean(e1_srcs.cpu().numpy())

    e1_list =[]

    e1_list_field =[]

    e1_list_src =[]


    for j in range(e1.shape[0]):
        e1_list.append(np.mean(e1[j].cpu().numpy()))

        e1_list_field.append(np.nanmean(e1_field[j].cpu().numpy()))

        e1_list_src.append(np.nanmean(e1_srcs[j].cpu().numpy()))


    error1_per_im.extend(e1_list)

    error1_per_im_field.extend(e1_list_field)

    error1_per_im_src.extend(e1_list_src)

    return error1, error1_field,  error1_src,  \
           error1_per_im, error1_per_im_field, error1_per_im_src

@torch.no_grad()    
def errInDS(neural_net, loader, device, transformation="linear",
                    error_fnc=nn.L1Loss(reduction='none'), tol1 = 0.2, tol2 = 0.1, tol3 = 0.05):
    error1 = 0.0
    
    error1_field = 0.0
    
    error1_src = 0.0
    error1_ring1 = 0.0
    error1_ring2 = 0.0
    error1_ring3 = 0.0
    
    errorMax = 0.0
    
    errorMax_field = 0.0
    
    errorMax_src = 0.0
    errorMax_ring1 = 0.0
    errorMax_ring2 = 0.0
    errorMax_ring3 = 0.0
    
    errorMaxm = 0.0
    
    errorMaxm_field = 0.0
    
    errorMaxm_src = 0.0
    errorMaxm_ring1 = 0.0
    errorMaxm_ring2 = 0.0
    errorMaxm_ring3 = 0.0
    
    errorMin = 0.0
    
    errorMin_field = 0.0
    
    errorMin_src = 0.0
    errorMin_ring1 = 0.0
    errorMin_ring2 = 0.0
    errorMin_ring3 = 0.0
    
    errorMinm = 0.0
    
    errorMinm_field = 0.0
    
    errorMinm_src = 0.0
    errorMinm_ring1 = 0.0
    errorMinm_ring2 = 0.0
    errorMinm_ring3 = 0.0
    
#     error1_per_im = []
    
#     error1_per_im_field = []
    
#     error1_per_im_src = []
    save_x = torch.tensor([])
    save_y = torch.tensor([])
    save_pred = torch.tensor([])
    for i, data in enumerate(loader):
        x = data[0].to(device)
        y = data[1].to(device)
        
        srcs = x > 0
        rings1 = (y - y * torch.sign(x)) >= tol1
        rings2 = ( (y - y * torch.sign(x)) >= tol2 ) * ((y - y * torch.sign(x)) < tol1)
        rings3 = ( (y - y * torch.sign(x)) >= tol3 ) * ((y - y * torch.sign(x)) < tol2)
        
        nan_srcs = torch.where(srcs, float('nan'), 1.0)
        nan_rest = torch.where(srcs, 1.0, float('nan'))
        nan_notring1 = torch.where(rings1, 1.0, float('nan'))
        nan_notring2 = torch.where(rings2, 1.0, float('nan'))
        nan_notring3 = torch.where(rings3, 1.0, float('nan'))
        
        
        yhat = neural_net(x)
        
        
        yhat, y = transformation_inverse(yhat, y, transformation)
        
        e1 = error_fnc(yhat,y)
        
        e1_srcs = nan_rest * e1
        e1_field = nan_srcs * e1
        
        e1_ring1 = nan_notring1 * e1
        e1_ring2 = nan_notring2 * e1
        e1_ring3 = nan_notring3 * e1
        
        
#         error1 += np.mean(e1.cpu().numpy())
#         error1_field += np.nanmean(e1_field.cpu().numpy())
#         error1_src += np.nanmean(e1_srcs.cpu().numpy())
#         error1_ring1 += np.nanmean(e1_ring1.cpu().numpy())
#         error1_ring2 += np.nanmean(e1_ring2.cpu().numpy())
#         error1_ring3 += np.nanmean(e1_ring3.cpu().numpy())
        
#         errorMax = np.maximum(errorMax, np.nanmax(e1.cpu().numpy()))
#         errorMax_field = np.maximum(errorMax_field, np.nanmax(e1_field.cpu().numpy()))
#         errorMax_src = np.maximum(errorMax_src, np.nanmax(e1_srcs.cpu().numpy()))
#         errorMax_ring1 = np.maximum(errorMax_ring1, np.nanmax(e1_ring1.cpu().numpy()))
#         errorMax_ring2 = np.maximum(errorMax_ring2, np.nanmax(e1_ring2.cpu().numpy()))
#         errorMax_ring3 = np.maximum(errorMax_ring3, np.nanmax(e1_ring3.cpu().numpy()))
        
#         errorMaxm += np.nanmax(e1.cpu().numpy())
#         errorMaxm_field += np.nanmax(e1_field.cpu().numpy())
#         errorMaxm_src += np.nanmax(e1_srcs.cpu().numpy())
#         errorMaxm_ring1 += np.nanmax(e1_ring1.cpu().numpy())
#         errorMaxm_ring2 += np.nanmax(e1_ring2.cpu().numpy())
#         errorMaxm_ring3 += np.nanmax(e1_ring3.cpu().numpy())
        
#         errorMin = np.minimum(errorMin, np.nanmin(e1.cpu().numpy()))
#         errorMin_field = np.minimum(errorMin_field, np.nanmin(e1_field.cpu().numpy()))
#         errorMin_src = np.minimum(errorMin_src, np.nanmin(e1_srcs.cpu().numpy()))
#         errorMin_ring1 = np.minimum(errorMin_ring1, np.nanmin(e1_ring1.cpu().numpy()))
#         errorMin_ring2 = np.minimum(errorMin_ring2, np.nanmin(e1_ring2.cpu().numpy()))
#         errorMin_ring3 = np.minimum(errorMin_ring3, np.nanmin(e1_ring3.cpu().numpy()))
        
#         errorMinm += np.nanmin(e1.cpu().numpy())
#         errorMinm_field += np.nanmin(e1_field.cpu().numpy())
#         errorMinm_src += np.nanmin(e1_srcs.cpu().numpy())
#         errorMinm_ring1 += np.nanmin(e1_ring1.cpu().numpy())
#         errorMinm_ring2 += np.nanmin(e1_ring2.cpu().numpy())
#         errorMinm_ring3 += np.nanmin(e1_ring3.cpu().numpy())
        error1 += torch.mean(e1).cpu().numpy()
        error1_field += torch.nanmean(e1_field).cpu().numpy()
        error1_src += torch.nanmean(e1_srcs).cpu().numpy()
        error1_ring1 += torch.nanmean(e1_ring1).cpu().numpy()
        error1_ring2 += torch.nanmean(e1_ring2).cpu().numpy()
        error1_ring3 += torch.nanmean(e1_ring3).cpu().numpy()
        
        errorMax = np.maximum(errorMax, e1[~e1.isnan()].max().cpu().numpy())
        errorMax_field = np.maximum(errorMax_field, e1_field[~e1_field.isnan()].max().cpu().numpy())
        errorMax_src = np.maximum(errorMax_src, e1_srcs[~e1_srcs.isnan()].max().cpu().numpy() )
        errorMax_ring1 = np.maximum(errorMax_ring1, e1_ring1[~e1_ring1.isnan()].max().cpu().numpy())
        errorMax_ring2 = np.maximum(errorMax_ring2, e1_ring2[~e1_ring2.isnan()].max().cpu().numpy())
        errorMax_ring3 = np.maximum(errorMax_ring3, e1_ring3[~e1_ring3.isnan()].max().cpu().numpy())
        
        errorMaxm += e1[~e1.isnan()].max().cpu().numpy()
        errorMaxm_field += e1_field[~e1_field.isnan()].max().cpu().numpy()
        errorMaxm_src += e1_srcs[~e1_srcs.isnan()].max().cpu().numpy()
        errorMaxm_ring1 += e1_ring1[~e1_ring1.isnan()].max().cpu().numpy()
        errorMaxm_ring2 += e1_ring2[~e1_ring2.isnan()].max().cpu().numpy()
        errorMaxm_ring3 += e1_ring3[~e1_ring3.isnan()].max().cpu().numpy()
        
        errorMin = np.minimum(errorMin, e1[~e1.isnan()].min().cpu().numpy())
        errorMin_field = np.minimum(errorMin_field, e1_field[~e1_field.isnan()].min().cpu().numpy())
        errorMin_src = np.minimum(errorMin_src, e1_srcs[~e1_srcs.isnan()].min().cpu().numpy())
        errorMin_ring1 = np.minimum(errorMin_ring1, e1_ring1[~e1_ring1.isnan()].min().cpu().numpy())
        errorMin_ring2 = np.minimum(errorMin_ring2, e1_ring2[~e1_ring2.isnan()].min().cpu().numpy())
        errorMin_ring3 = np.minimum(errorMin_ring3, e1_ring3[~e1_ring3.isnan()].min().cpu().numpy())
        
        errorMinm += e1[~e1.isnan()].min().cpu().numpy()
        errorMinm_field += e1_field[~e1_field.isnan()].min().cpu().numpy()
        errorMinm_src += e1_srcs[~e1_srcs.isnan()].min().cpu().numpy()
        errorMinm_ring1 += e1_ring1[~e1_ring1.isnan()].min().cpu().numpy()
        errorMinm_ring2 += e1_ring2[~e1_ring2.isnan()].min().cpu().numpy()
        errorMinm_ring3 += e1_ring3[~e1_ring3.isnan()].min().cpu().numpy()
                

        
    return error1/(i+1), error1_field/(i+1),  error1_src/(i+1),  \
            errorMax, errorMax_field, errorMax_src, \
            errorMaxm/(i+1), errorMaxm_field/(i+1), errorMaxm_src/(i+1), \
            errorMin, errorMin_field, errorMin_src, \
            errorMinm/(i+1), errorMinm_field/(i+1), errorMinm_src/(i+1), \
            error1_ring1/(i+1), error1_ring2/(i+1),  error1_ring3/(i+1),  \
            errorMax_ring1, errorMax_ring2, errorMax_ring3, \
            errorMaxm_ring1/(i+1), errorMaxm_ring2/(i+1), errorMaxm_ring3/(i+1), \
            errorMin_ring1, errorMin_ring2, errorMin_ring3, \
            errorMinm_ring1/(i+1), errorMinm_ring2/(i+1), errorMinm_ring3/(i+1)

@torch.no_grad()    
def plotFigureDS(neural_net, loader, device, transformation="linear",
                    error_fnc=nn.L1Loss(reduction='none'), tol1 = 0.2, tol2 = 0.1, tol3 = 0.05, it=1, pref="w1"):
    save_x = torch.tensor([])
    save_y = torch.tensor([])
    save_pred = torch.tensor([])
    for i, data in enumerate(loader):
        x = data[0].to(device)
        y = data[1].to(device)
        yhat = neural_net(x)
        
        if i == 0:
            save_x = x
            save_y = y
            save_pred = yhat
        else:
            save_x = torch.concat((save_x, x))
            save_y = torch.concat((save_y, y))
            save_pred = torch.concat((save_pred, yhat))
        
        break
    for i in range(10):
        matplotlib.rcParams.update({'font.size': 50})
        plt.close("all")
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        im = ax.imshow(save_x.detach().cpu().numpy()[i].reshape(100, 100), cmap="magma")
        plt.xticks([]),plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        plt.savefig("{}-x-{}-{}.png".format(pref, it, i), dpi=300, transparent=True)
        # exit(0)
        
        plt.close("all")
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        im = ax.imshow(save_y.detach().cpu().numpy()[i].reshape(100, 100), cmap="magma")
        plt.xticks([]),plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        plt.savefig("{}-y-{}-{}.png".format(pref, it, i), dpi=300, transparent=True)
        
        plt.close("all")
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        im = ax.imshow(save_pred.detach().cpu().numpy()[i].reshape(100, 100), cmap="magma")
        plt.xticks([]),plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        plt.savefig("{}-yh-{}-{}.png".format(pref, it, i), dpi=300, transparent=True)
        
        plt.close("all")
        plt.figure(figsize=(24, 20))
        ax = plt.gca()
        im = ax.imshow(((save_y-save_pred)**2).detach().cpu().numpy()[i].reshape(100, 100), cmap="magma", vmin=0, vmax=0.0003)
        plt.xticks([]),plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        plt.savefig("{}-err-{}-{}.png".format(pref, it, i), dpi=300, transparent=True)
    exit(0)
    
    

@torch.no_grad()    
def errInDS_2(neural_net, loader, device, transformation="linear",
                    error_fnc=nn.L1Loss(reduction='none'), tol1 = 0.2, tol2 = 0.1, tol3 = 0.05):
    error1 = 0.0
    
    error1_field = 0.0
    
    error1_src = 0.0
    error1_ring1 = 0.0
    error1_ring2 = 0.0
    error1_ring3 = 0.0
    
    errorMax = 0.0
    
    errorMax_field = 0.0
    
    errorMax_src = 0.0
    errorMax_ring1 = 0.0
    errorMax_ring2 = 0.0
    errorMax_ring3 = 0.0
    
    errorMaxm = 0.0
    
    errorMaxm_field = 0.0
    
    errorMaxm_src = 0.0
    errorMaxm_ring1 = 0.0
    errorMaxm_ring2 = 0.0
    errorMaxm_ring3 = 0.0
    
    errorMin = 0.0
    
    errorMin_field = 0.0
    
    errorMin_src = 0.0
    errorMin_ring1 = 0.0
    errorMin_ring2 = 0.0
    errorMin_ring3 = 0.0
    
    errorMinm = 0.0
    
    errorMinm_field = 0.0
    
    errorMinm_src = 0.0
    errorMinm_ring1 = 0.0
    errorMinm_ring2 = 0.0
    errorMinm_ring3 = 0.0
    
#     error1_per_im = []
    
#     error1_per_im_field = []
    
#     error1_per_im_src = []
    
    for i, data in enumerate(loader):
        x = data[0].to(device)
        y = data[1].to(device)
        
        srcs = x > 0
        rings1 = (y - y * torch.sign(x)) >= tol1
        rings2 = ( (y - y * torch.sign(x)) >= tol2 ) * ((y - y * torch.sign(x)) < tol1)
        rings3 = ( (y - y * torch.sign(x)) >= tol3 ) * ((y - y * torch.sign(x)) < tol2)
        
        nan_srcs = torch.where(srcs, float('nan'), 1.0)
        nan_rest = torch.where(srcs, 1.0, float('nan'))
        nan_notring1 = torch.where(rings1, 1.0, float('nan'))
        nan_notring2 = torch.where(rings2, 1.0, float('nan'))
        nan_notring3 = torch.where(rings3, 1.0, float('nan'))
        
        
        yhat = neural_net(x)
        
        yhat, y = transformation_inverse(yhat, y, transformation)
        
        e1 = error_fnc(yhat,y)
        
        e1_srcs = nan_rest * e1
        e1_field = nan_srcs * e1
        
        e1_ring1 = nan_notring1 * e1
        e1_ring2 = nan_notring2 * e1
        e1_ring3 = nan_notring3 * e1
        
        
        error1 += np.mean(e1.cpu().numpy())
        error1_field += np.nanmean(e1_field.cpu().numpy())
        error1_src += np.nanmean(e1_srcs.cpu().numpy())
        error1_ring1 += np.nanmean(e1_ring1.cpu().numpy())
        error1_ring2 += np.nanmean(e1_ring2.cpu().numpy())
        error1_ring3 += np.nanmean(e1_ring3.cpu().numpy())
        
        errorMax = np.maximum(errorMax, np.nanmax(e1.cpu().numpy()))
        errorMax_field = np.maximum(errorMax_field, np.nanmax(e1_field.cpu().numpy()))
        errorMax_src = np.maximum(errorMax_src, np.nanmax(e1_srcs.cpu().numpy()))
        errorMax_ring1 = np.maximum(errorMax_ring1, np.nanmax(e1_ring1.cpu().numpy()))
        errorMax_ring2 = np.maximum(errorMax_ring2, np.nanmax(e1_ring2.cpu().numpy()))
        errorMax_ring3 = np.maximum(errorMax_ring3, np.nanmax(e1_ring3.cpu().numpy()))
        
        errorMaxm += np.nanmax(e1.cpu().numpy())
        errorMaxm_field += np.nanmax(e1_field.cpu().numpy())
        errorMaxm_src += np.nanmax(e1_srcs.cpu().numpy())
        errorMaxm_ring1 += np.nanmax(e1_ring1.cpu().numpy())
        errorMaxm_ring2 += np.nanmax(e1_ring2.cpu().numpy())
        errorMaxm_ring3 += np.nanmax(e1_ring3.cpu().numpy())
        
        errorMin = np.minimum(errorMin, np.nanmin(e1.cpu().numpy()))
        errorMin_field = np.minimum(errorMin_field, np.nanmin(e1_field.cpu().numpy()))
        errorMin_src = np.minimum(errorMin_src, np.nanmin(e1_srcs.cpu().numpy()))
        errorMin_ring1 = np.minimum(errorMin_ring1, np.nanmin(e1_ring1.cpu().numpy()))
        errorMin_ring2 = np.minimum(errorMin_ring2, np.nanmin(e1_ring2.cpu().numpy()))
        errorMin_ring3 = np.minimum(errorMin_ring3, np.nanmin(e1_ring3.cpu().numpy()))
        
        errorMinm += np.nanmin(e1.cpu().numpy())
        errorMinm_field += np.nanmin(e1_field.cpu().numpy())
        errorMinm_src += np.nanmin(e1_srcs.cpu().numpy())
        errorMinm_ring1 += np.nanmin(e1_ring1.cpu().numpy())
        errorMinm_ring2 += np.nanmin(e1_ring2.cpu().numpy())
        errorMinm_ring3 += np.nanmin(e1_ring3.cpu().numpy())
                      
    return error1/(i+1), error1_field/(i+1),  error1_src/(i+1),  \
            errorMax, errorMax_field, errorMax_src, \
            errorMaxm/(i+1), errorMaxm_field/(i+1), errorMaxm_src/(i+1), \
            errorMin, errorMin_field, errorMin_src, \
            errorMinm/(i+1), errorMinm_field/(i+1), errorMinm_src/(i+1), \
            error1_ring1/(i+1), error1_ring2/(i+1),  error1_ring3/(i+1),  \
            errorMax_ring1, errorMax_ring2, errorMax_ring3, \
            errorMaxm_ring1/(i+1), errorMaxm_ring2/(i+1), errorMaxm_ring3/(i+1), \
            errorMin_ring1, errorMin_ring2, errorMin_ring3, \
            errorMinm_ring1/(i+1), errorMinm_ring2/(i+1), errorMinm_ring3/(i+1)


class tools(object):
    def errorPerDataset(self, PATH, theModel, device, BATCH_SIZE=50, NUM_WORKERS=0, std_tr=0.0, s=512):
        self.datasetNameList = [f'{i}SourcesRdm' for i in range(1,21)]
        self.errPerDS = [0 for i in range(1,21)]
        for (j, ds) in enumerate(self.datasetNameList):
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS, std_tr=std_tr, s=s).getDataLoaders()
            with torch.no_grad():
                erSum = 0
                for (i, data) in enumerate(testloader):
                    x = data[0].to(device)
                    y = data[1].to(device)
                    erSum += torch.mean(torch.abs(theModel(x) - y)).item()

                self.errPerDS[j] = erSum / len(testloader)
        return self.errPerDS
    
    def plotErrorPerDataset(self, slope=0.00015, ymax=0.004, type=0):
        if type == 0:
            plt.plot(self.datasetNameList, self.errPerDS)
            plt.xticks(rotation='45')
            plt.plot(slope * np.array(range(20)))
            plt.ylim(0,ymax)
            plt.title('Mean Abs Error per pixel')
            plt.xlabel('Dataset')
            plt.ylabel('Error')
            plt.show()
        elif type == 1:
            plt.plot(self.datasetNameList, self.errPerDS / np.array(range(1,20)))
            plt.xticks(rotation='45')
            plt.ylim(0,ymax)
            plt.title('Mean Abs Error per pixel / Num of sources')
            plt.xlabel('Dataset')
            plt.ylabel('Error')
            plt.show()

    def getSnapshots(self, theModel):
        r = next(iter(testloader))
        plt.imshow(vutils.make_grid(
            r[0].to(device)[1:2], padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],
                   cmap='cividis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.imshow(vutils.make_grid(
            r[1].to(device)[1:2], padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],
                   cmap='cividis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        with torch.no_grad():
            plt.imshow(vutils.make_grid(
                theModel(r[0].to(device)).cpu()[1:2],
                        padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],
                cmap='cividis', interpolation='nearest')
            plt.colorbar()
            plt.show()
        with torch.no_grad():
            plt.imshow(vutils.make_grid((theModel(r[0].to(device)).cpu()[1:2]) - r[1][1:2], padding=2, normalize=False,
                                        range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],cmap='cividis',
                       interpolation='nearest')
            plt.colorbar()
            plt.show()
            
    def errorHist(self, theModel):
        with torch.no_grad():
        #     erSum = 0
            for (i, data) in enumerate(testloader):
                if i > 0:
                    break
                x = data[0].to(device)
                y = data[1].to(device)
                erSum = torch.abs(theModel(x) - y)
                print(i)
                
        with torch.no_grad():
        #     plt.hist(erSum.view(50*512*512).cpu())
            self.perPixel = erSum.view(50*512*512).cpu().numpy()
            
        with torch.no_grad():
        #     plt.hist(erSum.view(50*512*512).cpu())
            self.perImage = erSum.mean(dim=(2,3)).view(50).cpu().numpy()


class accuracy(object):
    def __init__(self, nLabels=10):
        self.nLabels = nLabels

    def perLabel(self, x,y):
        totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
        correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
        for i in range(1, self.nLabels):
            totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
            correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
        l = torch.Tensor(totalPerLabel)
        correct = torch.Tensor(correctPerLabel)
        r = torch.zeros(self.nLabels)
        for idx in range(len(l)):
            if l[idx] != 0:
                # print(idx, l[idx], correct[idx])
                r[idx] = correct[idx]/l[idx]
            else:
                r[idx] = 0.0
    #     print(r)
        return r, l

    def overAllLabel(self, x,y):
        totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
        correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
        for i in range(1, self.nLabels):
            totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
            correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
        l = torch.Tensor(totalPerLabel)
        correct = torch.Tensor(correctPerLabel)
    #     print(r)
        return sum(correct)/sum(l), l

    def validationPerLabel(self, dataL, model):
        acc = torch.zeros(self.nLabels)
        ds = torch.zeros(self.nLabels)
        for i, data in enumerate(dataL, 0):
    #         r1.zero_grad()
            # Format batch
            x = data[0].to(device)
            y = data[1].to(device)
            output = model(x).max(1)[1]
            r, l = self.perLabel(output,y)
            acc = acc + r
            ds = ds + l
    #     print(i)
        return acc/(i+1)

    def validation(self, dataL, model):
        acc = 0
        ds = 0
        for i, data in enumerate(dataL, 0):
    #         r1.zero_grad()
            # Format batch
            x = data[0].to(device)
            y = data[1].to(device)
            output = model(x).max(1)[1]
            r, l = self.overAllLabel(output,y)
            acc = acc + r
            ds = ds + l
    #     print(i)
        return acc/(i+1)
