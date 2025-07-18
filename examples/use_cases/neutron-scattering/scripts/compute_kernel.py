import torch
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

def metric_average(val, size, name):
    # Sum everything and divide by total size:
    dist.all_reduce(val,op=dist.ReduceOp.SUM)
    val /= size
    return val

def train(epoch, rank, size,
          model, optimizer, 
          train_loader, 
          train_sampler,
          criterion_reg, criterion_class, 
          lr_scheduler,
          on_gpu, 
          log_interval, 
          loss_list):
    
    model.train()
    train_sampler.set_epoch(epoch)
    if rank == 0:
        print("lr = ", optimizer.param_groups[0]['lr'])

    if epoch % log_interval == 0:
        running_loss  = torch.tensor(0.0)
        running_loss1 = torch.tensor(0.0)
        running_loss2 = torch.tensor(0.0)
        if on_gpu:
            running_loss, running_loss1, running_loss2  = running_loss.cuda(), running_loss1.cuda(), running_loss2.cuda()

    for batch_idx, current_batch in enumerate(train_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]

        optimizer.zero_grad()
        class_output, regression_output = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if on_gpu:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        loss1 = criterion_reg(regression_output, regression_gndtruth)
        loss2 = criterion_class(class_output, class_gndtruth)
        loss  = loss1 + loss2
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
#        max_grad = max(param.grad.abs().max() for param in model.parameters() if param.grad is not None)
#        print("TW: Looking at param grad, rank = {}, batch_idx = {}, |grad|_max = {}".format(rank, batch_idx, max_grad))

        optimizer.step()

        if epoch % log_interval == 0:
            running_loss  += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        lr_scheduler.step()

    if epoch % log_interval == 0:
        running_loss  = running_loss  / len(train_loader)
        running_loss1 = running_loss1 / len(train_loader)
        running_loss2 = running_loss2 / len(train_loader)
        loss_avg  = metric_average(running_loss, size, 'running_loss')
        loss1_avg = metric_average(running_loss1, size, 'running_loss1')
        loss2_avg = metric_average(running_loss2, size, 'running_loss2')
        if rank == 0:
            print("epoch: {}, Average loss_reg: {:15.8f}, loss_class: {:15.8f}, loss_tot: {:15.8f}".format(epoch, loss1_avg, loss2_avg, loss_avg))
        loss_list.append(loss_avg)

def test(epoch, rank, size,
         model,
         test_loader,
         criterion_reg, criterion_class,
         on_gpu,
         log_interval,
         loss_list):

    model.eval()
    
    test_loss  = torch.tensor(0.0)
    test_loss1 = torch.tensor(0.0)
    test_loss2 = torch.tensor(0.0)
    if on_gpu:
        test_loss, test_loss1, test_loss2  = test_loss.cuda(), test_loss1.cuda(), test_loss2.cuda()
    
    for batch_idx, current_batch in enumerate(test_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]        

        with torch.no_grad():
            y_pred_torch_class, y_pred_torch_regression = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if on_gpu:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        test_loss1 += criterion_reg(y_pred_torch_regression, regression_gndtruth).item()
        test_loss2 += criterion_class(y_pred_torch_class, class_gndtruth).item()
    
    test_loss1 = test_loss1 / len(test_loader)
    test_loss2 = test_loss2 / len(test_loader)
    test_loss  = test_loss1 + test_loss2

    loss_avg  = metric_average(test_loss, size, "loss_avg")
    loss_avg1 = metric_average(test_loss1, size, "loss_avg1")
    loss_avg2 = metric_average(test_loss2, size, "loss_avg2")

    loss_avg = loss_avg.cpu().numpy()

    if epoch % log_interval == 0:
        if rank == 0:
            print("epoch: {}, Average test_loss_reg: {:15.8f}, test_loss_class: {:15.8f}, test_loss_tot: {:15.8f}".format(epoch, loss_avg1, loss_avg2, loss_avg))
        loss_list.append(loss_avg)

    return loss_avg

#Here criterion_reg could be different from that in train since it does not have uncertainty term
def validation(rank, size,
               model,
               test_loader,
               criterion_reg, criterion_class,
               on_gpu):

    model.eval()

    diff0  = torch.tensor(0.0)
    sigma2 = torch.tensor(0.0)
    class_loss = torch.tensor(0.0)
    if on_gpu:
        diff0, sigma2, class_loss  = diff0.cuda(), sigma2.cuda(), class_loss.cuda()
 
    for batch_idx, current_batch in enumerate(test_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]        

        with torch.no_grad():
            y_pred_torch_class, y_pred_torch_regression = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if on_gpu:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        diff0 += criterion_reg(y_pred_torch_regression[:,0:3], regression_gndtruth).item()
        class_loss += criterion_class(y_pred_torch_class, class_gndtruth).item()
        sigma2 += torch.mean(torch.exp(y_pred_torch_regression[:,3]))
    
    diff0 = diff0 / len(test_loader)
    class_loss = class_loss / len(test_loader)
    sigma2 = sigma2 / len(test_loader)

    diff0 = metric_average(diff0, size, "diff0").cpu().numpy()
    class_loss = metric_average(class_loss, size, "class_loss").cpu().numpy()
    sigma2 = metric_average(sigma2, size, "sigma2").cpu().numpy()

    if rank == 0:
        print("Avg diff on test set = ", diff0)
        print("Avg sigma^2 on test set = ", class_loss)
        print("Avg class loss on test set = ", sigma2)
    
    return diff0, sigma2, class_loss
