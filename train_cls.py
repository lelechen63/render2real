import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
import cv2
from options.step1_train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    print(y_pred)
    print('===========')
    print(y_actual)
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    print('---------------')
    print(pred)
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

opt = TrainOptions().parse()
opt.name = 'cls'
opt.clsname = 'idcls'
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
opt.model = 'cls'
opt.lr = 0.001
opt.datasetname = 'fs_tex'
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

optimizer = model.module.optimizer

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        if opt.clsname == 'idcls':
            gt_lbs = data['id']
        else:
            gt_lbs = data['exp']

        loss, out_labels, gt_labels = model( data['tex'], gt_lbs, infer=save_fake)

        # sum per device losses
        # loss =  torch.mean(loss) 
        # calculate final loss scalar
        ############### Backward Pass ####################
        # update generator weights
        optimizer.zero_grad()    
        loss.backward()          
        optimizer.step()

        ############## Display results and errors ##########
        ### print out errors
        # if total_steps % opt.print_freq == print_delta:

        # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
        # t = (time.time() - iter_start_time) / opt.print_freq
        # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        # visualizer.plot_current_errors(errors, total_steps)
        print( 'loss: ' loss.data, 'step: ', total_steps)
        prec1, temp_var = accuracy(out_labels.data, gt_labels.data , topk=(1, 1))
        print (prec1)
        ### display output images

        ## save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ## save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ## linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
