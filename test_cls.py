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
from PIL import Image
from models import networks
def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
   
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

opt = TrainOptions().parse()
opt.name = 'cls'
# opt.clsname = 'idcls'
opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
opt.model = 'cls'
opt.isTrain = False
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
print('#testing images = %d' % dataset_size)
if opt.clsname == 'idcls':
    cls = networks.TexClassifier(opt.loadSize, 301, 64, opt.n_downsample_global, opt.n_blocks_global)
    cls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/100_net_idcls.pth'))
else:
    cls = networks.TexClassifier(opt.loadSize, 20, 64, opt.n_downsample_global, opt.n_blocks_global)
    cls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/100_net_expcls.pth'))
cls = cls.cuda()
criterion = torch.nn.CrossEntropyLoss()

print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for i, data in enumerate(dataset):
    if total_steps % opt.print_freq == print_delta:
        iter_start_time = time.time()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize
    # whether to collect output images

    ############## Forward Pass ######################
    if opt.clsname == 'idcls':
        gt_lbs = data['id']
    else:
        gt_lbs = data['exp']
    
    out_labels = cls( data['tex'].cuda())
    loss = criterion(out_labels, gt_lbs)
    loss = loss.mean()
    # calculate final loss scalar
    ############### Backward Pass ####################
    # update generator weights
    try:
        print( 'loss: ', loss.data.sum())
    except:
        print('+++++')
        print(loss)
    print( 'step: ', total_steps)

    ############## Display results and errors #########
    ## save latest model
    
    prec1, temp_var = accuracy(out_labels.data, gt_lbs.data , topk=(1, 1))
    try:
        print( 'acc: ', prec1)
    except:
        print('******')
        print(out_labels)


