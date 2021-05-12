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

opt = TrainOptions().parse()
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
if opt.fp16:    
    from apex import amp
    model, [optimizer_G] = amp.initialize(model, [model.optimizer_G], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G = model.module.optimizer_G

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
    
        losses, generated, errormap = model(  image = Variable(data['image']) , 
                                    map_image =  Variable(data['pair_image']),
                                    map_type = data['pair_type'],
                                    viewpoint = Variable(data['viewpoint']),
                                    infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_pix = (loss_dict['A_pix'] + loss_dict.get('B_pix',0) + loss_dict.get('mis_pix',0) ) 
        loss_vgg = loss_dict.get('A_vgg',0) + loss_dict.get('B_vgg',0) + loss_dict.get('mis_vgg',0)

        loss_G = loss_pix + loss_vgg
        # loss_G = loss_dict['A_pix']
        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()    
        loss_G.backward()          
        optimizer_G.step()

        ############## Display results and errors ##########
        ### print out errors
        # if total_steps % opt.print_freq == print_delta:

        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
        t = (time.time() - iter_start_time) / opt.print_freq
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        # save_fake = True
        if save_fake:
            A_img = util.tensor2im(data['image'][0])
            print (A_img.shape)
            print(type(A_img))
            A_img = util.writeText(A_img, data['A_path'][0])
            # A_img = cv2.putText(A_img, data['A_path'][0], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
            #        1, (255, 0, 0), 2, cv2.LINE_AA)
            
            B_img = util.tensor2im(data['pair_image'][0])
            B_img = util.writeText(B_img, data['B_path'][0])
            # B_img = cv2.putText(B_img, data['B_path'][0], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
            #        1, (255, 0, 0), 2, cv2.LINE_AA)

            visuals = OrderedDict([
                                    ('image', A_img),
                                    ('pair_image', B_img),
                                   ('Aexp_Aid_image', util.tensor2im(generated[0].data[0])),
                                   ('Bexp_Bid_image', util.tensor2im(generated[1].data[0])),
                                   ('Aexp_Bid_image', util.tensor2im(generated[2].data[0])),
                                   ('Aid_Bexp_image', util.tensor2im(generated[3].data[0])),
                                   ('errormap', util.tensor2im(errormap.data[0]))
                                 ])
            visualizer.display_current_results(visuals, epoch, total_steps)

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
