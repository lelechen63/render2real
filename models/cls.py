import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

class ClsNet(BaseModel):
    def name(self):
        return 'ClsNet'

   
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.clsname = opt.clsname
        if self.clsname == 'idcls':
            output_nc =300
        else:
            output_nc = 20

        self.classifier = networks.define_Classifier(opt.loadSize, output_nc, 
                                                opt.ngf, opt.netG, opt.n_downsample_global, 
                                                opt.n_blocks_global, opt.norm, gpu_ids=self.gpu_ids)  

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.classifier, self.clsname , opt.which_epoch, pretrained_path)    
        
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # define loss functions
                    
            self.criterionl1 = torch.nn.L1Loss()
            self.criterionl2 = torch.nn.MSELoss()
            self.criterionCEL = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            # optimizer G
            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=opt.lr)  
                           


    def forward(self, texture, gt_labels, infer=False):
        out_labels = self.classifier(texture)
        loss = self.criterionCEL( out_labels, gt_labels)
        
        return [ loss , out_labels, gt_labels ]

    def save(self, which_epoch):
        self.save_network(self.classifier, self.clsname , which_epoch, self.gpu_ids)
        
 