import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

class DisentNet(BaseModel):
    def name(self):
        return 'DisentNet'

    def init_loss_filter(self, use_feat_loss, use_mismatch_loss):
        flags = (True, use_mismatch_loss, use_mismatch_loss,use_feat_loss, use_mismatch_loss and use_feat_loss, use_mismatch_loss and use_feat_loss)
        def loss_filter(A_pix_loss, B_pix_loss, pix_loss, A_feat_loss, B_feat_loss, mismatch_loss):
            return [l for (l,f) in zip((A_pix_loss,B_pix_loss, pix_loss, A_feat_loss,B_feat_loss,mismatch_loss),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 3
        output_nc =3
        linearity = not opt.no_linearity
        self.netEncoderDecoder = networks.define_Dis_EncoderDecoder(linearity, input_nc, opt.code_n,opt.encoder_fc_n, 
                                                opt.ngf, opt.netG, opt.n_downsample_global, 
                                                opt.n_blocks_global, opt.norm, gpu_ids=self.gpu_ids)  

     
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netEncoderDecoder, 'DisED', opt.which_epoch, pretrained_path)    
        
            # if self.isTrain:
            #     self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_vgg_loss, not opt.no_mismatch_loss)
                    
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()

            self.criterionPix = torch.nn.MSELoss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('A_pix','B_pix','mis_pix','A_vgg', 'B_vgg', "mis_vgg")

            # initialize optimizers
            # optimizer G
            params = list(self.netEncoderDecoder.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            


    def forward(self, image, map_image, map_type, viewpoint, infer=False):
        A_viewpoint = viewpoint[:,0]
        B_viewpoint = viewpoint[:,1]

        image = Variable(image.cuda())
        map_image = Variable(map_image.cuda())
        A_viewpoint = Variable(A_viewpoint.cuda())
        B_viewpoint = Variable(B_viewpoint.cuda())

        # Fake Generation
        A_exp_code, A_id_code,Aexp_Aid_image, B_exp_code, B_id_code, Bexp_Bid_image, Aexp_Bid_image, Bexp_Aid_image   = self.netEncoderDecoder(image, A_viewpoint, map_image, B_viewpoint, map_type)

        # mismatch reconstruction
        # Aexp_Bid_image
        # if map_type == 0: toss 0-> same iden, diff exp
        # replace B's exp_code with A's exp_code, feed(B's id_code, A's exp_code) to decoder, it will output A''s image.
        # same exp as A, same id as A/B, compute loss with A'
        
        #if map_type == 1: toss 1-> same exp, diff iden
        # replace B's exp_code with A's exp_code, feed(B's id_code, A's exp_code) to decoder, it will output B''s image.
        # same exp as A/B, same id as B, compute loss with B'

        # Bexp_Aid_image
        # if map_type == 0: toss 0-> same iden, diff exp
        # replace A's exp_code with B's exp_code, feed(A's id_code, B's exp_code) to decoder, it will output B''s image.
        # same exp as B, same id as A/B, compute loss with B'
        
        #if map_type == 1: toss 1-> same exp, diff iden
        # replace A's exp_code with B's exp_code, feed(A's id_code, B's exp_code) to decoder, it will output A''s image.
        # same exp as A/B, same id as A, compute loss with A'

        ############################################################################

        # VGG feature matching loss
        loss_G_VGG = 0
        loss_G_VGG3 = 0
        loss_G_VGG4 = 0
        loss_G_VGG1 = 0
        loss_G_VGG2 = 0
        if not self.opt.no_vgg_loss:
            # mismatch loss
            for i in range(map_type.shape[0]):
                if map_type[i] == 0:
                    loss_G_VGG1 += self.criterionVGG(Aexp_Bid_image[i].unsqueeze(0), image[i].unsqueeze(0)) * self.opt.lambda_feat
                    loss_G_VGG2 += self.criterionVGG(Bexp_Aid_image[i].unsqueeze(0), map_image[i].unsqueeze(0)) * self.opt.lambda_feat
                else:
                    loss_G_VGG1 += self.criterionVGG(Aexp_Bid_image[i].unsqueeze(0), map_image[i].unsqueeze(0)) * self.opt.lambda_feat
                    loss_G_VGG2 += self.criterionVGG(Bexp_Aid_image[i].unsqueeze(0), image[i].unsqueeze(0)) * self.opt.lambda_feat
            
            # reconstruction loss
            loss_G_VGG3 = self.criterionVGG(Aexp_Aid_image, image) * self.opt.lambda_feat
            loss_G_VGG4 = self.criterionVGG(Bexp_Bid_image, map_image) * self.opt.lambda_feat
            loss_G_VGG = loss_G_VGG1 + loss_G_VGG2 
        
        loss_G_pix = 0
        loss_G_pix1 = 0
        loss_G_pix2 = 0
        # mismatch loss
        for i in range(map_type.shape[0]):
            if map_type[i] == 0:
                loss_G_pix1 += self.criterionPix(Aexp_Bid_image[i].unsqueeze(0), image[i].unsqueeze(0)) * self.opt.lambda_pix
                loss_G_pix2 += self.criterionPix(Bexp_Aid_image[i].unsqueeze(0), map_image[i].unsqueeze(0)) * self.opt.lambda_pix
            else:
                loss_G_pix1 += self.criterionPix(Aexp_Bid_image[i].unsqueeze(0), map_image[i].unsqueeze(0)) * self.opt.lambda_pix
                loss_G_pix2 += self.criterionPix(Bexp_Aid_image[i].unsqueeze(0), image[i].unsqueeze(0)) * self.opt.lambda_pix
        
        # reconstruction loss
        # loss_G_pix3 = self.criterionPix(Aexp_Aid_image, image) * self.opt.lambda_pix
        loss_G_pix4 = self.criterionPix(Bexp_Bid_image, map_image) * self.opt.lambda_pix
        loss_G_pix = loss_G_pix1 + loss_G_pix2 

        loss_G_pix3 = ( ( Aexp_Aid_image - image ) ** 2 ).mean()

        A_err_map = (Aexp_Aid_image - image).sum(1).unsqueeze(1)

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_pix3, loss_G_pix4, loss_G_pix, loss_G_VGG3, loss_G_VGG4, loss_G_VGG), [Aexp_Aid_image, Bexp_Bid_image, Aexp_Bid_image, Bexp_Aid_image], A_err_map ]
                                    # A iamge l1, B image l1, mismatch l1, A vgg loss, B vgg loss, mismatch vgg 

    def inference(self, image, viewpoint):

        image = Variable(image.data.cuda())
        viewpoint = Variable(viewpoint.data.cuda())
        # Fake Generation
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                id_code, exp_code = self.netEncoderDecoder(image)
                fake_image = self.netDecoder(exp_code, id_code, viewpoint[:,0])
        else:

            d_code, exp_code = self.netEncoder(image)
            fake_image = self.netDecoder(exp_code, id_code, viewpoint[:,0])
        return fake_image


    def save(self, which_epoch):
        self.save_network(self.netEncoderDecoder, 'DisED', which_epoch, self.gpu_ids)
        

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceDisentNet(DisentNet):
    def forward(self, image, viewpoint):
        return self.inference(image, viewpoint)
