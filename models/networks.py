import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

def define_Dis_EncoderDecoder(linearity, input_nc, code_n,encoder_fc_n, ngf, netG, n_downsample_global=5, n_blocks_global=9, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'disent':    
        encoderdecoder = DisentEncoderDecoder(linearity, input_nc, code_n,encoder_fc_n, ngf, n_downsample_global, n_blocks_global)       
    
    else:
        raise('generator not implemented!')
    print(encoderdecoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        encoderdecoder.cuda(gpu_ids[0])
    encoderdecoder.apply(weights_init)
    return encoderdecoder

def define_TexMesh_EncoderDecoder(tex_shape, linearity, input_nc, code_n,encoder_fc_n, ngf, netG, n_downsample_global=5, n_blocks_global=9, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'disent':    
        encoderdecoder = TexMeshEncoderDecoder(tex_shape, linearity, input_nc, code_n,encoder_fc_n, ngf, n_downsample_global, n_blocks_global)       
    
    else:
        raise('generator not implemented!')
    print(encoderdecoder)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        encoderdecoder.cuda(gpu_ids[0])
    encoderdecoder.apply(weights_init)
    return encoderdecoder

def define_Classifier( tex_size, output_nc, ngf, netG, n_downsample_global=5, n_blocks_global=9, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
   
    classifier = TexClassifier(tex_size, output_nc, ngf, n_downsample_global, n_blocks_global)       
    
    print(classifier)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        classifier.cuda(gpu_ids[0])
    classifier.apply(weights_init)
    return classifier


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_G_fewshot(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator_fewhsot(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class CLSLoss(nn.Module):
    def __init__(self, opt):
        super(CLSLoss, self).__init__()        
        self.idcls = TexClassifier(opt.loadSize, 301, 64, opt.n_downsample_global, opt.n_blocks_global)
        self.expcls = TexClassifier(opt.loadSize, 20, 64, opt.n_downsample_global, opt.n_blocks_global)
        
        self.idcls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/100_net_idcls.pth'))
        self.expcls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/100_net_expcls.pth'))
        self.expcls = self.expcls.cuda()
        self.idcls = self.idcls.cuda()
        for param in self.idcls.parameters():
            param.requires_grad = False
        for param in self.expcls.parameters():
            param.requires_grad = False


        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tex, gt_lab, mode):
        device_id = tex.device.index
        print('tex', device_id)

        device_id = self.expcls.device.index
        print('expcls', device_id)
        device_id = self.idcls.device.index
        print('idcls', device_id)
        print('++++++++++++++++++++')

        if mode == 'id':
            
            out_lab = self.idcls(tex)
        else:
            out_lab = self.expcls(tex)
            
        loss = self.criterion(out_lab, gt_lab.detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample 16 times
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.encoder = nn.Sequential(*model)
        model = []
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.resblocks = nn.Sequential(*model)
        ### upsample
        model = []         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        self.decoder = nn.Sequential(*model)

        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)
            
    def forward(self, input):
        # print (input.shape, 'input')
        encoded = self.encoder(input)
        # print (encoded.shape, "encoded")
        encoded = self.resblocks(encoded)
        # print (encoded.shape, "encoded")
        decoded = self.decoder(encoded)
        # print (decoded.shape, "decoded")
        output = self.output_layer(decoded)
        # print (output.shape, "output")
        return output        
   

class DisentEncoderDecoder(nn.Module):
    def __init__(self, linearity, input_nc,  code_n, encoder_fc_n, ngf=64, n_downsampling=5, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DisentEncoderDecoder, self).__init__()        
        activation = nn.ReLU(True)        

        self.CNNencoder = nn.Sequential(
                            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                            norm_layer(ngf), 
                            nn.ReLU(True),  

                            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  # 512

                            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  #256

                            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 128

                            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 64

                            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #32

                            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #16

                            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 16),
                            nn.ReLU(True),  #8

                        )
        
        self.identity_enc = nn.Sequential(
                                    nn.Linear( ngf * 16 * 4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )

        self.expression_enc = nn.Sequential(
                                    nn.Linear( ngf * 16 * 4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )
        self.identity_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.exp_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.viewencoder = nn.Sequential(
                                    nn.Linear( 12, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*4),
                                    nn.ReLU(True)
                                    )
        self.code_dec = nn.Sequential(
                                    nn.Linear( ngf*4 * 3, ngf*16),
                                    nn.ReLU(True)
                                    )

        ### upsample

        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf), 
                        nn.ReLU(True),
                    )


        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.resblocks = nn.Sequential(*model)

    
    def forward(self, A_img, A_view, B_img , B_view, map_type ):
        return_list = []

        A_encoded = self.CNNencoder(A_img)
        A_encoded = self.resblocks(A_encoded).view(A_encoded.shape[0], -1)
        A_identity_code = self.identity_enc(A_encoded)
        A_expression_code = self.expression_enc(A_encoded)
        return_list.append(A_expression_code)
        return_list.append( A_identity_code)

        A_view_fea = self.viewencoder(A_view)
        A_exp_fea = self.exp_dec(A_expression_code)
        A_id_fea = self.identity_dec(A_identity_code)
        A_feature = torch.cat([A_exp_fea, A_id_fea, A_view_fea], axis = 1)
        A_code = self.code_dec(A_feature)
        A_code = A_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        A_decoded = self.decoder(A_code)
        recons_A = self.output_layer(A_decoded)
        return_list.append( recons_A)

        B_encoded = self.CNNencoder(B_img)
        B_encoded = self.resblocks(B_encoded).view(B_encoded.shape[0], -1)
        B_identity_code = self.identity_enc(B_encoded)
        B_expression_code = self.expression_enc(B_encoded)

        return_list.append( B_expression_code)
        return_list.append( B_identity_code)

        B_view_fea = self.viewencoder(B_view)
        B_exp_fea = self.exp_dec(B_expression_code)
        B_id_fea = self.identity_dec(B_identity_code)

        B_feature = torch.cat([B_exp_fea, B_id_fea, A_view_fea], axis = 1)
        B_code = self.code_dec(B_feature)
        B_code = B_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
       
        B_decoded = self.decoder(B_code)
        recons_B = self.output_layer(B_decoded)

        return_list.append( recons_B)

        Aexp_Bid_fea =[]
        Bexp_Aid_fea = []
        for i in range(map_type.shape[0]):
            if map_type[i] == 0:
                Aexp_Bid_fea.append( torch.cat([A_exp_fea[i], B_id_fea[i], A_view_fea[i]], axis = 0) )
                Bexp_Aid_fea.append( torch.cat([B_exp_fea[i], A_id_fea[i], B_view_fea[i]], axis = 0) )

            else:
                Aexp_Bid_fea.append( torch.cat([A_exp_fea[i], B_id_fea[i], B_view_fea[i]], axis = 0) )
                Bexp_Aid_fea.append( torch.cat([B_exp_fea[i], A_id_fea[i], A_view_fea[i]], axis = 0) )

        Aexp_Bid_fea = torch.stack(Aexp_Bid_fea, dim = 0)
        Bexp_Aid_fea = torch.stack(Bexp_Aid_fea, dim = 0)
        Aexp_Bid_code = self.code_dec(Aexp_Bid_fea)
        Aexp_Bid_code = Aexp_Bid_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        Aexp_Bid_decoded = self.decoder(Aexp_Bid_code)
        recons_Aexp_Bid = self.output_layer(Aexp_Bid_decoded)

        return_list.append( recons_Aexp_Bid)

        Bexp_Aid_code = self.code_dec(Bexp_Aid_fea)
        Bexp_Aid_code = Bexp_Aid_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        Bexp_Aid_decoded = self.decoder(Bexp_Aid_code)
        recons_Bexp_Aid = self.output_layer(Bexp_Aid_decoded)
    
        return_list.append( recons_Bexp_Aid)

        return return_list


class TexClassifier(nn.Module):
    def __init__(self, tex_size, output_nc, ngf=64, n_downsampling=5, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(TexClassifier, self).__init__()        
        activation = nn.ReLU(True)        

        self.CNNencoder = nn.Sequential(
                            nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                            norm_layer(ngf), 
                            nn.ReLU(True),  

                            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  # 512

                            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  #256

                            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 128

                            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 64

                            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #32

                            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #16

                            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 16),
                            nn.ReLU(True),  #8

                        )
        
        self.fc_layer = nn.Sequential(
                                    nn.Linear( ngf * 16 * 4, ngf*4),
                                    nn.ReLU(),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(),
                                    nn.Linear( ngf*4,output_nc)
                                    )

    def forward(self, tex ):
        fea = self.CNNencoder(tex)
        label = self.fc_layer(fea.view(fea.shape[0], -1))
        return label

class TexMeshEncoderDecoder(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc,  code_n, encoder_fc_n, ngf=64, n_downsampling=5, n_blocks=4, norm_layer=nn.BatchNorm2d,padding_type='reflect'):
        assert(n_blocks >= 0)
        super(TexMeshEncoderDecoder, self).__init__()        
        activation = nn.ReLU(True)     
        self.tex_shape = tex_shape
        self.CNNencoder = nn.Sequential(
                            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                            norm_layer(ngf), 
                            nn.ReLU(True),  

                            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  # 512

                            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  #256

                            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 128

                            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 64

                            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #32

                            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #16

                            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 16),
                            nn.ReLU(True),  #8
                            # nn.Conv2d(ngf*16 , ngf  * 16, kernel_size=3, stride=2, padding=1),
                            # norm_layer(ngf  * 16),
                            # nn.ReLU(True),  #4
                        )

        self.enc_input_size = int(ngf * 16 * self.tex_shape/256 * self.tex_shape/256 * 4 + ngf * 4)
        self.identity_enc = nn.Sequential(
                                    nn.Linear( self.enc_input_size, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )

        self.expression_enc = nn.Sequential(
                                    nn.Linear( self.enc_input_size, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )
        self.identity_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    # nn.Linear( ngf*4, ngf*4),
                                    # nn.ReLU(True),
                                    # nn.Linear( ngf*4, ngf*4),
                                    # nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.exp_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    # nn.Linear( ngf*4, ngf*4),
                                    # nn.ReLU(True),
                                    # nn.Linear( ngf*4, ngf*4),
                                    # nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.meshencoder = nn.Sequential(
                                    nn.Linear( 78951, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*4),
                                    nn.ReLU(True)
                                    )
        self.tex_fc_dec = nn.Sequential(
                                    nn.Linear( ngf*4 * 2, ngf*16),
                                    nn.ReLU(True)
                                    )
        self.mesh_fc_dec = nn.Sequential(
                                    nn.Linear( ngf*4 * 2, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, 78951),
                                    )
        ### upsample

        self.tex_decoder = nn.Sequential(
                        # nn.ConvTranspose2d(ngf * 16, ngf * 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                        # norm_layer(ngf * 16), 
                        # nn.ReLU(True),
                        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf), 
                        nn.ReLU(True),
                    )


        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.resblocks = nn.Sequential(*model)

    
    def forward(self, A_tex, A_mesh, B_tex , B_mesh, map_type ):
        return_list = []
        A_tex_encoded = self.CNNencoder(A_tex)
        A_tex_encoded = self.resblocks(A_tex_encoded).view(A_tex_encoded.shape[0], -1)
        A_mesh_encoded = self.meshencoder(A_mesh)
        A_encoded= torch.cat([A_mesh_encoded, A_tex_encoded], 1)
        A_identity_code = self.identity_enc(A_encoded)
        A_expression_code = self.expression_enc(A_encoded)
        return_list.append(A_expression_code)
        return_list.append( A_identity_code)

        A_exp_fea = self.exp_dec(A_expression_code)
        A_id_fea = self.identity_dec(A_identity_code)
        A_feature = torch.cat([A_exp_fea, A_id_fea], axis = 1)
        A_rec_mesh = self.mesh_fc_dec(A_feature)
        return_list.append( A_rec_mesh)

        A_tex_dec = self.tex_fc_dec(A_feature)
        if self.tex_shape == 256:
            A_tex_dec = A_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        elif self.tex_shape == 512:
            A_tex_dec = A_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4) # not sure 
        else:
            A_tex_dec = A_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8,8) # not sure 

        A_decoded = self.tex_decoder(A_tex_dec)
        A_rec_tex = self.output_layer(A_decoded)
        return_list.append( A_rec_tex)        

        B_tex_encoded = self.CNNencoder(B_tex)
        B_tex_encoded = self.resblocks(B_tex_encoded).view(B_tex_encoded.shape[0], -1)
        B_mesh_encoded = self.meshencoder(B_mesh)
        B_encoded= torch.cat([B_mesh_encoded, B_tex_encoded], 1)
        
        B_identity_code = self.identity_enc(B_encoded)
        B_expression_code = self.expression_enc(B_encoded)
        return_list.append(B_expression_code)
        return_list.append( B_identity_code)

        B_exp_fea = self.exp_dec(B_expression_code)
        B_id_fea = self.identity_dec(B_identity_code)
        B_feature = torch.cat([B_exp_fea, B_id_fea], axis = 1)
        B_rec_mesh = self.mesh_fc_dec(B_feature)
        return_list.append( B_rec_mesh)

        B_tex_dec = self.tex_fc_dec(B_feature)
        if self.tex_shape == 256:
            B_tex_dec = B_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        elif self.tex_shape == 512:
            B_tex_dec = B_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4) # not sure 
        else:
            B_tex_dec = B_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8,8) # not sure  
        B_decoded = self.tex_decoder(B_tex_dec)
        B_rec_tex = self.output_layer(B_decoded)
        return_list.append( B_rec_tex)

        Aexp_Bid_fea =[]
        Bexp_Aid_fea = []
        for i in range(map_type.shape[0]):
            if map_type[i] == 0:
                Aexp_Bid_fea.append( torch.cat([A_exp_fea[i], B_id_fea[i]], axis = 0) )
                Bexp_Aid_fea.append( torch.cat([B_exp_fea[i], A_id_fea[i],], axis = 0) )
            else:
                Aexp_Bid_fea.append( torch.cat([A_exp_fea[i], B_id_fea[i]], axis = 0) )
                Bexp_Aid_fea.append( torch.cat([B_exp_fea[i], A_id_fea[i]], axis = 0) )

        Aexp_Bid_fea = torch.stack(Aexp_Bid_fea, dim = 0)
        Bexp_Aid_fea = torch.stack(Bexp_Aid_fea, dim = 0)

        Aexp_Bid_mesh = self.mesh_fc_dec(Aexp_Bid_fea)
        return_list.append( Aexp_Bid_mesh)

        Aexp_Bid_tex_dec = self.tex_fc_dec(Aexp_Bid_fea)

        if self.tex_shape == 256:
            Aexp_Bid_tex_dec = Aexp_Bid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        elif self.tex_shape == 512:
            Aexp_Bid_tex_dec = Aexp_Bid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4) # not sure 
        else:
            Aexp_Bid_tex_dec = Aexp_Bid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8,8) # not sure  
        Aexp_Bid_decoded = self.tex_decoder(Aexp_Bid_tex_dec)
        Aexp_Bid_rec_tex = self.output_layer(Aexp_Bid_decoded)
        return_list.append( Aexp_Bid_rec_tex)

        Bexp_Aid_mesh = self.mesh_fc_dec(Bexp_Aid_fea)
        return_list.append( Bexp_Aid_mesh)

        Bexp_Aid_tex_dec = self.tex_fc_dec(Bexp_Aid_fea)
        if self.tex_shape == 256:
            Bexp_Aid_tex_dec = Bexp_Aid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 
        elif self.tex_shape == 512:
            Bexp_Aid_tex_dec = Bexp_Aid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4) # not sure 
        else:
            Bexp_Aid_tex_dec = Bexp_Aid_tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8,8)
        Bexp_Aid_decoded = self.tex_decoder(Bexp_Aid_tex_dec)
        Bexp_Aid_rec_tex = self.output_layer(Bexp_Aid_decoded)
        return_list.append( Bexp_Aid_rec_tex)


        return return_list


class DisentEncoderDecoder2(nn.Module):
    def __init__(self, linearity, input_nc,  code_n, encoder_fc_n, ngf=64, n_downsampling=5, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DisentEncoderDecoder2, self).__init__()        
        activation = nn.ReLU(True)        

        ### downsample 16 times
        
        self.CNNencoder = nn.Sequential(
                            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                            norm_layer(ngf), 
                            nn.ReLU(True),  

                            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  # 512

                            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 2),
                            nn.ReLU(True),  #256

                            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 128

                            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 4),
                            nn.ReLU(True), # 64

                            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #32

                            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 8),
                            nn.ReLU(True),  #16

                            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf  * 16),
                            nn.ReLU(True),  #8

                        )
        
        self.identity_enc = nn.Sequential(
                                    nn.Linear( ngf * 16 * 4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )

        self.expression_enc = nn.Sequential(
                                    nn.Linear( ngf * 16 * 4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,code_n),
                                    nn.ReLU(True),
                                    )
        self.identity_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.exp_dec = nn.Sequential(
                                    nn.Linear( code_n, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*4,ngf*4),
                                    nn.ReLU(True),
                                    )
        self.viewencoder = nn.Sequential(
                                    nn.Linear( 12, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*2),
                                    nn.ReLU(True),
                                    nn.Linear( ngf*2, ngf*4),
                                    nn.ReLU(True)
                                    )
        self.code_dec = nn.Sequential(
                                    nn.Linear( ngf*4 * 3, ngf*16),
                                    nn.ReLU(True)
                                    )

        ### upsample

        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 8), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 4), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf * 2), 
                        nn.ReLU(True),

                        nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(ngf), 
                        nn.ReLU(True),
                    )

        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.resblocks = nn.Sequential(*model)

    
    def forward(self, A_img, A_view, B_img , B_view, map_type ):
        return_list = []

        A_encoded = self.CNNencoder(A_img)

        # A_encoded = self.pool(A_encoded)

        A_encoded = self.resblocks(A_encoded)
        # A_encoded = self.pool(A_encoded)
        # print(A_encoded.shape)
        A_encoded = A_encoded.view(A_encoded.shape[0], -1)

        A_identity_code = self.identity_enc(A_encoded)
        A_expression_code = self.expression_enc(A_encoded)

        A_view_fea = self.viewencoder(A_view)
        A_exp_fea = self.exp_dec(A_expression_code)
        A_id_fea = self.identity_dec(A_identity_code)
        A_feature = torch.cat([A_exp_fea, A_id_fea, A_view_fea], axis = 1)
        A_code = self.code_dec(A_feature)
        A_code = A_code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2,2) # not sure 

        A_decoded = self.decoder(A_code)
        recons_A = self.output_layer(A_decoded)

        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
        return_list.append( recons_A)
     

        return return_list     

class DisentDecoder(nn.Module):
    def __init__(self, linearity, output_nc,  code_n, encoder_fc_n, ngf=64, n_downsampling=5, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DisentDecoder, self).__init__()        
        activation = nn.ReLU(True)        

        #################
        # manipulate module
        if linearity:
            pass  
        ##################
        mult = 2**n_downsampling

        model = []
        model.append(LinearBlock(code_n, ngf*4, norm = 'none' , activation = 'relu'))

        for i in range(int(encoder_fc_n/2)):
            model.append(LinearBlock(ngf*4, ngf*4, norm = 'none' , activation = 'relu'))
        self.identity_dec = nn.Sequential(*model)

        model = []
        model.append(LinearBlock(code_n, ngf*4, norm = 'none' , activation = 'relu'))
        for i in range(int(encoder_fc_n/2)):
            model.append(LinearBlock(ngf*4, ngf*4, norm = 'none' , activation = 'relu'))
        
        self.exp_dec = nn.Sequential(*model)

        model = []
        model.append(LinearBlock(ngf*4 * 2 , ngf * mult , norm = 'none' , activation = 'relu'))
        self.code_dec = nn.Sequential(*model)

        model = []
        model.append(LinearBlock(3, ngf*2, norm = 'none' , activation = 'relu'))
        for i in range(2):
            model.append(LinearBlock(ngf*2, ngf*2, norm = 'none' , activation = 'relu'))
        model.append(LinearBlock(ngf*2, ngf  * 4, norm = 'none' , activation = 'relu'))
        self.viewencoder = nn.Sequential(*model)

        ### resnet blocks
        # model = []
        # for i in range(n_blocks):
        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # self.resblocks = nn.Sequential(*model)

        ### upsample
        model = []         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        self.decoder = nn.Sequential(*model)

        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)
            
    def forward(self, exp_code, id_code, viewpoint):
        # print (input.shape, 'input')
        view_fea = self.viewencoder(viewpoint)
        exp_fea = self.exp_dec(exp_code)
        # print (exp_fea.shape, "exp_fea")

        id_fea = self.identity_dec(id_code)
        # print (id_fea.shape, "id_fea")

        feature = torch.cat([exp_fea, id_fea], axis = 1)
        # print (feature.shape, "feature")
        code = self.code_dec(feature)
        # print (code.shape, "code")

        code = code.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32,32) # not sure 
        # print (code.shape, "code")
        # code = self.resblocks(code)
        # print (id_fea.shape, "id_fea")
        decoded = self.decoder(code)
        # print (decoded.shape, "decoded")
        output = self.output_layer(decoded)
        # print (output.shape, "output")
        return output           


class GlobalGenerator_fewhsot(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator_fewhsot, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample 16 times
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.encoder = nn.Sequential(*model)
        model = []
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.resblocks = nn.Sequential(*model)
        ### upsample
        model = []         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        self.decoder = nn.Sequential(*model)

        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)
            
    def forward(self, input):
        print (input.shape, 'input')
        encoded = self.encoder(input)
        print (encoded.shape, "encoded")
        encoded = self.resblocks(encoded)
        print (encoded.shape, "encoded")
        decoded = self.decoder(encoded)
        print (decoded.shape, "decoded")
        output = self.output_layer(decoded)
        print (output.shape, "output")
        return output   


class DefultGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(DefultGlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
