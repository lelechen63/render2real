import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_facescape import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_fewshot':
        from .pix2pixHD_facescape import Pix2PixHDFewshotModel, InferenceFewshotModel
        if opt.isTrain:
            model = Pix2PixHDFewshotModel()
        else:
            model = InferenceFewshotModel()
    
    elif opt.model == 'step1':
        from .DisentNet import DisentNet
        if opt.isTrain:
            model = DisentNet()
        else:
            model = InferenceDisentNet()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
